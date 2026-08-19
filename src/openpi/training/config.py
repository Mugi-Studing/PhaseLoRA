"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter

# PyTorch training needs a converted base checkpoint. Keep its location out of
# source control and let each machine provide it through the environment (or
# override `--pytorch-weight-path` on the command line).
_PI0_BASE_PYTORCH_PATH = os.environ.get("OPENPI_PI0_BASE_CHECKPOINT")
_PI05_BASE_PYTORCH_PATH = os.environ.get("OPENPI_PI05_BASE_CHECKPOINT")


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True
    # Optional causal P/E labels and router action histories keyed by the
    # LeRobot sample index. Histories for ALOHA must already be expressed in
    # the same 14D adapted delta-action space produced by AlohaInputs followed
    # by DeltaActions; Normalize will then apply the training action statistics.
    coarse_fine_label_path: str | None = None
    coarse_fine_label_strict: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        if self.coarse_fine_label_path is not None:
            data_transforms = data_transforms.push(
                inputs=[
                    _transforms.InjectOfflineCoarseFineLabel(
                        self.coarse_fine_label_path,
                        index_key="index",
                        output_key="route_label",
                        strict=self.coarse_fine_label_strict,
                        only_when_actions_present=True,
                    )
                ]
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False
    condition_dropout_prob: float = 0.0
    condition_dropout_prompt: bool = False
    condition_dropout_image: bool = False
    condition_dropout_state: bool = False
    frequency_blur_prob: float = 0.0
    frequency_blur_min_cutoff_ratio: float = 0.08
    frequency_blur_max_cutoff_ratio: float = 1.0
    # Optional path to precomputed per-sample coarse/fine labels keyed by LeRobot sample index.
    coarse_fine_label_path: str | None = None
    # If true, fail fast when a sample index is missing from the label mapping.
    coarse_fine_label_strict: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "index": "index",
                        "episode_index": "episode_index",
                        "frame_index": "frame_index",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_input_transforms = []
        if self.coarse_fine_label_path is not None:
            data_input_transforms.append(
                _transforms.InjectOfflineCoarseFineLabel(
                    self.coarse_fine_label_path,
                    index_key="index",
                    output_key="route_label",
                    strict=self.coarse_fine_label_strict,
                    only_when_actions_present=True,
                )
            )
        data_input_transforms.append(libero_policy.LiberoInputs(model_type=model_config.model_type))

        data_transforms = _transforms.Group(
            inputs=data_input_transforms,
            outputs=[libero_policy.LiberoOutputs()],
        )

        if self.condition_dropout_prob > 0.0:
            data_transforms = data_transforms.push(
                inputs=[
                    _transforms.ConditionDropout(
                        prob=self.condition_dropout_prob,
                        drop_prompt=self.condition_dropout_prompt,
                        drop_image=self.condition_dropout_image,
                        drop_state=self.condition_dropout_state,
                        only_when_actions_present=True,
                    )
                ]
            )

        if self.frequency_blur_prob > 0.0:
            data_transforms = data_transforms.push(
                inputs=[
                    _transforms.FrequencyBlurImages(
                        prob=self.frequency_blur_prob,
                        min_cutoff_ratio=self.frequency_blur_min_cutoff_ratio,
                        max_cutoff_ratio=self.frequency_blur_max_cutoff_ratio,
                        only_when_actions_present=True,
                    )
                ]
            )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None
    # Fail instead of silently training from partially loaded PyTorch weights.
    strict_pytorch_weight_loading: bool = False
    # Shard AdamW optimizer state across DDP ranks (ZeRO stage 1).
    # Full-parameter PI0/PI05 fine-tuning generally requires this on 32 GB GPUs.
    shard_optimizer: bool = False

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # If true, PyTorch training will skip loading pretrained weights for the entire action branch
    # and will train that branch from scratch. This is only used by `scripts/train_pytorch.py`.
    train_action_from_scratch: bool = False

    # If true, PyTorch training clips gradients only for the action branch (instead of all trainable params).
    # This is useful when the action branch is trained from scratch while the VLM side is LoRA-finetuned.
    action_expert_only_grad_clip: bool = False
    # Multiplier used to derive action-branch clip threshold from VLM grad norm:
    # clip_threshold = action_expert_grad_clip_multiplier * vlm_grad_norm.
    action_expert_grad_clip_multiplier: float = 2.0

    # Consistency regularization weights for action denoising in PyTorch training.
    # Defaults keep this regularizer disabled to preserve historical behavior.
    direction_consistency_weight: float = 0.0
    magnitude_consistency_weight: float = 0.0
    # Numerical stability epsilon used in cosine-style consistency terms.
    consistency_loss_eps: float = 1e-6
    # If true, scales consistency loss with w(t)=4t(1-t) over diffusion time.
    consistency_time_weighting: bool = False

    # Phase-gating regularization for multi-bank LoRA routing (PyTorch PI0/PI05 only).
    # Set `model.phase_gating=True` to enable routing in the model architecture.
    phase_gating_loss_weight: float = 0.0
    phase_gating_ce_weight: float = 1.0
    phase_gating_balance_weight: float = 0.0
    phase_gating_entropy_weight: float = 0.0
    phase_gating_consistency_weight: float = 0.0
    # Confidence threshold for weak pseudo-label supervision.
    phase_gating_confidence_threshold: float = 0.75
    # Target normalized entropy in [0, 1] for anti-collapse regularization.
    phase_gating_entropy_target: float = 0.7
    # Stddev of hidden noise used by router consistency regularization.
    phase_gating_noise_std: float = 0.05
    # Temperature for Gumbel-Softmax route sampling during training.
    phase_gating_temperature: float = 1.0
    # If true, use straight-through Gumbel-Softmax for hard route sampling in training.
    # If false, use temperature-soft routing in training and keep hard top-1 routing for inference.
    phase_gating_use_gumbel_st: bool = False
    # Number of near-term action steps used to generate weak phase pseudo labels.
    phase_gating_pseudo_window: int = 3

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        pytorch_weight_path=_PI0_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        num_train_steps=30_000,
    ),
    # Inference-only config for the converted pi05_base PyTorch checkpoint.
    # That checkpoint contains zero-initialized LoRA wrappers with ranks 16 (VLM)
    # and 32 (action expert), so these variants are required for exact key/shape matching.
    TrainConfig(
        name="pi05_libero_base_zeroshot",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora_16",
            action_expert_variant="gemma_300m_lora_32",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
    ),
    # Dense (non-LoRA), full-parameter PI0.5 fine-tuning configs for the four
    # standard LIBERO suites. The PyTorch loader folds the base checkpoint's
    # LoRA wrappers into the equivalent dense transformer weights.
    TrainConfig(
        name="pi05_libero_spatial_full_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_spatial",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        strict_pytorch_weight_loading=True,
        shard_optimizer=True,
        num_train_steps=30_000,
        save_interval=5_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_libero_object_full_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_object",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        strict_pytorch_weight_loading=True,
        shard_optimizer=True,
        num_train_steps=30_000,
        save_interval=5_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_libero_goal_full_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_goal",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        strict_pytorch_weight_loading=True,
        shard_optimizer=True,
        num_train_steps=30_000,
        save_interval=5_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_libero_10_full_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_10",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        strict_pytorch_weight_loading=True,
        shard_optimizer=True,
        num_train_steps=30_000,
        save_interval=5_000,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_libero_low_mem_finetune1",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi05_libero_spatial_vlm_lora_action_full_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_spatial",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi0_libero_spatial_vlm_lora_action_full_finetune",
        model=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_spatial",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI0_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi05_libero_spatial_lora_sp_baseline_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            lora_sp_enabled=True,
            lora_sp_rank=None,
            lora_sp_energy_threshold=0.9,
            lora_sp_router_hidden_dim=256,
            lora_sp_router_activation="silu",
            lora_sp_router_nonnegative="softplus",
            lora_sp_spec_loss_weight=1e-2,
            lora_sp_router_loss_weight=1e-3,
            lora_sp_router_balance_weight=1.0,
            lora_sp_router_z_loss_weight=1.0,
            lora_sp_inference_prune=True,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_spatial",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            lora_sp_enabled=True,
            lora_sp_rank=None,
            lora_sp_energy_threshold=0.9,
            lora_sp_router_hidden_dim=256,
            lora_sp_router_activation="silu",
            lora_sp_router_nonnegative="softplus",
            lora_sp_spec_loss_weight=1e-2,
            lora_sp_router_loss_weight=1e-3,
            lora_sp_router_balance_weight=1.0,
            lora_sp_router_z_loss_weight=1.0,
            lora_sp_inference_prune=True,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi05_libero_10_lora_sp_baseline_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            lora_sp_enabled=True,
            lora_sp_rank=None,
            lora_sp_energy_threshold=0.9,
            lora_sp_router_hidden_dim=256,
            lora_sp_router_activation="silu",
            lora_sp_router_nonnegative="softplus",
            lora_sp_spec_loss_weight=1e-2,
            lora_sp_router_loss_weight=1e-3,
            lora_sp_router_balance_weight=1.0,
            lora_sp_router_z_loss_weight=1.0,
            lora_sp_inference_prune=True,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_10",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            lora_sp_enabled=True,
            lora_sp_rank=None,
            lora_sp_energy_threshold=0.9,
            lora_sp_router_hidden_dim=256,
            lora_sp_router_activation="silu",
            lora_sp_router_nonnegative="softplus",
            lora_sp_spec_loss_weight=1e-2,
            lora_sp_router_loss_weight=1e-3,
            lora_sp_router_balance_weight=1.0,
            lora_sp_router_z_loss_weight=1.0,
            lora_sp_inference_prune=True,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi05_libero_goal_lora_sp_baseline_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            lora_sp_enabled=True,
            lora_sp_rank=None,
            lora_sp_energy_threshold=0.9,
            lora_sp_router_hidden_dim=256,
            lora_sp_router_activation="silu",
            lora_sp_router_nonnegative="softplus",
            lora_sp_spec_loss_weight=1e-2,
            lora_sp_router_loss_weight=1e-3,
            lora_sp_router_balance_weight=1.0,
            lora_sp_router_z_loss_weight=1.0,
            lora_sp_inference_prune=True,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_goal",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            lora_sp_enabled=True,
            lora_sp_rank=None,
            lora_sp_energy_threshold=0.9,
            lora_sp_router_hidden_dim=256,
            lora_sp_router_activation="silu",
            lora_sp_router_nonnegative="softplus",
            lora_sp_spec_loss_weight=1e-2,
            lora_sp_router_loss_weight=1e-3,
            lora_sp_router_balance_weight=1.0,
            lora_sp_router_z_loss_weight=1.0,
            lora_sp_inference_prune=True,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi05_libero_object_lora_sp_baseline_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            lora_sp_enabled=True,
            lora_sp_rank=None,
            lora_sp_energy_threshold=0.9,
            lora_sp_router_hidden_dim=256,
            lora_sp_router_activation="silu",
            lora_sp_router_nonnegative="softplus",
            lora_sp_spec_loss_weight=1e-2,
            lora_sp_router_loss_weight=1e-3,
            lora_sp_router_balance_weight=1.0,
            lora_sp_router_z_loss_weight=1.0,
            lora_sp_inference_prune=True,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_object",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            lora_sp_enabled=True,
            lora_sp_rank=None,
            lora_sp_energy_threshold=0.9,
            lora_sp_router_hidden_dim=256,
            lora_sp_router_activation="silu",
            lora_sp_router_nonnegative="softplus",
            lora_sp_spec_loss_weight=1e-2,
            lora_sp_router_loss_weight=1e-3,
            lora_sp_router_balance_weight=1.0,
            lora_sp_router_z_loss_weight=1.0,
            lora_sp_inference_prune=True,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi05_libero_10_vlm_lora_action_full_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_10",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=30,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi05_libero_goal_vlm_lora_action_full_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_goal",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=28,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi05_libero_object_vlm_lora_action_full_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_object",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=28,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi05_libero_spatial_single_lora_tau_gated_router_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora_120",
            action_expert_variant="gemma_300m_lora_240",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_spatial",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            coarse_fine_label_path="./assets/pi05_libero_spatial_vlm_lora_action_full_finetune/libero_spatial_coarse_fine_pe_prefix5_w5_q30_local.json",
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora_120",
            action_expert_variant="gemma_300m_lora_240",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    # PhaseLoRA ablation: keep the complete P/E router and routing path, but remove
    # the auxiliary label-supervision gradients for both P and E. The offline label
    # file is retained because it also provides the past-safe router action history.
    TrainConfig(
        name="pi05_libero_spatial_phaselora_no_pe_supervision_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora_120",
            action_expert_variant="gemma_300m_lora_240",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            coarse_fine_p_loss_weight=0.0,
            coarse_fine_e_loss_weight=0.0,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_spatial",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            coarse_fine_label_path="./assets/pi05_libero_spatial_vlm_lora_action_full_finetune/libero_spatial_coarse_fine_pe_prefix5_w5_q30_local.json",
        ),
        batch_size=16,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora_120",
            action_expert_variant="gemma_300m_lora_240",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            coarse_fine_p_loss_weight=0.0,
            coarse_fine_e_loss_weight=0.0,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    # PhaseLoRA temporal-signal control. Architecture, router inputs, losses,
    # optimizer, and B0/BP/BE/BPE routing are identical to the P/E-supervised
    # baseline; only the targets are replaced by S1=tau and S2=4*tau*(1-tau).
    TrainConfig(
        name="pi05_libero_spatial_phaselora_temporal_signal_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora_120",
            action_expert_variant="gemma_300m_lora_240",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            coarse_fine_p_loss_weight=0.5,
            coarse_fine_e_loss_weight=0.2,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_spatial",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            coarse_fine_label_path="./assets/pi05_libero_spatial_vlm_lora_action_full_finetune/libero_spatial_temporal_signal_s1_s2_prefix5_history6x5.json",
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora_120",
            action_expert_variant="gemma_300m_lora_240",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            coarse_fine_p_loss_weight=0.5,
            coarse_fine_e_loss_weight=0.2,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    # Privileged symbolic supervision control. The simulator is used only by the
    # offline label builder. Architecture, router inputs, P/E losses, optimizer,
    # and B0/BP/BE/BPE parameterization match the weak-P/E PhaseLoRA baseline.
    TrainConfig(
        name="pi05_libero_spatial_phaselora_symbolic_supervision_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora_120",
            action_expert_variant="gemma_300m_lora_240",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            coarse_fine_p_loss_weight=0.5,
            coarse_fine_e_loss_weight=0.2,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_spatial",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            coarse_fine_label_path="./assets/pi05_libero_spatial_vlm_lora_action_full_finetune/libero_spatial_symbolic_phase_pe_h4_w5.json",
        ),
        batch_size=28,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora_120",
            action_expert_variant="gemma_300m_lora_240",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            coarse_fine_p_loss_weight=0.5,
            coarse_fine_e_loss_weight=0.2,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    # pi0 variants for object / goal / libero_10 suites (mirrors pi05 entries)
    TrainConfig(
        name="pi0_libero_10_vlm_lora_action_full_finetune",
        model=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_10",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI0_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi0_libero_goal_vlm_lora_action_full_finetune",
        model=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_goal",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI0_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi0_libero_object_vlm_lora_action_full_finetune",
        model=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_object",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI0_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi0_libero_10_single_lora_tau_gated_router_finetune",
        model=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora_120",
            action_expert_variant="gemma_300m_lora_240",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_10",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            coarse_fine_label_path="./assets/pi05_libero_10_low_mem_finetune/libero_10_coarse_fine_pe_prefix5_w5_q30_local.json",
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=40_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI0_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    TrainConfig(
        name="pi0_libero_goal_single_lora_tau_gated_router_finetune",
        model=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_goal",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            coarse_fine_label_path="./assets/pi05_libero_goal_single_lora_tau_gated_router_finetune/libero_goal_coarse_fine_pe_prefix5_w5_q30_local.json",
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI0_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    TrainConfig(
        name="pi0_libero_object_single_lora_tau_gated_router_finetune",
        model=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_object",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            coarse_fine_label_path="./assets/pi05_libero_object_single_lora_tau_gated_router_finetune/libero_object_coarse_fine_pe_prefix5_w5_q30_local.json",
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI0_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    TrainConfig(
        name="pi0_libero_spatial_single_lora_tau_gated_router_finetune",
        model=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_spatial",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            coarse_fine_label_path="./assets/pi05_libero_spatial_vlm_lora_action_full_finetune/libero_spatial_coarse_fine_pe_prefix5_w5_q30_local.json",
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=False,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI0_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    TrainConfig(
        name="pi05_libero_spatial_single_lora_tau_gated_router_ablation_bp_e_be_e_bpe_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            action_expert_pe_route_mode="bp_e_ebpe",
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_spatial",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            coarse_fine_label_path="./assets/pi05_libero_spatial_vlm_lora_action_full_finetune/libero_spatial_coarse_fine_pe_prefix5_w5_q30_local.json",
        ),
        batch_size=28,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            action_expert_pe_route_mode="bp_e_ebpe",
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    TrainConfig(
        name="pi05_libero_spatial_single_lora_tau_gated_router_ablation_p_bp_be_p_bpe_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            action_expert_pe_route_mode="pbp_be_pbpe",
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_spatial",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            coarse_fine_label_path="./assets/pi05_libero_spatial_vlm_lora_action_full_finetune/libero_spatial_coarse_fine_pe_prefix5_w5_q30_local.json",
        ),
        batch_size=28,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            action_expert_pe_route_mode="pbp_be_pbpe",
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    TrainConfig(
        name="pi05_libero_spatial_single_lora_tau_gated_router_ablation_p_bp_e_be_bpe_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            action_expert_pe_route_mode="pbp_ebe_bpe",
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_spatial",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            coarse_fine_label_path="./assets/pi05_libero_spatial_vlm_lora_action_full_finetune/libero_spatial_coarse_fine_pe_prefix5_w5_q30_local.json",
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            action_expert_pe_route_mode="pbp_ebe_bpe",
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    TrainConfig(
        name="pi05_libero_10_single_lora_tau_gated_router_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            # This checkpoint was trained before the generic LoRA aliases were
            # retuned. Keep its historical VLM / action-expert ranks explicit.
            paligemma_variant="gemma_2b_lora_120",
            action_expert_variant="gemma_300m_lora_240",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_10",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            coarse_fine_label_path="./assets/pi05_libero_10_low_mem_finetune/libero_10_coarse_fine_pe_prefix5_w5_q30_local.json",
        ),
        batch_size=33,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora_120",
            action_expert_variant="gemma_300m_lora_240",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    TrainConfig(
        name="pi05_libero_goal_single_lora_tau_gated_router_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_goal",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            coarse_fine_label_path="./assets/pi05_libero_goal_single_lora_tau_gated_router_finetune/libero_goal_coarse_fine_pe_prefix5_w5_q30_local.json",
        ),
        batch_size=28,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    TrainConfig(
        name="pi05_libero_object_single_lora_tau_gated_router_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_object",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            coarse_fine_label_path="./assets/pi05_libero_object_single_lora_tau_gated_router_finetune/libero_object_coarse_fine_pe_prefix5_w5_q30_local.json",
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_rot_weight=0.5,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    TrainConfig(
        name="pi05_libero_low_mem_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            condition_dropout_prob=0.2,
            condition_dropout_prompt=False,
            condition_dropout_image=True,
            condition_dropout_state=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi05_libero_low_mem_finetune_regression",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_training_type="regression",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            condition_dropout_prob=0.1,
            condition_dropout_prompt=False,
            condition_dropout_image=True,
            condition_dropout_state=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_training_type="regression",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi05_libero_10_low_mem_finetune_action_scratch",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_10",
            assets=AssetsConfig(
                # Reuse the norm stats computed for the original Libero-10 low-memory config.
                assets_dir="./assets/pi05_libero_10_low_mem_finetune",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=28,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        train_action_from_scratch=False,
        action_expert_only_grad_clip=False,
        action_expert_grad_clip_multiplier=2.0,
    ),
    TrainConfig(
        name="pi05_libero_10_low_mem_finetune_action_scratch_regression",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            action_training_type="regression",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="your_hf_username/libero_libero_10",
            assets=AssetsConfig(
                # Reuse the norm stats computed for the original Libero-10 low-memory config.
                assets_dir="./assets/pi05_libero_10_low_mem_finetune",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            action_training_type="regression",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        train_action_from_scratch=True,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap_lora_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
    ),
    # Parameter-count-matched pure-LoRA baseline for the bimanual PhaseLoRA
    # experiment below. The VLM uses rank 48 and the action expert uses rank
    # 624, giving approximately the same number of trainable adapter parameters
    # as rank-240 B0/BP/BE/BPE PhaseLoRA plus its P/E router.
    TrainConfig(
        name="pi05_aloha_pen_uncap_bimanual_lora_rank48_624_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora_48",
            action_expert_variant="gemma_300m_lora_624",
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora_48",
            action_expert_variant="gemma_300m_lora_624",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap_bimanual_phaselora_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora_120",
            action_expert_variant="gemma_300m_lora_240",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            coarse_fine_action_layout="aloha_bimanual_14d",
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            coarse_fine_label_path=(
                "./assets/pi05_aloha_pen_uncap_bimanual_phaselora_finetune/"
                "aloha_pen_uncap_bimanual_pe_prefix5_h6x5.json"
            ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "index": "index",
                            "episode_index": "episode_index",
                            "frame_index": "frame_index",
                        }
                    )
                ]
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora_120",
            action_expert_variant="gemma_300m_lora_240",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            coarse_fine_action_layout="aloha_bimanual_14d",
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    # ALOHA TransferCube comparison with a 1:2 VLM/action-expert rank ratio.
    #
    # The PhaseLoRA action expert uses one shared A and eight B matrices:
    # B0, three arm-specific matrices per arm (P/E/PE), and one coordination
    # matrix. Pure-LoRA ranks (122, 244) match the trainable adapter parameter
    # count of bimanual PhaseLoRA ranks (48, 96), up to the small router.
    TrainConfig(
        name="pi05_aloha_sim_transfer_cube_lora_rank122_244_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_122",
            action_expert_variant="gemma_300m_lora_244",
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Pick up the cube with the right arm and transfer it to the left arm.",
            use_delta_joint_actions=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_122",
            action_expert_variant="gemma_300m_lora_244",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
    ),
    # High-capacity pure-LoRA ablation. This intentionally does not preserve
    # parameter matching with the bimanual PhaseLoRA baseline. It reuses the
    # exact normalization statistics of the rank-122/244 run so that only
    # adapter capacity and training duration change.
    TrainConfig(
        name="pi05_aloha_sim_transfer_cube_lora_rank512_1024_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_512",
            action_expert_variant="gemma_300m_lora_1024",
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_aloha_sim_transfer_cube_lora_rank122_244_finetune",
                asset_id="lerobot/aloha_sim_transfer_cube_human",
            ),
            default_prompt="Pick up the cube with the right arm and transfer it to the left arm.",
            use_delta_joint_actions=False,
        ),
        # Keep the global batch identical to the rank-122/244 baseline so a
        # training step represents the same number of examples. The adapters
        # contain about 1.07B trainable parameters, so shard Adam states across
        # DDP ranks with the mixed-dtype ZeRO-1 wrapper.
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        shard_optimizer=True,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_512",
            action_expert_variant="gemma_300m_lora_1024",
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
    ),
    TrainConfig(
        name="pi05_aloha_sim_transfer_cube_bimanual_phaselora_rank48_96_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_48",
            action_expert_variant="gemma_300m_lora_96",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_bimanual_routing=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            coarse_fine_action_layout="aloha_bimanual_14d",
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            # Share exactly the same normalization statistics as the pure-LoRA
            # baseline; these are computed once with compute_norm_stats.py.
            assets=AssetsConfig(
                assets_dir="./assets/pi05_aloha_sim_transfer_cube_lora_rank122_244_finetune",
                asset_id="lerobot/aloha_sim_transfer_cube_human",
            ),
            default_prompt="Pick up the cube with the right arm and transfer it to the left arm.",
            use_delta_joint_actions=False,
            coarse_fine_label_path=(
                "./assets/pi05_aloha_sim_transfer_cube_bimanual_phaselora_finetune/"
                "aloha_transfer_cube_bimanual_coord_pe_absolute_prefix5_h6x5.json"
            ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {"cam_high": "observation.images.top"},
                            "state": "observation.state",
                            "actions": "action",
                            "index": "index",
                            "episode_index": "episode_index",
                            "frame_index": "frame_index",
                        }
                    )
                ]
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_48",
            action_expert_variant="gemma_300m_lora_96",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_bimanual_routing=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            coarse_fine_action_layout="aloha_bimanual_14d",
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    # High-capacity, parameter-matched bimanual PhaseLoRA ablation. With one
    # shared A and the arm/coordination-specific B matrices, ranks 201/403 give
    # 1,071,262,213 trainable parameters versus 1,071,120,384 for pure LoRA
    # ranks 512/1024 (a 0.0132% difference).
    TrainConfig(
        name="pi05_aloha_sim_transfer_cube_bimanual_phaselora_rank201_403_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_201",
            action_expert_variant="gemma_300m_lora_403",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_bimanual_routing=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            coarse_fine_action_layout="aloha_bimanual_14d",
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_aloha_sim_transfer_cube_lora_rank122_244_finetune",
                asset_id="lerobot/aloha_sim_transfer_cube_human",
            ),
            default_prompt="Pick up the cube with the right arm and transfer it to the left arm.",
            use_delta_joint_actions=False,
            coarse_fine_label_path=(
                "./assets/pi05_aloha_sim_transfer_cube_bimanual_phaselora_finetune/"
                "aloha_transfer_cube_bimanual_coord_pe_absolute_prefix5_h6x5.json"
            ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {"cam_high": "observation.images.top"},
                            "state": "observation.state",
                            "actions": "action",
                            "index": "index",
                            "episode_index": "episode_index",
                            "frame_index": "frame_index",
                        }
                    )
                ]
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        shard_optimizer=True,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_201",
            action_expert_variant="gemma_300m_lora_403",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_bimanual_routing=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            coarse_fine_action_layout="aloha_bimanual_14d",
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    # High-capacity bimanual PhaseLoRA ablation using the same raw ranks and
    # training duration as the rank-512/1024 pure-LoRA run. Because the action
    # expert contains multiple phase/coordination B matrices, this is not a
    # parameter-matched comparison: it has about 2.72B trainable parameters.
    TrainConfig(
        name="pi05_aloha_sim_transfer_cube_bimanual_phaselora_rank512_1024_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_512",
            action_expert_variant="gemma_300m_lora_1024",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_bimanual_routing=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            coarse_fine_action_layout="aloha_bimanual_14d",
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_gripper_weight=0.1,
        ),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_aloha_sim_transfer_cube_lora_rank122_244_finetune",
                asset_id="lerobot/aloha_sim_transfer_cube_human",
            ),
            default_prompt="Pick up the cube with the right arm and transfer it to the left arm.",
            use_delta_joint_actions=False,
            coarse_fine_label_path=(
                "./assets/pi05_aloha_sim_transfer_cube_bimanual_phaselora_finetune/"
                "aloha_transfer_cube_bimanual_coord_pe_absolute_prefix5_h6x5.json"
            ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {"cam_high": "observation.images.top"},
                            "state": "observation.state",
                            "actions": "action",
                            "index": "index",
                            "episode_index": "episode_index",
                            "frame_index": "frame_index",
                        }
                    )
                ]
            ),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        shard_optimizer=True,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora_512",
            action_expert_variant="gemma_300m_lora_1024",
            phase_gating=False,
            coarse_fine_shared_routing=False,
            coarse_fine_rank_gating=True,
            coarse_fine_bimanual_routing=True,
            coarse_fine_score_label=True,
            coarse_fine_label_prefix_steps=5,
            coarse_fine_router_history_steps=6,
            coarse_fine_router_chunk_prefix_steps=5,
            coarse_fine_action_layout="aloha_bimanual_14d",
            lora_bank_count=1,
            vlm_lora_bank_count=1,
            action_expert_lora_bank_count=1,
            coarse_fine_history_window=5,
            coarse_fine_quantile=0.3,
            coarse_fine_min_history=5,
            coarse_fine_hysteresis=2,
            coarse_fine_gripper_weight=0.1,
        ).get_freeze_filter(),
        ema_decay=None,
        pytorch_weight_path=_PI05_BASE_PYTORCH_PATH,
        phase_gating_loss_weight=0.0,
        phase_gating_ce_weight=0.0,
        phase_gating_balance_weight=0.0,
        phase_gating_entropy_weight=0.0,
        phase_gating_consistency_weight=0.0,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
