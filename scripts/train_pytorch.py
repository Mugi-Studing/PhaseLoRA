"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import datetime
import gc
import hashlib
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


def get_action_branch_parameter_prefixes(config: _config.TrainConfig) -> tuple[str, ...]:
    """Return parameter prefixes belonging to the action branch for PyTorch training."""
    prefixes = [
        "paligemma_with_expert.gemma_expert.",
        "action_in_proj.",
        "action_out_proj.",
    ]
    if getattr(config.model, "pi05", False):
        prefixes.extend(["time_mlp_in.", "time_mlp_out."])
    else:
        prefixes.extend(["state_proj.", "action_time_mlp_in.", "action_time_mlp_out."])
    return tuple(prefixes)


def is_action_branch_parameter(name: str, config: _config.TrainConfig) -> bool:
    """Check whether a parameter belongs to the action branch."""
    return name.startswith(get_action_branch_parameter_prefixes(config))


def is_phase_router_parameter(name: str) -> bool:
    """Check whether a parameter belongs to any routing head/trunk."""
    router_tokens = ("phase_router_", "tau_router_")
    return name.startswith(router_tokens) or any(token in name for token in router_tokens)


def strip_ddp_prefix(name: str) -> str:
    """Normalize parameter names to work with both wrapped and unwrapped modules."""
    return name.removeprefix("module.")


class MixedDtypeZeroRedundancyOptimizer:
    """One ZeRO optimizer per parameter dtype, exposed as a single optimizer.

    ``ZeroRedundancyOptimizer`` rejects a parameter list containing both
    FloatTensor and BFloat16Tensor. PI0.5 intentionally keeps a small subset of
    numerically sensitive parameters in float32, so splitting by dtype preserves
    model precision while still sharding all AdamW states.
    """

    _STATE_FORMAT = "openpi_mixed_dtype_zero_v1"

    def __init__(
        self,
        parameters: list[torch.nn.Parameter],
        *,
        optimizer_class: type[torch.optim.Optimizer],
        **optimizer_kwargs,
    ) -> None:
        parameters_by_dtype: dict[torch.dtype, list[torch.nn.Parameter]] = {}
        for parameter in parameters:
            parameters_by_dtype.setdefault(parameter.dtype, []).append(parameter)

        if not parameters_by_dtype:
            raise ValueError("Cannot create an optimizer with no parameters.")

        self._dtype_keys = sorted(parameters_by_dtype, key=str)
        self._optimizers = [
            ZeroRedundancyOptimizer(
                parameters_by_dtype[dtype],
                optimizer_class=optimizer_class,
                **optimizer_kwargs,
            )
            for dtype in self._dtype_keys
        ]
        self.parameter_counts = {
            str(dtype): sum(parameter.numel() for parameter in parameters_by_dtype[dtype]) for dtype in self._dtype_keys
        }

    @property
    def param_groups(self) -> list[dict]:
        return [param_group for optimizer in self._optimizers for param_group in optimizer.param_groups]

    def step(self) -> None:
        for optimizer in self._optimizers:
            optimizer.step()

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for optimizer in self._optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def consolidate_state_dict(self, to: int = 0) -> None:
        for optimizer in self._optimizers:
            optimizer.consolidate_state_dict(to=to)

    def state_dict(self) -> dict[str, object]:
        return {
            "format": self._STATE_FORMAT,
            "dtype_keys": [str(dtype) for dtype in self._dtype_keys],
            "optimizer_state_dicts": [optimizer.state_dict() for optimizer in self._optimizers],
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        if state_dict.get("format") != self._STATE_FORMAT:
            raise ValueError(
                f"Optimizer checkpoint is not a mixed-dtype ZeRO state dict. Found format={state_dict.get('format')!r}."
            )
        expected_dtype_keys = [str(dtype) for dtype in self._dtype_keys]
        checkpoint_dtype_keys = state_dict.get("dtype_keys")
        if checkpoint_dtype_keys != expected_dtype_keys:
            raise ValueError(
                "Optimizer dtype groups differ from checkpoint: "
                f"expected={expected_dtype_keys}, got={checkpoint_dtype_keys}."
            )
        optimizer_state_dicts = state_dict.get("optimizer_state_dicts")
        if not isinstance(optimizer_state_dicts, list):
            raise TypeError("optimizer_state_dicts must be a list.")
        if len(optimizer_state_dicts) != len(self._optimizers):
            raise ValueError(
                "Optimizer group count differs from checkpoint: "
                f"expected={len(self._optimizers)}, got={len(optimizer_state_dicts)}."
            )
        for optimizer, optimizer_state_dict in zip(
            self._optimizers,
            optimizer_state_dicts,
            strict=True,
        ):
            optimizer.load_state_dict(optimizer_state_dict)


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    if torch.cuda.is_available() and use_ddp:
        visible_device_count = torch.cuda.device_count()
        if visible_device_count < world_size:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
            raise RuntimeError(
                "DDP configuration mismatch: WORLD_SIZE is larger than number of visible CUDA devices. "
                f"WORLD_SIZE={world_size}, visible_cuda_devices={visible_device_count}, "
                f"CUDA_VISIBLE_DEVICES={visible_devices}. "
                "Please make sure --nproc_per_node <= number of visible GPUs and use a valid "
                "CUDA_VISIBLE_DEVICES list (e.g. '0,1,2,3')."
            )

    # Bind CUDA device after validating the visible-device/world-size mapping,
    # so an invalid local rank produces the actionable error above.
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        # Make NCCL hangs fail fast with actionable errors instead of silent stalls.
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        os.environ.setdefault("NCCL_DEBUG", "WARN")

        init_kwargs = {
            "backend": backend,
            "init_method": "env://",
            "timeout": datetime.timedelta(minutes=10),
        }
        if backend == "nccl" and torch.cuda.is_available():
            init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")

        try:
            torch.distributed.init_process_group(**init_kwargs)
        except TypeError:
            init_kwargs.pop("device_id", None)
            torch.distributed.init_process_group(**init_kwargs)

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def get_unwrapped_model(model):
    """Get the underlying model, handling DDP wrapper."""
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def synchronize_trainable_mask_across_ranks(model: torch.nn.Module, device: torch.device) -> None:
    """Synchronize requires_grad mask across ranks, using rank 0 as source of truth."""
    if not torch.distributed.is_initialized():
        return

    if device.type == "cuda":
        if device.index is None:
            raise RuntimeError("CUDA device index must be set when synchronizing trainable masks in DDP.")
        torch.cuda.set_device(device.index)

    named_parameters = list(model.named_parameters())
    local_mask = [1 if param.requires_grad else 0 for _, param in named_parameters]

    def _name_fingerprint(names: list[str]) -> tuple[int, int]:
        hasher = hashlib.blake2b(digest_size=16)
        for name in names:
            encoded = name.encode("utf-8")
            hasher.update(len(encoded).to_bytes(4, byteorder="little", signed=False))
            hasher.update(encoded)
        digest = hasher.digest()
        return (
            int.from_bytes(digest[:8], byteorder="little", signed=True),
            int.from_bytes(digest[8:], byteorder="little", signed=True),
        )

    local_names = [name for name, _ in named_parameters]
    fp0, fp1 = _name_fingerprint(local_names)

    collective_device = device if device.type == "cuda" else torch.device("cpu")
    world_size = torch.distributed.get_world_size()
    metadata = torch.tensor([len(local_names), fp0, fp1], dtype=torch.int64, device=collective_device)
    gathered_metadata = [torch.empty_like(metadata) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_metadata, metadata)

    ref_len, ref_fp0, ref_fp1 = gathered_metadata[0].tolist()
    for rank_idx, rank_metadata in enumerate(gathered_metadata):
        rank_len, rank_fp0, rank_fp1 = rank_metadata.tolist()
        if rank_len != ref_len or rank_fp0 != ref_fp0 or rank_fp1 != ref_fp1:
            raise RuntimeError(
                "Model parameter name ordering differs across ranks before DDP wrap; "
                f"rank0 metadata=(count={ref_len}, fp0={ref_fp0}, fp1={ref_fp1}), "
                f"rank{rank_idx} metadata=(count={rank_len}, fp0={rank_fp0}, fp1={rank_fp1})"
            )

    expected_count = int(ref_len)
    if len(local_mask) != expected_count:
        raise RuntimeError(
            "Local trainable mask length mismatch before synchronization: "
            f"local_count={len(local_mask)}, expected_count={expected_count}."
        )

    mask_tensor = torch.tensor(local_mask, dtype=torch.int32, device=collective_device)
    torch.distributed.broadcast(mask_tensor, src=0)

    for (_, param), keep_trainable in zip(named_parameters, mask_tensor.tolist(), strict=True):
        param.requires_grad = bool(keep_trainable)


def configure_trainable_parameters(model, config: _config.TrainConfig) -> list[torch.nn.Parameter]:
    """Configure trainable parameters based on the training config.

    For LoRA fine-tuning configs, only LoRA parameters are trainable to match low-memory behavior.
    For non-LoRA configs, all parameters are trainable.
    """
    model_cfg = config.model
    paligemma_variant = str(getattr(model_cfg, "paligemma_variant", ""))
    action_expert_variant = str(getattr(model_cfg, "action_expert_variant", ""))
    is_lora_finetune = ("lora" in paligemma_variant) or ("lora" in action_expert_variant)

    train_action_from_scratch = config.train_action_from_scratch
    phase_gating_enabled = bool(getattr(model_cfg, "phase_gating", False))
    rank_gating_enabled = bool(getattr(model_cfg, "coarse_fine_rank_gating", False))
    named_parameters = list(model.named_parameters())
    if not is_lora_finetune:
        for _, param in named_parameters:
            param.requires_grad = True
        trainable_parameters = [param for _, param in named_parameters if param.requires_grad]
        trainable_count = sum(param.numel() for param in trainable_parameters)
        total_count = sum(param.numel() for _, param in named_parameters)
        logging.info(
            f"Training all parameters: trainable={trainable_count:,} / total={total_count:,} "
            f"({100.0 * trainable_count / max(1, total_count):.2f}%)"
        )
        return trainable_parameters

    lora_named_parameters = [(name, param) for name, param in named_parameters if "lora" in name.lower()]
    total_count = sum(param.numel() for _, param in named_parameters)

    if lora_named_parameters:
        # LoRA mode: freeze all non-LoRA parameters, unless the config explicitly
        # requests that the whole action branch be trained from scratch.
        # For phase-gating configs, phase router weights must remain trainable.
        for name, param in named_parameters:
            normalized_name = strip_ddp_prefix(name)
            is_lora_param = "lora" in normalized_name.lower()
            train_action_param = train_action_from_scratch and is_action_branch_parameter(normalized_name, config)
            train_router_param = (phase_gating_enabled or rank_gating_enabled) and is_phase_router_parameter(
                normalized_name
            )
            param.requires_grad = is_lora_param or train_action_param or train_router_param

        trainable_parameters = [param for _, param in named_parameters if param.requires_grad]
        trainable_count = sum(param.numel() for param in trainable_parameters)
        mode_parts = ["LoRA fine-tuning"]
        if train_action_from_scratch:
            mode_parts.append("scratch action branch")
        if phase_gating_enabled or rank_gating_enabled:
            mode_parts.append("routing heads")
        mode = " + ".join(mode_parts)
        router_trainable_count = sum(
            param.numel()
            for name, param in named_parameters
            if param.requires_grad and is_phase_router_parameter(strip_ddp_prefix(name))
        )
        logging.info(
            f"{mode} enabled: trainable={trainable_count:,} / total={total_count:,} "
            f"({100.0 * trainable_count / max(1, total_count):.4f}%), "
            f"router_trainable={router_trainable_count:,}"
        )
        return trainable_parameters

    logging.warning(
        "LoRA fine-tuning config detected, but no LoRA parameters were found in the current PyTorch model. "
        "Falling back to full-parameter training."
    )
    for _, param in named_parameters:
        param.requires_grad = True
    trainable_parameters = [param for _, param in named_parameters if param.requires_grad]
    trainable_count = sum(param.numel() for param in trainable_parameters)
    logging.info(
        f"Fallback training parameters: trainable={trainable_count:,} / total={total_count:,} "
        f"({100.0 * trainable_count / max(1, total_count):.2f}%)"
    )
    return trainable_parameters


def prepare_pytorch_checkpoint_for_model(
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
	"""Adapt a PyTorch checkpoint to the instantiated model.
	
	This handles both directions between dense Linear and LoRALinear modules:
	1. LoRALinear checkpoint -> dense model:
    W_dense = W_base + B @ A.
    2. Dense checkpoint -> LoRALinear model:
    load the dense weight into base_linear.weight and leave the newly
    initialized LoRA A/B parameters unchanged.

    OpenPI LoRA variants use alpha == rank, so the LoRA scaling factor is 1.
    """
    filtered_state_dict: dict[str, torch.Tensor] = {}
    consumed_source_keys: set[str] = set()
    shape_mismatch_keys: list[str] = []
    missing_target_keys: list[str] = []
    folded_lora_modules = 0
    zero_lora_modules = 0

    for target_key, target_value in model_state_dict.items():
        source_value = state_dict.get(target_key)
        if source_value is not None:
            consumed_source_keys.add(target_key)
            if tuple(source_value.shape) == tuple(target_value.shape):
                filtered_state_dict[target_key] = source_value
            else:
                shape_mismatch_keys.append(target_key)
            continue
        if target_key.endswith(".base_linear.weight"):
            module_prefix = target_key.removesuffix(".base_linear.weight")
            dense_key = f"{module_prefix}.weight"
            dense_value = state_dict.get(dense_key)

            if dense_value is not None:
                consumed_source_keys.add(dense_key)
                if tuple(dense_value.shape) == tuple(target_value.shape):
                    filtered_state_dict[target_key] = dense_value
                else:
                    shape_mismatch_keys.append(dense_key)
                continue

        if target_key.endswith(".base_linear.bias"):
            module_prefix = target_key.removesuffix(".base_linear.bias")
            dense_key = f"{module_prefix}.bias"
            dense_value = state_dict.get(dense_key)

            if dense_value is not None:
                consumed_source_keys.add(dense_key)
                if tuple(dense_value.shape) == tuple(target_value.shape):
                    filtered_state_dict[target_key] = dense_value
                else:
                    shape_mismatch_keys.append(dense_key)
                continue
        if target_key.endswith(".weight"):
            module_prefix = target_key[: -len(".weight")]
            base_key = f"{module_prefix}.base_linear.weight"
            base_value = state_dict.get(base_key)
            if base_value is not None:
                consumed_source_keys.add(base_key)
                if tuple(base_value.shape) != tuple(target_value.shape):
                    shape_mismatch_keys.append(base_key)
                    continue

                lora_a_key = f"{module_prefix}.lora_a.weight"
                lora_b_key = f"{module_prefix}.lora_b.weight"
                lora_a = state_dict.get(lora_a_key)
                lora_b = state_dict.get(lora_b_key)
                dense_value = base_value
                if lora_a is not None and lora_b is not None:
                    consumed_source_keys.update((lora_a_key, lora_b_key))
                    if int(torch.count_nonzero(lora_b)) == 0:
                        zero_lora_modules += 1
                    else:
                        delta = torch.matmul(lora_b.float(), lora_a.float()).to(dtype=base_value.dtype)
                        dense_value = base_value + delta
                        folded_lora_modules += 1
                filtered_state_dict[target_key] = dense_value
                continue

        if target_key.endswith(".bias"):
            module_prefix = target_key[: -len(".bias")]
            base_key = f"{module_prefix}.base_linear.bias"
            base_value = state_dict.get(base_key)
            if base_value is not None:
                consumed_source_keys.add(base_key)
                if tuple(base_value.shape) == tuple(target_value.shape):
                    filtered_state_dict[target_key] = base_value
                else:
                    shape_mismatch_keys.append(base_key)
                continue

        tied_embed_suffix = "paligemma.model.language_model.embed_tokens.weight"
        if target_key.endswith(tied_embed_suffix):
            tied_source_key = target_key[: -len(tied_embed_suffix)] + "paligemma.lm_head.weight"
            tied_source_value = state_dict.get(tied_source_key)
            if tied_source_value is not None:
                consumed_source_keys.add(tied_source_key)
                if tuple(tied_source_value.shape) == tuple(target_value.shape):
                    filtered_state_dict[target_key] = tied_source_value
                else:
                    shape_mismatch_keys.append(tied_source_key)
                continue

        missing_target_keys.append(target_key)

    ignored_source_keys = sorted(set(state_dict) - consumed_source_keys)
    report: dict[str, object] = {
        "shape_mismatch_keys": shape_mismatch_keys,
        "missing_target_keys": missing_target_keys,
        "ignored_source_keys": ignored_source_keys,
        "folded_lora_modules": folded_lora_modules,
        "zero_lora_modules": zero_lora_modules,
    }
    return filtered_state_dict, report


def _prepare_safetensors_state_dict(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Prepare a state dict that is safe for safetensors.save_file.

    Some modules (e.g. GRU weights on certain PyTorch builds) may expose tensors that
    either share storage or are views into larger storage blocks. safetensors refuses to
    serialize these layouts. We clone only problematic tensors to break such aliases.
    """
    raw_state_dict = model.state_dict()

    # Track storages we have already emitted to avoid shared-storage aliasing.
    seen_storages: set[tuple[int, str, int | None, str]] = set()
    safe_state_dict: dict[str, torch.Tensor] = {}

    cloned_shared = 0
    cloned_view = 0
    made_contiguous = 0

    for name, value in raw_state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue

        tensor = value.detach()
        needs_clone = False

        # If tensor is a view into a larger storage, clone it to own storage.
        try:
            storage = tensor.untyped_storage()
            storage_nbytes = int(storage.nbytes())
            storage_ptr = int(storage.data_ptr())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            storage = None
            storage_nbytes = tensor.numel() * tensor.element_size()
            storage_ptr = None

        tensor_nbytes = tensor.numel() * tensor.element_size()
        if storage_nbytes > tensor_nbytes:
            needs_clone = True
            cloned_view += 1

        storage_key = None
        if storage_ptr is not None:
            storage_key = (storage_ptr, tensor.device.type, tensor.device.index, str(tensor.dtype))
            if storage_key in seen_storages and not needs_clone:
                needs_clone = True
                cloned_shared += 1

        if needs_clone:
            tensor = tensor.clone()

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
            made_contiguous += 1

        # Register final tensor storage key (after potential clone).
        try:
            final_storage = tensor.untyped_storage()
            final_key = (
                int(final_storage.data_ptr()),
                tensor.device.type,
                tensor.device.index,
                str(tensor.dtype),
            )
            seen_storages.add(final_key)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

        safe_state_dict[name] = tensor

    stats = {
        "tensor_count": len(safe_state_dict),
        "cloned_shared": cloned_shared,
        "cloned_view": cloned_view,
        "made_contiguous": made_contiguous,
    }
    return safe_state_dict, stats


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    # Only save if it's time to save or if it's the final step
    should_save = (
        global_step % config.save_interval == 0 and global_step > 0
    ) or global_step == config.num_train_steps - 1
    if not should_save:
        return

    if isinstance(optimizer, ZeroRedundancyOptimizer | MixedDtypeZeroRedundancyOptimizer):
        # All ranks must participate. Only rank 0 materializes the consolidated
        # optimizer state that is written below.
        optimizer.consolidate_state_dict(to=0)
        if dist.is_initialized():
            dist.barrier()

    if not is_main:
        if dist.is_initialized():
            # Do not let workers enter the next DDP step while rank 0 is still
            # serializing the full model and consolidated optimizer state.
            dist.barrier()
        return

    if should_save:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors with explicit handling for shared/view storages.
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safe_state_dict, save_stats = _prepare_safetensors_state_dict(model_to_save)
        safetensors.torch.save_file(safe_state_dict, tmp_ckpt_dir / "model.safetensors")
        del safe_state_dict
        logging.info(
            "Serialized model.safetensors: tensors=%d cloned_shared=%d cloned_view=%d contiguous=%d",
            save_stats["tensor_count"],
            save_stats["cloned_shared"],
            save_stats["cloned_view"],
            save_stats["made_contiguous"],
        )

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Keep the latest checkpoint plus explicitly preserved periodic steps.
        for old_ckpt_dir in config.checkpoint_dir.iterdir():
            if not old_ckpt_dir.is_dir() or not old_ckpt_dir.name.isdigit():
                continue
            old_step = int(old_ckpt_dir.name)
            keep_periodic = config.keep_period is not None and old_step % config.keep_period == 0
            if old_step != global_step and not keep_periodic:
                shutil.rmtree(old_ckpt_dir)
                logging.info("Removed superseded checkpoint at step %d", old_step)

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)

        if dist.is_initialized():
            dist.barrier()


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            state_dict = safetensors.torch.load_file(safetensors_path, device=str(device))
            missing_keys, unexpected_keys = model_to_load.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logging.warning("Missing keys while loading checkpoint model state: %d", len(missing_keys))
            if unexpected_keys:
                logging.warning("Unexpected keys while loading checkpoint model state: %d", len(unexpected_keys))
            del state_dict
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def ddp_sync_point(tag: str, device: torch.device | None = None):
    if not dist.is_initialized():
        return
    rank = dist.get_rank()
    logging.info(f"DDP sync enter: {tag} (rank={rank})")
    if device is not None and device.type == "cuda" and dist.get_backend() == "nccl" and device.index is not None:
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()
    logging.info(f"DDP sync leave: {tag} (rank={rank})")


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    if config.shard_optimizer and not use_ddp:
        raise ValueError(
            "shard_optimizer=True requires torchrun with more than one process. "
            "Launch full fine-tuning with --nproc_per_node equal to the number of GPUs."
        )
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        if is_main:
            shutil.rmtree(config.checkpoint_dir)
            logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")
        if use_ddp:
            dist.barrier()

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        if is_main:
            exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
        if use_ddp:
            dist.barrier()
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    if use_ddp:
        ddp_sync_point("after_sample_batch_logging", device)

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
            phase_gating=getattr(config.model, "phase_gating", False),
            coarse_fine_shared_routing=getattr(config.model, "coarse_fine_shared_routing", False),
            coarse_fine_rank_gating=getattr(config.model, "coarse_fine_rank_gating", False),
            coarse_fine_score_label=getattr(config.model, "coarse_fine_score_label", False),
            coarse_fine_label_prefix_steps=getattr(config.model, "coarse_fine_label_prefix_steps", 5),
            lora_bank_count=getattr(config.model, "lora_bank_count", 1),
            vlm_lora_bank_count=getattr(config.model, "vlm_lora_bank_count", None),
            action_expert_lora_bank_count=getattr(config.model, "action_expert_lora_bank_count", None),
            phase_count=getattr(config.model, "phase_count", 1),
            phase_router_hidden_dim=getattr(config.model, "phase_router_hidden_dim", 512),
            phase_router_dropout=getattr(config.model, "phase_router_dropout", 0.1),
            coarse_fine_history_window=getattr(config.model, "coarse_fine_history_window", 64),
            coarse_fine_quantile=getattr(config.model, "coarse_fine_quantile", 0.3),
            coarse_fine_min_history=getattr(config.model, "coarse_fine_min_history", 16),
            coarse_fine_hysteresis=getattr(config.model, "coarse_fine_hysteresis", 2),
            coarse_fine_rot_weight=getattr(config.model, "coarse_fine_rot_weight", 0.5),
            coarse_fine_gripper_weight=getattr(config.model, "coarse_fine_gripper_weight", 0.1),
            coarse_fine_router_chunk_prefix_steps=getattr(config.model, "coarse_fine_router_chunk_prefix_steps", 5),
            lora_sp_enabled=getattr(config.model, "lora_sp_enabled", False),
            lora_sp_rank=getattr(config.model, "lora_sp_rank", None),
            lora_sp_energy_threshold=getattr(config.model, "lora_sp_energy_threshold", 0.9),
            lora_sp_router_hidden_dim=getattr(config.model, "lora_sp_router_hidden_dim", 256),
            lora_sp_router_activation=getattr(config.model, "lora_sp_router_activation", "silu"),
            lora_sp_router_nonnegative=getattr(config.model, "lora_sp_router_nonnegative", "softplus"),
            lora_sp_eps=getattr(config.model, "lora_sp_eps", 1e-8),
            lora_sp_spec_loss_weight=getattr(config.model, "lora_sp_spec_loss_weight", 1e-2),
            lora_sp_router_loss_weight=getattr(config.model, "lora_sp_router_loss_weight", 1e-3),
            lora_sp_router_balance_weight=getattr(config.model, "lora_sp_router_balance_weight", 1.0),
            lora_sp_router_z_loss_weight=getattr(config.model, "lora_sp_router_z_loss_weight", 1.0),
            lora_sp_inference_prune=getattr(config.model, "lora_sp_inference_prune", True),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    model.configure_consistency_regularization(
        direction_weight=config.direction_consistency_weight,
        magnitude_weight=config.magnitude_consistency_weight,
        eps=config.consistency_loss_eps,
        time_weighting=config.consistency_time_weighting,
    )
    model.configure_phase_gating_regularization(
        loss_weight=config.phase_gating_loss_weight,
        ce_weight=config.phase_gating_ce_weight,
        balance_weight=config.phase_gating_balance_weight,
        entropy_weight=config.phase_gating_entropy_weight,
        consistency_weight=config.phase_gating_consistency_weight,
        confidence_threshold=config.phase_gating_confidence_threshold,
        entropy_target=config.phase_gating_entropy_target,
        noise_std=config.phase_gating_noise_std,
        temperature=config.phase_gating_temperature,
        use_gumbel_st=config.phase_gating_use_gumbel_st,
        pseudo_window=config.phase_gating_pseudo_window,
    )

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        state_dict = safetensors.torch.load_file(model_path)
        if config.train_action_from_scratch:
            action_branch_prefixes = get_action_branch_parameter_prefixes(config)
            state_dict = {key: value for key, value in state_dict.items() if not key.startswith(action_branch_prefixes)}

        model_state_dict = model_to_load.state_dict()
        filtered_state_dict, load_report = prepare_pytorch_checkpoint_for_model(state_dict, model_state_dict)

        missing_keys, unexpected_keys = model_to_load.load_state_dict(filtered_state_dict, strict=False)

        shape_mismatch_keys = load_report["shape_mismatch_keys"]
        missing_target_keys = load_report["missing_target_keys"]
        ignored_source_keys = load_report["ignored_source_keys"]
        folded_lora_modules = int(load_report["folded_lora_modules"])
        zero_lora_modules = int(load_report["zero_lora_modules"])
        if folded_lora_modules or zero_lora_modules:
            logging.info(
                "Adapted LoRA-wrapped checkpoint to dense model: folded_nonzero=%d zero_delta=%d",
                folded_lora_modules,
                zero_lora_modules,
            )
        if shape_mismatch_keys:
            lora_shape_mismatch_count = sum(1 for key in shape_mismatch_keys if "lora" in key.lower())
            non_lora_shape_mismatch_count = len(shape_mismatch_keys) - lora_shape_mismatch_count
            logging.warning(
                "Skipped %d checkpoint keys due to shape mismatch (%d LoRA, %d non-LoRA).",
                len(shape_mismatch_keys),
                lora_shape_mismatch_count,
                non_lora_shape_mismatch_count,
            )
        if ignored_source_keys:
            logging.info("Ignored %d source-only checkpoint keys.", len(ignored_source_keys))
        if missing_keys:
            expected_reason = "new LoRA params"
            if config.train_action_from_scratch:
                expected_reason += " and the randomly initialized action branch"
            logging.info(f"Missing keys when loading weights: {len(missing_keys)} (expected for {expected_reason})")
        if unexpected_keys:
            logging.warning(f"Unexpected keys when loading weights: {len(unexpected_keys)}")
        if config.strict_pytorch_weight_loading and (
            shape_mismatch_keys or missing_target_keys or missing_keys or unexpected_keys
        ):
            raise RuntimeError(
                "Strict PyTorch checkpoint loading failed: "
                f"shape_mismatch={len(shape_mismatch_keys)}, "
                f"missing_target={len(missing_target_keys)}, "
                f"missing_after_load={len(missing_keys)}, "
                f"unexpected_after_load={len(unexpected_keys)}. "
                f"Shape mismatch sample: {shape_mismatch_keys[:3]}; "
                f"missing target sample: {missing_target_keys[:3]}; "
                f"missing after-load sample: {missing_keys[:3]}."
            )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # Configure trainable parameters (LoRA configs train only LoRA weights).
    _ = configure_trainable_parameters(model, config)

    if use_ddp:
        synchronize_trainable_mask_across_ranks(model, device)

    trainable_parameters = [param for _, param in model.named_parameters() if param.requires_grad]

    trainable_named_parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    action_expert_trainable_parameters = [
        param
        for name, param in trainable_named_parameters
        if is_action_branch_parameter(strip_ddp_prefix(name), config)
    ]
    vlm_trainable_parameters = [
        param
        for name, param in trainable_named_parameters
        if not is_action_branch_parameter(strip_ddp_prefix(name), config)
    ]

    if use_ddp:
        # DDP may overflow internal int counters when verifying extremely large frozen models.
        # Explicitly ignore frozen parameters so only trainable tensors participate in DDP sync/verification.
        frozen_param_names = [
            strip_ddp_prefix(name) for name, param in model.named_parameters() if not param.requires_grad
        ]
        if frozen_param_names:
            model._ddp_params_and_buffers_to_ignore = frozen_param_names  # type: ignore[attr-defined]  # noqa: SLF001
            logging.info(
                "Configured DDP to ignore %d frozen parameters (trainable=%d)",
                len(frozen_param_names),
                len(trainable_named_parameters),
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ddp_sync_point("before_ddp_wrap", device)
        logging.info(
            f"Wrapping model with DDP (rank={dist.get_rank()}, local_rank={local_rank}, world_size={world_size})"
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            broadcast_buffers=False,
            bucket_cap_mb=16,
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )
        logging.info(f"DDP wrapper created successfully (rank={dist.get_rank()})")
        ddp_sync_point("after_ddp_wrap", device)

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters. Full-parameter training uses
    # ZeRO stage 1 to avoid replicating AdamW state on every DDP rank.
    optimizer_kwargs = {
        "lr": peak_lr,
        "betas": (config.optimizer.b1, config.optimizer.b2),
        "eps": config.optimizer.eps,
        "weight_decay": config.optimizer.weight_decay,
    }
    if config.shard_optimizer:
        optim = MixedDtypeZeroRedundancyOptimizer(
            trainable_parameters,
            optimizer_class=torch.optim.AdamW,
            **optimizer_kwargs,
        )
        logging.info(
            "Enabled mixed-dtype ZeRO stage-1 optimizer-state sharding across %d ranks: %s",
            world_size,
            optim.parameter_counts,
        )
    else:
        optim = torch.optim.AdamW(trainable_parameters, **optimizer_kwargs)

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        if config.direction_consistency_weight > 0.0 or config.magnitude_consistency_weight > 0.0:
            weighting_desc = "w(t)=4t(1-t)" if config.consistency_time_weighting else "uniform"
            logging.info(
                "Consistency regularization enabled: "
                f"direction_weight={config.direction_consistency_weight:.3g}, "
                f"magnitude_weight={config.magnitude_consistency_weight:.3g}, "
                f"eps={config.consistency_loss_eps:.1e}, "
                f"time_weighting={weighting_desc}"
            )
        if getattr(model_cfg, "phase_gating", False):
            logging.info(
                "Phase routing enabled: "
                f"phase_count={model_cfg.phase_count}, hidden_dim={model_cfg.phase_router_hidden_dim}, "
                f"dropout={model_cfg.phase_router_dropout:.3f}, loss_weight={config.phase_gating_loss_weight:.3g}, "
                f"ce={config.phase_gating_ce_weight:.3g}, balance={config.phase_gating_balance_weight:.3g}, "
                f"entropy={config.phase_gating_entropy_weight:.3g}, consistency={config.phase_gating_consistency_weight:.3g}, "
                f"tau={config.phase_gating_temperature:.3g}, gumbel_st={config.phase_gating_use_gumbel_st}"
            )
        elif getattr(model_cfg, "coarse_fine_shared_routing", False):
            logging.info(
                "Deterministic coarse/fine/shared routing enabled: "
                f"lora_bank_count={model_cfg.lora_bank_count}, "
                f"history_window={model_cfg.coarse_fine_history_window}, "
                f"quantile={model_cfg.coarse_fine_quantile:.3f}, "
                f"min_history={model_cfg.coarse_fine_min_history}, "
                f"hysteresis={model_cfg.coarse_fine_hysteresis}, "
                f"rot_weight={model_cfg.coarse_fine_rot_weight:.3g}, "
                f"gripper_weight={model_cfg.coarse_fine_gripper_weight:.3g}"
            )
        elif getattr(model_cfg, "lora_sp_enabled", False):
            logging.info(
                "LoRA-SP enabled: "
                f"rank={model_cfg.lora_sp_rank}, eta={model_cfg.lora_sp_energy_threshold:.3f}, "
                f"router_hidden={model_cfg.lora_sp_router_hidden_dim}, "
                f"act={model_cfg.lora_sp_router_activation}, nonneg={model_cfg.lora_sp_router_nonnegative}, "
                f"spec_w={model_cfg.lora_sp_spec_loss_weight:.3g}, "
                f"router_w={model_cfg.lora_sp_router_loss_weight:.3g}, "
                f"balance_w={model_cfg.lora_sp_router_balance_weight:.3g}, "
                f"z_w={model_cfg.lora_sp_router_z_loss_weight:.3g}, "
                f"inference_prune={model_cfg.lora_sp_inference_prune}"
            )
        if config.action_expert_only_grad_clip:
            logging.info(
                "Action-expert-only gradient clipping enabled: "
                f"action_params={sum(p.numel() for p in action_expert_trainable_parameters):,}, "
                f"vlm_params={sum(p.numel() for p in vlm_trainable_parameters):,}, "
                f"multiplier={config.action_expert_grad_clip_multiplier}"
            )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            losses = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            model_loss_components = {}
            raw_loss_components = getattr(get_unwrapped_model(model), "latest_loss_components", {})
            if isinstance(raw_loss_components, dict):
                for key, value in raw_loss_components.items():
                    if isinstance(value, torch.Tensor):
                        model_loss_components[key] = float(value.item())
                    elif isinstance(value, int | float):
                        model_loss_components[key] = float(value)

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            if config.action_expert_only_grad_clip and action_expert_trainable_parameters:
                if vlm_trainable_parameters:
                    vlm_grad_norm = torch.nn.utils.clip_grad_norm_(vlm_trainable_parameters, float("inf"))
                    vlm_grad_norm_value = float(vlm_grad_norm)
                else:
                    vlm_grad_norm_value = 0.0

                action_expert_grad_norm = torch.nn.utils.clip_grad_norm_(
                    action_expert_trainable_parameters,
                    float("inf"),
                )
                action_expert_grad_norm_before_clip = float(action_expert_grad_norm)

                if vlm_grad_norm_value > 0.0:
                    clip_threshold = config.action_expert_grad_clip_multiplier * vlm_grad_norm_value
                else:
                    clip_threshold = config.optimizer.clip_gradient_norm

                if action_expert_grad_norm_before_clip > clip_threshold:
                    torch.nn.utils.clip_grad_norm_(action_expert_trainable_parameters, clip_threshold)

                action_expert_grad_norm_after_clip = torch.nn.utils.clip_grad_norm_(
                    action_expert_trainable_parameters,
                    float("inf"),
                )

                grad_norm = float(action_expert_grad_norm_after_clip)
                extra_grad_logs = {
                    "action_expert_grad_norm_before_clip": action_expert_grad_norm_before_clip,
                    "action_expert_grad_norm_after_clip": float(action_expert_grad_norm_after_clip),
                    "vlm_grad_norm": vlm_grad_norm_value,
                    "action_expert_clip_threshold": float(clip_threshold),
                }
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=config.optimizer.clip_gradient_norm,
                )
                extra_grad_logs = {}

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in trainable_parameters:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        **model_loss_components,
                        **extra_grad_logs,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)

                averaged_extra_loss_metrics = {}
                metric_keys = [
                    "main_loss",
                    "direction_consistency_loss",
                    "magnitude_consistency_loss",
                    "consistency_regularizer",
                    "fine_usage",
                    "coarse_usage",
                    "shared_usage",
                    "offline_label_mode",
                    "offline_event_label_mode",
                    "window_motion",
                    "window_motion_threshold",
                    "window_motion_threshold_ready",
                    "route_p_mean",
                    "route_e_mean",
                    "route_p_hat_mean",
                    "route_e_hat_mean",
                    "p_supervision_loss",
                    "e_supervision_loss",
                    "phase_ce_loss",
                    "phase_balance_loss",
                    "phase_entropy_loss",
                    "phase_consistency_loss",
                    "phase_entropy",
                    "phase_pseudo_coverage",
                    "phase_pseudo_confidence",
                    "phase_regularizer",
                    "phase_weighted_regularizer",
                    "lora_sp_spec_loss",
                    "lora_sp_router_loss",
                    "lora_sp_router_balance_loss",
                    "lora_sp_router_z_loss",
                    "lora_sp_active_rank_mean",
                    "lora_sp_energy_at_k_mean",
                    "lora_sp_weighted_spec_loss",
                    "lora_sp_weighted_router_loss",
                    "lora_sp_layer_count",
                    "total_loss",
                ]
                phase_count = (
                    int(getattr(model_cfg, "phase_count", 0)) if getattr(model_cfg, "phase_gating", False) else 0
                )
                metric_keys.extend([f"phase_usage_{i}" for i in range(phase_count)])

                for key in metric_keys:
                    vals = [info[key] for info in infos if key in info]
                    if len(vals) > 0:
                        averaged_extra_loss_metrics[key] = sum(vals) / len(vals)

                metric_suffix = ""
                if "direction_consistency_loss" in averaged_extra_loss_metrics:
                    metric_suffix += f" dir={averaged_extra_loss_metrics['direction_consistency_loss']:.4f}"
                if "magnitude_consistency_loss" in averaged_extra_loss_metrics:
                    metric_suffix += f" mag={averaged_extra_loss_metrics['magnitude_consistency_loss']:.4f}"
                if "consistency_regularizer" in averaged_extra_loss_metrics:
                    metric_suffix += f" reg={averaged_extra_loss_metrics['consistency_regularizer']:.4f}"
                if "fine_usage" in averaged_extra_loss_metrics:
                    metric_suffix += f" fine={averaged_extra_loss_metrics['fine_usage']:.3f}"
                if "coarse_usage" in averaged_extra_loss_metrics:
                    metric_suffix += f" coarse={averaged_extra_loss_metrics['coarse_usage']:.3f}"
                if "phase_regularizer" in averaged_extra_loss_metrics:
                    metric_suffix += f" phase_reg={averaged_extra_loss_metrics['phase_regularizer']:.4f}"
                if "lora_sp_spec_loss" in averaged_extra_loss_metrics:
                    metric_suffix += f" spec={averaged_extra_loss_metrics['lora_sp_spec_loss']:.4f}"
                if "lora_sp_router_loss" in averaged_extra_loss_metrics:
                    metric_suffix += f" rtr={averaged_extra_loss_metrics['lora_sp_router_loss']:.4f}"
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s{metric_suffix}"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s{metric_suffix}"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    log_payload.update(averaged_extra_loss_metrics)
                    if config.action_expert_only_grad_clip:
                        action_before_vals = [
                            info["action_expert_grad_norm_before_clip"]
                            for info in infos
                            if "action_expert_grad_norm_before_clip" in info
                        ]
                        action_after_vals = [
                            info["action_expert_grad_norm_after_clip"]
                            for info in infos
                            if "action_expert_grad_norm_after_clip" in info
                        ]
                        vlm_vals = [info["vlm_grad_norm"] for info in infos if "vlm_grad_norm" in info]
                        threshold_vals = [
                            info["action_expert_clip_threshold"]
                            for info in infos
                            if "action_expert_clip_threshold" in info
                        ]
                        if len(action_before_vals) > 0:
                            log_payload["action_expert_grad_norm_before_clip"] = sum(action_before_vals) / len(
                                action_before_vals
                            )
                        if len(action_after_vals) > 0:
                            log_payload["action_expert_grad_norm_after_clip"] = sum(action_after_vals) / len(
                                action_after_vals
                            )
                        if len(vlm_vals) > 0:
                            log_payload["vlm_grad_norm"] = sum(vlm_vals) / len(vlm_vals)
                        if len(threshold_vals) > 0:
                            log_payload["action_expert_clip_threshold"] = sum(threshold_vals) / len(threshold_vals)
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
