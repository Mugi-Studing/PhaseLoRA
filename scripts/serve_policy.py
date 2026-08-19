import dataclasses
import enum
import logging
import random
import socket

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Disable LeRobot libero extra-delta action transforms in the selected train config.
    # Useful for checkpoints whose actions are already in the target control space.
    disable_extra_delta_transform: bool = False

    # Port to serve the policy on.
    port: int = 8000
    # Seed PyTorch/NumPy/Python inference randomness for reproducible policy
    # sampling. Use None to retain nondeterministic process seeding.
    inference_seed: int | None = 0
    # Record the policy's behavior for debugging.
    record: bool = False

    # Replace the router's P/E outputs with random values in [0, 1] during inference.
    randomize_router_pe: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    sample_kwargs: dict[str, bool] | None = None,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            sample_kwargs=sample_kwargs,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    sample_kwargs = None
    if args.randomize_router_pe and args.env == EnvMode.LIBERO:
        sample_kwargs = {"randomize_router_pe": True}

    def _maybe_patch_config(train_cfg: _config.TrainConfig) -> _config.TrainConfig:
        if not args.disable_extra_delta_transform:
            return train_cfg
        data_cfg = train_cfg.data
        if hasattr(data_cfg, "extra_delta_transform"):
            data_cfg = dataclasses.replace(data_cfg, extra_delta_transform=False)
            return dataclasses.replace(train_cfg, data=data_cfg)
        logging.warning(
            "--disable-extra-delta-transform was set, but config %s has no extra_delta_transform field.",
            train_cfg.name,
        )
        return train_cfg

    match args.policy:
        case Checkpoint():
            train_cfg = _maybe_patch_config(_config.get_config(args.policy.config))
            return _policy_config.create_trained_policy(
                train_cfg,
                args.policy.dir,
                default_prompt=args.default_prompt,
                sample_kwargs=sample_kwargs,
            )
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                sample_kwargs=sample_kwargs,
            )


def main(args: Args) -> None:
    if args.inference_seed is not None:
        random.seed(args.inference_seed)
        np.random.seed(args.inference_seed)
        try:
            import torch

            torch.manual_seed(args.inference_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.inference_seed)
        except ImportError:
            pass

    policy = create_policy(args)
    policy_metadata = dict(policy.metadata)

    # Attach provenance so evaluators can persist exactly which checkpoint/config was served.
    served_checkpoint = args.policy if isinstance(args.policy, Checkpoint) else DEFAULT_CHECKPOINT.get(args.env)

    if served_checkpoint is not None:
        policy_metadata["checkpoint_config"] = served_checkpoint.config
        policy_metadata["checkpoint_dir"] = served_checkpoint.dir
    policy_metadata["env_mode"] = args.env.value
    policy_metadata["disable_extra_delta_transform"] = args.disable_extra_delta_transform
    policy_metadata["randomize_router_pe"] = args.randomize_router_pe
    policy_metadata["inference_seed"] = args.inference_seed

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
