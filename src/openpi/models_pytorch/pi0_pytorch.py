from collections import deque
import logging
import math
import os

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import LoRALinear
from openpi.models_pytorch.gemma_pytorch import LoRASPConfig
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.action_training_type = getattr(config, "action_training_type", "flow_matching")
        if self.action_training_type not in {"flow_matching", "regression"}:
            raise ValueError(f"Unsupported action_training_type: {self.action_training_type}")
        self.timestamp_conditioned_image_blur = bool(getattr(config, "timestamp_conditioned_image_blur", False))
        self.image_blur_min_cutoff_ratio = float(getattr(config, "image_blur_min_cutoff_ratio", 0.08))
        self.image_blur_max_cutoff_ratio = float(getattr(config, "image_blur_max_cutoff_ratio", 1.0))
        self.phase_gating_enabled = bool(getattr(config, "phase_gating", False))
        self.coarse_fine_shared_routing = bool(getattr(config, "coarse_fine_shared_routing", False))
        self.coarse_fine_rank_gating = bool(getattr(config, "coarse_fine_rank_gating", False))
        self.coarse_fine_bimanual_routing = bool(getattr(config, "coarse_fine_bimanual_routing", False))
        self.coarse_fine_score_label = bool(getattr(config, "coarse_fine_score_label", False))
        self.coarse_fine_label_prefix_steps = max(1, int(getattr(config, "coarse_fine_label_prefix_steps", 5)))
        self.coarse_fine_router_hidden_dim = int(getattr(config, "coarse_fine_router_hidden_dim", 128))
        self.coarse_fine_router_history_steps = max(2, int(getattr(config, "coarse_fine_router_history_steps", 6)))
        self.coarse_fine_router_chunk_prefix_steps = max(
            1,
            int(getattr(config, "coarse_fine_router_chunk_prefix_steps", self.coarse_fine_label_prefix_steps)),
        )
        self.coarse_fine_action_layout = str(getattr(config, "coarse_fine_action_layout", "libero_7d"))
        if self.coarse_fine_action_layout == "libero_7d":
            self.coarse_fine_router_action_dim = 7
            self.coarse_fine_gripper_indices = (6,)
        elif self.coarse_fine_action_layout == "aloha_bimanual_14d":
            self.coarse_fine_router_action_dim = 14
            self.coarse_fine_gripper_indices = (6, 13)
        else:
            raise ValueError(f"Unsupported coarse/fine action layout: {self.coarse_fine_action_layout!r}")
        self.coarse_fine_p_loss_weight = max(0.0, float(getattr(config, "coarse_fine_p_loss_weight", 0.5)))
        self.coarse_fine_e_loss_weight = max(0.0, float(getattr(config, "coarse_fine_e_loss_weight", 0.2)))
        self.coarse_fine_c_loss_weight = max(0.0, float(getattr(config, "coarse_fine_c_loss_weight", 0.2)))
        self.coarse_fine_p_huber_delta = max(1e-4, float(getattr(config, "coarse_fine_p_huber_delta", 0.1)))
        self.lora_sp_enabled = bool(getattr(config, "lora_sp_enabled", False))
        self.lora_sp_spec_loss_weight = max(0.0, float(getattr(config, "lora_sp_spec_loss_weight", 1e-2)))
        self.lora_sp_router_loss_weight = max(0.0, float(getattr(config, "lora_sp_router_loss_weight", 1e-3)))

        if self.phase_gating_enabled and self.coarse_fine_shared_routing:
            raise ValueError("phase_gating and coarse_fine_shared_routing cannot both be enabled")
        if self.coarse_fine_rank_gating and (self.phase_gating_enabled or self.coarse_fine_shared_routing):
            raise ValueError(
                "coarse_fine_rank_gating is mutually exclusive with phase_gating and coarse_fine_shared_routing"
            )

        self.phase_count = max(1, int(getattr(config, "phase_count", 1)))
        self.phase_router_hidden_dim = int(getattr(config, "phase_router_hidden_dim", 512))
        self.phase_router_dropout = float(getattr(config, "phase_router_dropout", 0.1))

        # Optional per-branch bank counts may be None in older configs; in that case
        # fall back to lora_bank_count for backward compatibility.
        def _resolve_bank_count(raw_value, fallback: int) -> int:
            value = fallback if raw_value is None else raw_value
            return max(1, int(value))

        default_bank_count = _resolve_bank_count(getattr(config, "lora_bank_count", 1), fallback=1)
        self.vlm_lora_bank_count = max(
            1,
            _resolve_bank_count(getattr(config, "vlm_lora_bank_count", None), fallback=default_bank_count),
        )
        self.action_expert_lora_bank_count = max(
            1,
            _resolve_bank_count(getattr(config, "action_expert_lora_bank_count", None), fallback=default_bank_count),
        )
        # Route weight dimensions always follow the action expert bank count.
        self.lora_bank_count = self.action_expert_lora_bank_count
        if self.phase_gating_enabled and self.action_expert_lora_bank_count == 1 and self.phase_count > 1:
            # Backward compatibility: historical phase-gated configs did not set lora_bank_count explicitly.
            self.action_expert_lora_bank_count = self.phase_count
            self.lora_bank_count = self.action_expert_lora_bank_count
        if self.coarse_fine_shared_routing and self.action_expert_lora_bank_count < 3:
            raise ValueError("coarse_fine_shared_routing requires lora_bank_count >= 3 (fine, coarse, shared banks)")
        if self.coarse_fine_rank_gating and self.action_expert_lora_bank_count != 1:
            raise ValueError(
                "coarse_fine_rank_gating requires action_expert_lora_bank_count == 1, got "
                f"{self.action_expert_lora_bank_count}"
            )

        self.coarse_fine_history_window = max(1, int(getattr(config, "coarse_fine_history_window", 64)))
        self.coarse_fine_quantile = float(getattr(config, "coarse_fine_quantile", 0.3))
        self.coarse_fine_min_history = max(1, int(getattr(config, "coarse_fine_min_history", 16)))
        self.coarse_fine_hysteresis = max(1, int(getattr(config, "coarse_fine_hysteresis", 2)))
        self.coarse_fine_rot_weight = max(0.0, float(getattr(config, "coarse_fine_rot_weight", 0.5)))
        self.coarse_fine_gripper_weight = max(0.0, float(getattr(config, "coarse_fine_gripper_weight", 0.1)))

        self._coarse_fine_train_history: deque[float] = deque(maxlen=self.coarse_fine_history_window)
        self._coarse_fine_inference_history: deque[float] = deque(maxlen=self.coarse_fine_history_window)
        self._last_inference_motion: float | None = None
        self._coarse_fine_inference_mode = False
        self._coarse_fine_pending_switches = 0
        self._router_inference_action_history: deque[list[list[float]]] = deque(
            maxlen=self.coarse_fine_router_history_steps
        )
        self._router_inference_state_history: deque[list[float]] = deque(maxlen=2)
        self._router_train_feature_stats_logged = False
        self._router_infer_feature_stats_logged = False

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        lora_sp_config = LoRASPConfig(
            enabled=self.lora_sp_enabled,
            rank=getattr(config, "lora_sp_rank", None),
            energy_threshold=float(getattr(config, "lora_sp_energy_threshold", 0.9)),
            router_hidden_dim=int(getattr(config, "lora_sp_router_hidden_dim", 256)),
            router_activation=getattr(config, "lora_sp_router_activation", "silu"),
            router_nonnegative=getattr(config, "lora_sp_router_nonnegative", "softplus"),
            eps=float(getattr(config, "lora_sp_eps", 1e-8)),
            router_balance_weight=float(getattr(config, "lora_sp_router_balance_weight", 1.0)),
            router_z_loss_weight=float(getattr(config, "lora_sp_router_z_loss_weight", 1.0)),
            inference_prune=bool(getattr(config, "lora_sp_inference_prune", True)),
        )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
            vlm_lora_bank_count=self.vlm_lora_bank_count,
            action_expert_lora_bank_count=self.action_expert_lora_bank_count,
            action_expert_pe_routed=(self.coarse_fine_rank_gating and not self.coarse_fine_bimanual_routing),
            action_expert_bimanual_pe_routed=(self.coarse_fine_rank_gating and self.coarse_fine_bimanual_routing),
            lora_sp_config=lora_sp_config,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        compile_enabled = os.getenv("OPENPI_TORCH_COMPILE", "0").lower() in {"1", "true", "yes", "on"}
        if compile_enabled:
            compile_mode = os.getenv("OPENPI_TORCH_COMPILE_MODE", "reduce-overhead")
            try:
                self.sample_actions = torch.compile(self.sample_actions, mode=compile_mode)
                logging.info("Enabled torch.compile for sample_actions (mode=%s)", compile_mode)
            except Exception as e:
                logging.warning("torch.compile failed (%s). Falling back to eager sample_actions.", e)
        else:
            logging.info("torch.compile disabled for sample_actions (set OPENPI_TORCH_COMPILE=1 to enable).")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Auxiliary consistency losses for stabilizing vector fields under LoRA conditioning shifts.
        self.direction_consistency_weight = 0.0
        self.magnitude_consistency_weight = 0.0
        self.consistency_loss_eps = 1e-6
        self.consistency_time_weighting = False
        self.latest_loss_components: dict[str, Tensor] = {}
        self.latest_router_components: dict[str, Tensor] = {}

        # Phase routing regularization weights and behavior.
        self.phase_gating_loss_weight = 0.0
        self.phase_gating_ce_weight = 1.0
        self.phase_gating_balance_weight = 0.0
        self.phase_gating_entropy_weight = 0.0
        self.phase_gating_consistency_weight = 0.0
        self.phase_gating_confidence_threshold = 0.6
        self.phase_gating_entropy_target = 0.7
        self.phase_gating_noise_std = 0.05
        self.phase_gating_temperature = 0.7
        self.phase_gating_use_gumbel_st = True
        self.phase_gating_pseudo_window = 3

        if self.phase_gating_enabled and self.phase_count > 1:
            self.phase_router_prefix_proj = nn.Linear(paligemma_config.width, self.phase_router_hidden_dim)
            self.phase_router_state_proj = nn.Linear(32, self.phase_router_hidden_dim)
            self.phase_router_out = nn.Linear(self.phase_router_hidden_dim, self.phase_count)

        if self.coarse_fine_rank_gating:
            self.tau_router_prefix_proj = nn.Linear(paligemma_config.width, self.coarse_fine_router_hidden_dim)
            self.tau_router_state_proj = nn.Linear(32, self.coarse_fine_router_hidden_dim)
            # Per step: action, first difference, second difference, gripper
            # state(s), and gripper change(s). This is 23D for LIBERO and 46D
            # for the two 7D ALOHA arms.
            router_feature_dim = 3 * self.coarse_fine_router_action_dim + 2 * len(self.coarse_fine_gripper_indices)
            self.tau_router_action_proj = nn.Linear(router_feature_dim, self.coarse_fine_router_hidden_dim)
            self.tau_router_gru = nn.GRU(
                input_size=self.coarse_fine_router_hidden_dim,
                hidden_size=self.coarse_fine_router_hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            if self.coarse_fine_bimanual_routing:
                self.tau_router_p_left_head = nn.Linear(self.coarse_fine_router_hidden_dim, 1)
                self.tau_router_e_left_head = nn.Linear(self.coarse_fine_router_hidden_dim, 1)
                self.tau_router_p_right_head = nn.Linear(self.coarse_fine_router_hidden_dim, 1)
                self.tau_router_e_right_head = nn.Linear(self.coarse_fine_router_hidden_dim, 1)
                self.tau_router_coordination_head = nn.Linear(self.coarse_fine_router_hidden_dim, 1)
            else:
                self.tau_router_p_head = nn.Linear(self.coarse_fine_router_hidden_dim, 1)
                self.tau_router_e_head = nn.Linear(self.coarse_fine_router_hidden_dim, 1)

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def reset(self) -> None:
        """Reset model-side inference state between episodes."""
        if not (self.coarse_fine_shared_routing or self.coarse_fine_rank_gating):
            return

        self._coarse_fine_inference_history.clear()
        self._last_inference_motion = None
        self._coarse_fine_inference_mode = False
        self._coarse_fine_pending_switches = 0
        self._router_inference_action_history.clear()
        self._router_inference_state_history.clear()

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def configure_consistency_regularization(
        self,
        *,
        direction_weight: float,
        magnitude_weight: float,
        eps: float,
        time_weighting: bool,
    ):
        self.direction_consistency_weight = max(0.0, float(direction_weight))
        self.magnitude_consistency_weight = max(0.0, float(magnitude_weight))
        self.consistency_loss_eps = max(1e-12, float(eps))
        self.consistency_time_weighting = bool(time_weighting)

    def configure_phase_gating_regularization(
        self,
        *,
        loss_weight: float,
        ce_weight: float,
        balance_weight: float,
        entropy_weight: float,
        consistency_weight: float,
        confidence_threshold: float,
        entropy_target: float,
        noise_std: float,
        temperature: float,
        use_gumbel_st: bool,
        pseudo_window: int,
    ):
        self.phase_gating_loss_weight = max(0.0, float(loss_weight))
        self.phase_gating_ce_weight = max(0.0, float(ce_weight))
        self.phase_gating_balance_weight = max(0.0, float(balance_weight))
        self.phase_gating_entropy_weight = max(0.0, float(entropy_weight))
        self.phase_gating_consistency_weight = max(0.0, float(consistency_weight))
        self.phase_gating_confidence_threshold = float(min(1.0, max(0.0, confidence_threshold)))
        self.phase_gating_entropy_target = float(min(1.0, max(0.0, entropy_target)))
        self.phase_gating_noise_std = max(0.0, float(noise_std))
        self.phase_gating_temperature = max(1e-3, float(temperature))
        self.phase_gating_use_gumbel_st = bool(use_gumbel_st)
        self.phase_gating_pseudo_window = max(1, int(pseudo_window))

    def _aggregate_lora_sp_stats(
        self,
        lora_sp_stats: list[dict[str, Tensor]],
        *,
        device,
        dtype: torch.dtype,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        zero = torch.zeros((), device=device, dtype=dtype)
        if not lora_sp_stats:
            return {}, zero, zero

        def _mean_of(key: str) -> Tensor:
            values = [entry[key] for entry in lora_sp_stats if key in entry]
            if not values:
                return zero
            return torch.stack(values).mean()

        spec_loss = _mean_of("spec_loss")
        router_loss = _mean_of("router_loss")
        components = {
            "lora_sp_spec_loss": spec_loss,
            "lora_sp_router_loss": router_loss,
            "lora_sp_router_balance_loss": _mean_of("router_balance_loss"),
            "lora_sp_router_z_loss": _mean_of("router_z_loss"),
            "lora_sp_active_rank_mean": _mean_of("active_rank_mean"),
            "lora_sp_energy_at_k_mean": _mean_of("energy_at_k_mean"),
            "lora_sp_layer_count": torch.tensor(float(len(lora_sp_stats)), device=device, dtype=dtype),
        }
        return components, spec_loss, router_loss

    def _extract_action_deltas(self, actions: torch.Tensor, *, prefix_steps: int | None = None) -> torch.Tensor:
        # Ignore the model's zero padding, but preserve every physical action
        # dimension. In particular, ALOHA uses two consecutive 7D arms.
        expected_dim = self.coarse_fine_router_action_dim
        action_dim = min(actions.shape[-1], expected_dim)
        trimmed = actions[:, :, :action_dim].to(dtype=torch.float32)
        if prefix_steps is not None:
            trimmed = trimmed[:, :prefix_steps, :]
        if action_dim == expected_dim:
            return trimmed
        padding = torch.zeros(
            trimmed.shape[0],
            trimmed.shape[1],
            expected_dim - action_dim,
            device=trimmed.device,
            dtype=trimmed.dtype,
        )
        return torch.cat([trimmed, padding], dim=-1)

    def _compute_window_motion_scalar(self, actions: torch.Tensor, *, prefix_steps: int | None = None) -> torch.Tensor:
        deltas = self._extract_action_deltas(actions, prefix_steps=prefix_steps)
        if self.coarse_fine_action_layout == "aloha_bimanual_14d":
            # ALOHA actions are joint-space targets/deltas, not Cartesian
            # xyz/rotation. Average the two arm magnitudes so the scale remains
            # comparable when one or both arms move, and explicitly include both
            # grippers. The expression is invariant to swapping the two arms.
            arm0_joint_motion = torch.norm(deltas[:, :, 0:6], dim=-1).sum(dim=1)
            arm1_joint_motion = torch.norm(deltas[:, :, 7:13], dim=-1).sum(dim=1)
            joint_motion = 0.5 * (arm0_joint_motion + arm1_joint_motion)
            if deltas.shape[1] > 1:
                arm0_gripper = torch.abs(deltas[:, 1:, 6] - deltas[:, :-1, 6]).sum(dim=1)
                arm1_gripper = torch.abs(deltas[:, 1:, 13] - deltas[:, :-1, 13]).sum(dim=1)
            else:
                arm0_gripper = torch.abs(deltas[:, 0, 6])
                arm1_gripper = torch.abs(deltas[:, 0, 13])
            gripper_motion = 0.5 * (arm0_gripper + arm1_gripper)
            return joint_motion + self.coarse_fine_gripper_weight * gripper_motion

        trans_motion = torch.norm(deltas[:, :, :3], dim=-1).sum(dim=1)
        rot_motion = torch.norm(deltas[:, :, 3:6], dim=-1).sum(dim=1)
        if deltas.shape[1] > 1:
            gripper_motion = torch.abs(deltas[:, 1:, 6] - deltas[:, :-1, 6]).sum(dim=1)
        else:
            gripper_motion = torch.abs(deltas[:, 0, 6])
        return (
            trans_motion + self.coarse_fine_rot_weight * rot_motion + self.coarse_fine_gripper_weight * gripper_motion
        )

    def _history_quantile_threshold(
        self,
        history: deque[float],
        *,
        device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if len(history) < self.coarse_fine_min_history:
            return None
        values = torch.tensor(list(history), device=device, dtype=dtype)
        return torch.quantile(values, self.coarse_fine_quantile)

    def _build_coarse_fine_route_weights(
        self,
        fine_mask: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.lora_bank_count < 3:
            raise ValueError("coarse/fine/shared routing requires at least 3 LoRA banks")

        batch_size = fine_mask.shape[0]
        route_weights = torch.zeros(
            batch_size,
            self.lora_bank_count,
            device=fine_mask.device,
            dtype=dtype,
        )
        fine = fine_mask.to(dtype=dtype)
        coarse = (~fine_mask).to(dtype=dtype)

        # Normalized shared composition:
        # fine   => [0.5, 0.0, 0.5]
        # coarse => [0.0, 0.5, 0.5]
        route_weights[:, 0] = 0.5 * fine
        route_weights[:, 1] = 0.5 * coarse
        route_weights[:, 2] = 0.5

        route_ids = torch.where(
            fine_mask,
            torch.zeros(batch_size, device=fine_mask.device, dtype=torch.long),
            torch.ones(batch_size, device=fine_mask.device, dtype=torch.long),
        )
        return route_ids, route_weights

    def _compute_coarse_fine_routes_train(
        self,
        actions: torch.Tensor,
        route_label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        motion = self._compute_window_motion_scalar(actions)
        if route_label is not None:
            label_tensor = route_label
            if not isinstance(label_tensor, torch.Tensor):
                label_tensor = torch.as_tensor(label_tensor, device=motion.device)
            else:
                label_tensor = label_tensor.to(device=motion.device)

            if label_tensor.ndim > 1:
                label_tensor = label_tensor.reshape(label_tensor.shape[0], -1)[:, 0]
            fine_mask = label_tensor.to(dtype=torch.int64) > 0
            threshold_value = motion.new_tensor(0.0)
            threshold_ready = motion.new_tensor(1.0)
            offline_label_mode = motion.new_tensor(1.0)
        else:
            threshold = self._history_quantile_threshold(
                self._coarse_fine_train_history,
                device=motion.device,
                dtype=motion.dtype,
            )

            if threshold is None:
                fine_mask = torch.zeros_like(motion, dtype=torch.bool)
                threshold_value = motion.new_tensor(0.0)
                threshold_ready = motion.new_tensor(0.0)
            else:
                fine_mask = motion <= threshold
                threshold_value = threshold
                threshold_ready = motion.new_tensor(1.0)
            offline_label_mode = motion.new_tensor(0.0)

            with torch.no_grad():
                for value in motion.detach().cpu().tolist():
                    self._coarse_fine_train_history.append(float(value))

        route_ids, route_weights = self._build_coarse_fine_route_weights(fine_mask, dtype=actions.dtype)

        components = {
            "fine_usage": fine_mask.to(dtype=torch.float32).mean(),
            "coarse_usage": (~fine_mask).to(dtype=torch.float32).mean(),
            "shared_usage": motion.new_tensor(1.0),
            "offline_label_mode": offline_label_mode,
            "window_motion": motion.mean(),
            "window_motion_threshold": threshold_value,
            "window_motion_threshold_ready": threshold_ready,
        }
        return route_ids, route_weights, components

    def _compute_coarse_fine_routes_inference(
        self,
        batch_size: int,
        *,
        device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        threshold = self._history_quantile_threshold(
            self._coarse_fine_inference_history,
            device=device,
            dtype=torch.float32,
        )

        candidate_fine = False
        if threshold is not None and self._last_inference_motion is not None:
            candidate_fine = self._last_inference_motion <= float(threshold.item())

        if candidate_fine != self._coarse_fine_inference_mode:
            self._coarse_fine_pending_switches += 1
            if self._coarse_fine_pending_switches >= self.coarse_fine_hysteresis:
                self._coarse_fine_inference_mode = candidate_fine
                self._coarse_fine_pending_switches = 0
        else:
            self._coarse_fine_pending_switches = 0

        fine_mask = torch.full((batch_size,), self._coarse_fine_inference_mode, device=device, dtype=torch.bool)
        return self._build_coarse_fine_route_weights(fine_mask, dtype=dtype)

    def _update_coarse_fine_inference_history(self, actions: torch.Tensor, *, prefix_steps: int | None = None) -> None:
        motion = self._compute_window_motion_scalar(actions, prefix_steps=prefix_steps).mean()
        motion_value = float(motion.detach().cpu().item())
        self._coarse_fine_inference_history.append(motion_value)
        self._last_inference_motion = motion_value

        # Keep an executed-action chunk history for router inference features.
        chunk_steps = self.coarse_fine_router_chunk_prefix_steps
        deltas = self._extract_action_deltas(actions, prefix_steps=chunk_steps)
        if deltas.shape[1] == 0:
            return

        action_dim = self.coarse_fine_router_action_dim
        chunk = deltas[:, :chunk_steps, :action_dim]
        if chunk.shape[1] < chunk_steps:
            pad = torch.zeros(
                chunk.shape[0],
                chunk_steps - chunk.shape[1],
                action_dim,
                device=chunk.device,
                dtype=chunk.dtype,
            )
            chunk = torch.cat([chunk, pad], dim=1)

        # Keep a single history stream; for batched inference we aggregate by mean.
        chunk_seed = chunk.mean(dim=0)
        self._router_inference_action_history.append(
            [[float(v) for v in row] for row in chunk_seed.detach().cpu().tolist()]
        )

    def _normalize_route_scores(
        self,
        route_label: torch.Tensor | None,
        *,
        batch_size: int,
        device,
        label_name: str = "route_label",
    ) -> torch.Tensor | None:
        if route_label is None:
            return None

        label_tensor = route_label
        if not isinstance(label_tensor, torch.Tensor):
            label_tensor = torch.as_tensor(label_tensor, device=device)
        else:
            label_tensor = label_tensor.to(device=device)

        if label_tensor.ndim == 0:
            label_tensor = label_tensor.unsqueeze(0)
        if label_tensor.ndim > 1:
            label_tensor = label_tensor.reshape(label_tensor.shape[0], -1)[:, 0]
        if label_tensor.shape[0] == 1 and batch_size > 1:
            label_tensor = label_tensor.expand(batch_size)
        if label_tensor.shape[0] != batch_size:
            raise ValueError(f"{label_name} batch mismatch: expected {batch_size}, got {label_tensor.shape[0]}")

        label_tensor = label_tensor.to(dtype=torch.float32)
        if self.coarse_fine_score_label:
            return label_tensor.clamp(0.0, 1.0)
        return (label_tensor > 0).to(dtype=torch.float32)

    def _extract_router_action_sequence_train(self, actions: torch.Tensor) -> torch.Tensor:
        # Legacy fallback path (not used in aligned training): consume only past-safe prefix.
        expected_steps = self.coarse_fine_router_history_steps * self.coarse_fine_router_chunk_prefix_steps
        deltas = self._extract_action_deltas(actions, prefix_steps=expected_steps)
        if deltas.shape[1] == 0:
            return torch.zeros(
                deltas.shape[0],
                expected_steps,
                self.coarse_fine_router_action_dim,
                device=deltas.device,
                dtype=deltas.dtype,
            )

        action_dim = self.coarse_fine_router_action_dim
        seq = deltas[:, :, :action_dim]
        if seq.shape[1] < expected_steps:
            pad = torch.zeros(
                seq.shape[0],
                expected_steps - seq.shape[1],
                action_dim,
                device=seq.device,
                dtype=seq.dtype,
            )
            seq = torch.cat([pad, seq], dim=1)
        elif seq.shape[1] > expected_steps:
            seq = seq[:, -expected_steps:, :]
        return seq

    def _normalize_router_action_history(
        self,
        router_action_history: torch.Tensor,
        *,
        batch_size: int,
        device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        expected_steps = self.coarse_fine_router_history_steps * self.coarse_fine_router_chunk_prefix_steps

        seq = (
            router_action_history.to(device=device)
            if isinstance(router_action_history, torch.Tensor)
            else torch.as_tensor(router_action_history, device=device)
        )

        if seq.ndim == 2:
            # [T, 7]
            seq = seq.unsqueeze(0)
        elif seq.ndim == 4:
            # [B, K, S, 7] -> [B, K*S, 7]
            seq = seq.reshape(seq.shape[0], seq.shape[1] * seq.shape[2], seq.shape[3])

        if seq.ndim != 3:
            raise ValueError(f"router_action_history must be rank-3 or rank-4, got shape {tuple(seq.shape)}")

        if seq.shape[0] == 1 and batch_size > 1:
            seq = seq.expand(batch_size, -1, -1)
        if seq.shape[0] != batch_size:
            raise ValueError(f"router_action_history batch mismatch: expected {batch_size}, got {seq.shape[0]}")

        action_dim = self.coarse_fine_router_action_dim
        if seq.shape[-1] < action_dim:
            pad = torch.zeros(
                seq.shape[0],
                seq.shape[1],
                action_dim - seq.shape[-1],
                device=seq.device,
                dtype=seq.dtype,
            )
            seq = torch.cat([seq, pad], dim=-1)
        elif seq.shape[-1] > action_dim:
            seq = seq[:, :, :action_dim]

        if seq.shape[1] < expected_steps:
            pad = torch.zeros(
                seq.shape[0],
                expected_steps - seq.shape[1],
                action_dim,
                device=seq.device,
                dtype=seq.dtype,
            )
            seq = torch.cat([pad, seq], dim=1)
        elif seq.shape[1] > expected_steps:
            seq = seq[:, -expected_steps:, :]

        return seq.to(dtype=dtype)

    def _extract_router_action_sequence_inference(
        self,
        *,
        batch_size: int,
        device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        history_steps = self.coarse_fine_router_history_steps
        chunk_steps = self.coarse_fine_router_chunk_prefix_steps
        expected_steps = history_steps * chunk_steps
        action_dim = self.coarse_fine_router_action_dim
        if len(self._router_inference_action_history) == 0:
            seq = torch.zeros(1, expected_steps, action_dim, device=device, dtype=dtype)
            return seq.expand(batch_size, -1, -1)

        history = torch.tensor(list(self._router_inference_action_history), device=device, dtype=dtype)
        if history.ndim == 2:
            # Backward compatibility for old checkpoints/state format [N, 7].
            history = history.unsqueeze(1).expand(-1, chunk_steps, -1)
        if history.ndim != 3:
            raise ValueError(f"Invalid inference router history shape: {tuple(history.shape)}")

        if history.shape[-1] < action_dim:
            pad = torch.zeros(
                history.shape[0],
                history.shape[1],
                action_dim - history.shape[-1],
                device=device,
                dtype=dtype,
            )
            history = torch.cat([history, pad], dim=-1)
        elif history.shape[-1] > action_dim:
            history = history[:, :, :action_dim]

        if history.shape[1] < chunk_steps:
            pad = torch.zeros(
                history.shape[0],
                chunk_steps - history.shape[1],
                action_dim,
                device=device,
                dtype=dtype,
            )
            history = torch.cat([history, pad], dim=1)
        elif history.shape[1] > chunk_steps:
            history = history[:, :chunk_steps, :]

        if history.shape[0] < history_steps:
            pad = torch.zeros(
                history_steps - history.shape[0],
                chunk_steps,
                action_dim,
                device=device,
                dtype=dtype,
            )
            history = torch.cat([pad, history], dim=0)
        else:
            history = history[-history_steps:]

        return history.reshape(1, expected_steps, action_dim).expand(batch_size, -1, -1).contiguous()

    def _prepare_router_state_features(
        self,
        state: torch.Tensor,
        *,
        dtype: torch.dtype,
        update_inference_history: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_dim = self.tau_router_state_proj.in_features
        state_features = state.to(dtype=dtype)
        if state_features.shape[-1] < state_dim:
            padding = torch.zeros(
                state_features.shape[0],
                state_dim - state_features.shape[-1],
                device=state_features.device,
                dtype=state_features.dtype,
            )
            state_features = torch.cat([state_features, padding], dim=-1)
        elif state_features.shape[-1] > state_dim:
            state_features = state_features[:, :state_dim]

        state_delta = torch.zeros_like(state_features)
        if update_inference_history:
            if len(self._router_inference_state_history) > 0:
                prev_state = torch.tensor(
                    self._router_inference_state_history[-1],
                    device=state_features.device,
                    dtype=state_features.dtype,
                )
                state_delta = state_features - prev_state.unsqueeze(0)

            state_seed = state_features.mean(dim=0)
            self._router_inference_state_history.append([float(v) for v in state_seed.detach().cpu().tolist()])

        return state_features, state_delta

    def _build_router_temporal_features(self, action_seq: torch.Tensor) -> torch.Tensor:
        # action_seq: [B, K, D], D=7 for LIBERO or D=14 for bimanual ALOHA.
        diff = torch.zeros_like(action_seq)
        if action_seq.shape[1] > 1:
            diff[:, 1:, :] = action_seq[:, 1:, :] - action_seq[:, :-1, :]

        jerk = torch.zeros_like(action_seq)
        if action_seq.shape[1] > 2:
            jerk[:, 2:, :] = diff[:, 2:, :] - diff[:, 1:-1, :]

        gripper_state = action_seq[:, :, self.coarse_fine_gripper_indices]
        gripper_change = torch.abs(diff[:, :, self.coarse_fine_gripper_indices])
        return torch.cat([action_seq, diff, jerk, gripper_state, gripper_change], dim=-1)

    def _compute_pe_router_outputs(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        state: torch.Tensor,
        *,
        actions: torch.Tensor | None,
        router_action_history: torch.Tensor | None,
        batch_size: int,
        device,
        update_inference_history: bool,
    ) -> dict[str, torch.Tensor] | None:
        if not hasattr(self, "tau_router_gru"):
            return None

        prefix_mask = prefix_pad_masks.to(dtype=prefix_embs.dtype)
        denom = prefix_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        prefix_pool = torch.sum(prefix_embs * prefix_mask.unsqueeze(-1), dim=1) / denom

        router_dtype = self.tau_router_prefix_proj.weight.dtype
        prefix_pool = prefix_pool.to(dtype=router_dtype)
        state_features, state_delta = self._prepare_router_state_features(
            state,
            dtype=router_dtype,
            # Keep state_delta in the same feature domain as training (zeros).
            update_inference_history=False,
        )

        action_source = "zeros"
        if router_action_history is not None:
            action_seq = self._normalize_router_action_history(
                router_action_history,
                batch_size=batch_size,
                device=device,
                dtype=router_dtype,
            )
            action_source = "offline_history"
        elif update_inference_history:
            action_seq = self._extract_router_action_sequence_inference(
                batch_size=batch_size,
                device=device,
                dtype=router_dtype,
            )
            action_source = "online_history"
        elif actions is not None:
            # Legacy fallback for non-aligned paths.
            action_seq = self._extract_router_action_sequence_train(actions).to(dtype=router_dtype)
            action_source = "legacy_actions"
        else:
            expected_steps = self.coarse_fine_router_history_steps * self.coarse_fine_router_chunk_prefix_steps
            action_seq = torch.zeros(
                batch_size,
                expected_steps,
                self.coarse_fine_router_action_dim,
                device=device,
                dtype=router_dtype,
            )

        expected_steps = self.coarse_fine_router_history_steps * self.coarse_fine_router_chunk_prefix_steps
        assert action_seq.shape[1] == expected_steps, (
            "Router action history length mismatch: "
            f"expected {expected_steps}, got {action_seq.shape[1]} (source={action_source})"
        )
        assert torch.isfinite(action_seq).all(), "Router action history contains non-finite values"

        if self.training and not self._router_train_feature_stats_logged:
            logging.info(
                "Router(train) feature stats: source=%s steps=%d mean=%.6f std=%.6f",
                action_source,
                action_seq.shape[1],
                float(action_seq.mean().detach().cpu().item()),
                float(action_seq.std(unbiased=False).detach().cpu().item()),
            )
            self._router_train_feature_stats_logged = True
        if (not self.training) and not self._router_infer_feature_stats_logged:
            logging.info(
                "Router(infer) feature stats: source=%s steps=%d mean=%.6f std=%.6f",
                action_source,
                action_seq.shape[1],
                float(action_seq.mean().detach().cpu().item()),
                float(action_seq.std(unbiased=False).detach().cpu().item()),
            )
            self._router_infer_feature_stats_logged = True

        action_features = self._build_router_temporal_features(action_seq)
        action_hidden = self.tau_router_action_proj(action_features)

        cond_hidden = (
            self.tau_router_prefix_proj(prefix_pool)
            + self.tau_router_state_proj(state_features)
            + self.tau_router_state_proj(state_delta)
        )
        cond_hidden = F.silu(cond_hidden).unsqueeze(1)
        router_input = action_hidden + cond_hidden
        if self.phase_router_dropout > 0.0 and self.training:
            router_input = F.dropout(router_input, p=self.phase_router_dropout, training=True)

        gru_out, _ = self.tau_router_gru(router_input)
        if self.coarse_fine_bimanual_routing:
            outputs = {}
            for name, head in (
                ("p_left_hat", self.tau_router_p_left_head),
                ("e_left_hat", self.tau_router_e_left_head),
                ("p_right_hat", self.tau_router_p_right_head),
                ("e_right_hat", self.tau_router_e_right_head),
                ("coordination_hat", self.tau_router_coordination_head),
            ):
                outputs[name] = torch.sigmoid(head(gru_out).squeeze(-1))[:, -1].to(dtype=torch.float32)
            return outputs

        p_seq = torch.sigmoid(self.tau_router_p_head(gru_out).squeeze(-1)).to(dtype=torch.float32)
        e_seq = torch.sigmoid(self.tau_router_e_head(gru_out).squeeze(-1)).to(dtype=torch.float32)
        return {"p_hat": p_seq[:, -1], "e_hat": e_seq[:, -1]}

    def _compute_motion_scores_train(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        motion = self._compute_window_motion_scalar(actions, prefix_steps=self.coarse_fine_label_prefix_steps)
        threshold = self._history_quantile_threshold(
            self._coarse_fine_train_history,
            device=motion.device,
            dtype=motion.dtype,
        )
        if threshold is None:
            scores = torch.zeros_like(motion)
            threshold_value = motion.new_tensor(0.0)
            threshold_ready = motion.new_tensor(0.0)
        else:
            scale = threshold.abs().clamp_min(1e-4)
            scores = torch.sigmoid((threshold - motion) / scale)
            threshold_value = threshold
            threshold_ready = motion.new_tensor(1.0)

        with torch.no_grad():
            for value in motion.detach().cpu().tolist():
                self._coarse_fine_train_history.append(float(value))

        return scores.clamp(0.0, 1.0), motion, threshold_value, threshold_ready

    def _compute_event_scores_train(self, actions: torch.Tensor) -> torch.Tensor:
        deltas = self._extract_action_deltas(actions, prefix_steps=self.coarse_fine_label_prefix_steps)
        velocity = deltas

        if velocity.shape[1] > 1:
            acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
            gripper_change = torch.stack(
                [torch.abs(acceleration[:, :, index]).mean(dim=1) for index in self.coarse_fine_gripper_indices],
                dim=1,
            ).mean(dim=1)
        else:
            acceleration = velocity[:, :1, :] * 0.0
            gripper_change = torch.stack(
                [torch.abs(velocity[:, 0, index]) for index in self.coarse_fine_gripper_indices],
                dim=1,
            ).mean(dim=1)

        if acceleration.shape[1] > 1:
            jerk = acceleration[:, 1:, :] - acceleration[:, :-1, :]
            if self.coarse_fine_action_layout == "aloha_bimanual_14d":
                jerk_mag = 0.5 * (
                    torch.norm(jerk[:, :, 0:6], dim=-1).mean(dim=1) + torch.norm(jerk[:, :, 7:13], dim=-1).mean(dim=1)
                )
            else:
                jerk_mag = torch.norm(jerk[:, :, :6], dim=-1).mean(dim=1)
        elif acceleration.shape[1] == 1:
            if self.coarse_fine_action_layout == "aloha_bimanual_14d":
                jerk_mag = 0.5 * (
                    torch.norm(acceleration[:, :, 0:6], dim=-1).mean(dim=1)
                    + torch.norm(acceleration[:, :, 7:13], dim=-1).mean(dim=1)
                )
            else:
                jerk_mag = torch.norm(acceleration[:, :, :6], dim=-1).mean(dim=1)
        else:
            jerk_mag = torch.zeros_like(gripper_change)

        event_raw = jerk_mag + 0.5 * gripper_change
        center = event_raw.median()
        scale = torch.median(torch.abs(event_raw - center)).clamp_min(1e-4)
        return torch.sigmoid((event_raw - center) / scale).clamp(0.0, 1.0)

    def _compute_p_heuristic_scores(
        self,
        batch_size: int,
        *,
        device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        threshold = self._history_quantile_threshold(
            self._coarse_fine_inference_history,
            device=device,
            dtype=torch.float32,
        )
        if threshold is None or self._last_inference_motion is None:
            score_value = 0.0
            threshold_value = torch.tensor(0.0, device=device, dtype=dtype)
            threshold_ready = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            scale = max(abs(float(threshold.item())), 1e-4)
            score_value = 1.0 / (1.0 + math.exp((self._last_inference_motion - float(threshold.item())) / scale))
            threshold_value = torch.tensor(float(threshold.item()), device=device, dtype=dtype)
            threshold_ready = torch.tensor(1.0, device=device, dtype=dtype)

        score_value = min(max(float(score_value), 0.0), 1.0)
        scores = torch.full((batch_size,), score_value, device=device, dtype=dtype)
        return scores, threshold_value, threshold_ready

    def _compute_rank_gating_pe_train(
        self,
        actions: torch.Tensor,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        state: torch.Tensor,
        router_action_history: torch.Tensor | None = None,
        route_label: torch.Tensor | None = None,
        route_label_p: torch.Tensor | None = None,
        route_label_e: torch.Tensor | None = None,
        route_label_p_left: torch.Tensor | None = None,
        route_label_e_left: torch.Tensor | None = None,
        route_label_p_right: torch.Tensor | None = None,
        route_label_e_right: torch.Tensor | None = None,
        route_label_coordination: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if self.coarse_fine_bimanual_routing:
            return self._compute_bimanual_rank_gating_train(
                actions,
                prefix_embs,
                prefix_pad_masks,
                state,
                router_action_history=router_action_history,
                route_label_p_left=route_label_p_left,
                route_label_e_left=route_label_e_left,
                route_label_p_right=route_label_p_right,
                route_label_e_right=route_label_e_right,
                route_label_coordination=route_label_coordination,
            )

        batch_size = actions.shape[0]
        route_p = self._normalize_route_scores(
            route_label_p,
            batch_size=batch_size,
            device=actions.device,
            label_name="route_label_p",
        )
        p_from_offline = route_p is not None
        if route_p is None:
            route_p = self._normalize_route_scores(
                route_label,
                batch_size=batch_size,
                device=actions.device,
                label_name="route_label",
            )
            p_from_offline = route_p is not None

        if route_p is None:
            route_p, motion, threshold_value, threshold_ready = self._compute_motion_scores_train(actions)
            offline_label_mode = route_p.new_tensor(0.0)
        else:
            motion = self._compute_window_motion_scalar(actions, prefix_steps=self.coarse_fine_label_prefix_steps)
            threshold_value = route_p.new_tensor(0.0)
            threshold_ready = route_p.new_tensor(1.0)
            offline_label_mode = route_p.new_tensor(1.0 if p_from_offline else 0.0)

        route_e = self._normalize_route_scores(
            route_label_e,
            batch_size=batch_size,
            device=actions.device,
            label_name="route_label_e",
        )
        e_from_offline = route_e is not None
        if route_e is None:
            route_e = self._compute_event_scores_train(actions)

        assert router_action_history is not None, (
            "coarse_fine_rank_gating training requires offline router_action_history to avoid future-action leakage"
        )

        router_outputs = self._compute_pe_router_outputs(
            prefix_embs,
            prefix_pad_masks,
            state,
            actions=None,
            router_action_history=router_action_history,
            batch_size=batch_size,
            device=actions.device,
            update_inference_history=False,
        )

        p_supervision_loss = route_p.new_tensor(0.0)
        e_supervision_loss = route_p.new_tensor(0.0)
        if router_outputs is not None:
            p_hat = router_outputs["p_hat"].clamp(0.0, 1.0)
            e_hat = router_outputs["e_hat"].clamp(0.0, 1.0)
            p_supervision_loss = F.huber_loss(
                p_hat,
                route_p.detach(),
                reduction="mean",
                delta=self.coarse_fine_p_huber_delta,
            )
            e_supervision_loss = F.binary_cross_entropy(
                e_hat,
                route_e.detach().clamp(0.0, 1.0),
                reduction="mean",
            )
            router_supervision_loss = (
                self.coarse_fine_p_loss_weight * p_supervision_loss
                + self.coarse_fine_e_loss_weight * e_supervision_loss
            )
        else:
            p_hat = route_p.detach().clamp(0.0, 1.0)
            e_hat = route_e.detach().clamp(0.0, 1.0)
            router_supervision_loss = route_p.new_tensor(0.0)

        route_pe = torch.stack([p_hat, e_hat], dim=-1).to(dtype=torch.float32)

        components = {
            "route_score_mean": route_p.mean(),
            "route_p_mean": route_p.mean(),
            "route_e_mean": route_e.mean(),
            "route_p_hat_mean": p_hat.mean(),
            "route_e_hat_mean": e_hat.mean(),
            "router_supervision_loss": router_supervision_loss,
            "p_supervision_loss": p_supervision_loss,
            "e_supervision_loss": e_supervision_loss,
            "window_motion": motion.mean(),
            "window_motion_threshold": threshold_value,
            "window_motion_threshold_ready": threshold_ready,
            "offline_label_mode": offline_label_mode,
            "offline_event_label_mode": route_e.new_tensor(1.0 if e_from_offline else 0.0),
        }

        return route_pe, components, router_supervision_loss

    def _compute_bimanual_rank_gating_train(
        self,
        actions: torch.Tensor,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        state: torch.Tensor,
        *,
        router_action_history: torch.Tensor | None,
        route_label_p_left: torch.Tensor | None,
        route_label_e_left: torch.Tensor | None,
        route_label_p_right: torch.Tensor | None,
        route_label_e_right: torch.Tensor | None,
        route_label_coordination: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        batch_size = actions.shape[0]
        raw_targets = {
            "p_left": route_label_p_left,
            "e_left": route_label_e_left,
            "p_right": route_label_p_right,
            "e_right": route_label_e_right,
            "coordination": route_label_coordination,
        }
        targets = {
            name: self._normalize_route_scores(
                value,
                batch_size=batch_size,
                device=actions.device,
                label_name=f"route_label_{name}",
            )
            for name, value in raw_targets.items()
        }
        missing = [name for name, value in targets.items() if value is None]
        if missing:
            raise ValueError(f"Bimanual PhaseLoRA requires offline labels for {missing}")
        assert router_action_history is not None, (
            "bimanual PhaseLoRA training requires causal offline router_action_history"
        )

        outputs = self._compute_pe_router_outputs(
            prefix_embs,
            prefix_pad_masks,
            state,
            actions=None,
            router_action_history=router_action_history,
            batch_size=batch_size,
            device=actions.device,
            update_inference_history=False,
        )
        assert outputs is not None

        predictions = {
            "p_left": outputs["p_left_hat"].clamp(0.0, 1.0),
            "e_left": outputs["e_left_hat"].clamp(0.0, 1.0),
            "p_right": outputs["p_right_hat"].clamp(0.0, 1.0),
            "e_right": outputs["e_right_hat"].clamp(0.0, 1.0),
            "coordination": outputs["coordination_hat"].clamp(0.0, 1.0),
        }
        p_left_loss = F.huber_loss(
            predictions["p_left"],
            targets["p_left"].detach(),
            reduction="mean",
            delta=self.coarse_fine_p_huber_delta,
        )
        p_right_loss = F.huber_loss(
            predictions["p_right"],
            targets["p_right"].detach(),
            reduction="mean",
            delta=self.coarse_fine_p_huber_delta,
        )
        e_left_loss = F.binary_cross_entropy(
            predictions["e_left"],
            targets["e_left"].detach().clamp(0.0, 1.0),
        )
        e_right_loss = F.binary_cross_entropy(
            predictions["e_right"],
            targets["e_right"].detach().clamp(0.0, 1.0),
        )
        coordination_loss = F.binary_cross_entropy(
            predictions["coordination"],
            targets["coordination"].detach().clamp(0.0, 1.0),
        )
        p_loss = 0.5 * (p_left_loss + p_right_loss)
        e_loss = 0.5 * (e_left_loss + e_right_loss)
        router_supervision_loss = (
            self.coarse_fine_p_loss_weight * p_loss
            + self.coarse_fine_e_loss_weight * e_loss
            + self.coarse_fine_c_loss_weight * coordination_loss
        )

        route = torch.stack(
            [
                predictions["p_left"],
                predictions["e_left"],
                predictions["p_right"],
                predictions["e_right"],
                predictions["coordination"],
            ],
            dim=-1,
        ).to(dtype=torch.float32)
        components = {
            "route_p_left_mean": targets["p_left"].mean(),
            "route_e_left_mean": targets["e_left"].mean(),
            "route_p_right_mean": targets["p_right"].mean(),
            "route_e_right_mean": targets["e_right"].mean(),
            "route_coordination_mean": targets["coordination"].mean(),
            "route_p_left_hat_mean": predictions["p_left"].mean(),
            "route_e_left_hat_mean": predictions["e_left"].mean(),
            "route_p_right_hat_mean": predictions["p_right"].mean(),
            "route_e_right_hat_mean": predictions["e_right"].mean(),
            "route_coordination_hat_mean": predictions["coordination"].mean(),
            "p_left_supervision_loss": p_left_loss,
            "e_left_supervision_loss": e_left_loss,
            "p_right_supervision_loss": p_right_loss,
            "e_right_supervision_loss": e_right_loss,
            "coordination_supervision_loss": coordination_loss,
            "p_supervision_loss": p_loss,
            "e_supervision_loss": e_loss,
            "router_supervision_loss": router_supervision_loss,
            "offline_label_mode": actions.new_tensor(1.0),
            "offline_event_label_mode": actions.new_tensor(1.0),
        }
        return route, components, router_supervision_loss

    def _compute_rank_gating_pe_inference(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.coarse_fine_bimanual_routing:
            return self._compute_bimanual_rank_gating_inference(prefix_embs, prefix_pad_masks, state)

        batch_size = state.shape[0]
        device = state.device
        dtype = torch.float32

        router_outputs = self._compute_pe_router_outputs(
            prefix_embs,
            prefix_pad_masks,
            state,
            actions=None,
            router_action_history=None,
            batch_size=batch_size,
            device=device,
            update_inference_history=True,
        )

        p_heuristic, threshold_value, threshold_ready = self._compute_p_heuristic_scores(
            batch_size,
            device=device,
            dtype=dtype,
        )
        if router_outputs is not None:
            p_hat = router_outputs["p_hat"].clamp(0.0, 1.0)
            e_hat = router_outputs["e_hat"].clamp(0.0, 1.0)
        else:
            p_hat = p_heuristic
            e_hat = torch.full_like(p_hat, 0.5)

        route_pe = torch.stack([p_hat, e_hat], dim=-1).to(dtype=torch.float32)
        components = {
            "route_p_hat_mean": p_hat.mean(),
            "route_e_hat_mean": e_hat.mean(),
            "route_p_heuristic_mean": p_heuristic.mean(),
            "window_motion_threshold": threshold_value,
            "window_motion_threshold_ready": threshold_ready,
        }

        return route_pe, components

    def _compute_bimanual_rank_gating_inference(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        outputs = self._compute_pe_router_outputs(
            prefix_embs,
            prefix_pad_masks,
            state,
            actions=None,
            router_action_history=None,
            batch_size=state.shape[0],
            device=state.device,
            update_inference_history=True,
        )
        assert outputs is not None
        route = (
            torch.stack(
                [
                    outputs["p_left_hat"],
                    outputs["e_left_hat"],
                    outputs["p_right_hat"],
                    outputs["e_right_hat"],
                    outputs["coordination_hat"],
                ],
                dim=-1,
            )
            .clamp(0.0, 1.0)
            .to(dtype=torch.float32)
        )
        components = {
            "route_p_left_hat_mean": route[:, 0].mean(),
            "route_e_left_hat_mean": route[:, 1].mean(),
            "route_p_right_hat_mean": route[:, 2].mean(),
            "route_e_right_hat_mean": route[:, 3].mean(),
            "route_coordination_hat_mean": route[:, 4].mean(),
        }
        return route, components

    def _normalize_for_phase_heuristics(self, values: torch.Tensor) -> torch.Tensor:
        center = values.median()
        scale = torch.median(torch.abs(values - center)).clamp_min(1e-4)
        return (values - center) / scale

    def _compute_phase_router_logits(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not (self.phase_gating_enabled and self.phase_count > 1):
            return None, None

        prefix_mask = prefix_pad_masks.to(dtype=prefix_embs.dtype)
        denom = prefix_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        prefix_pool = torch.sum(prefix_embs * prefix_mask.unsqueeze(-1), dim=1) / denom

        state_dim = self.phase_router_state_proj.in_features
        state_features = state.to(dtype=prefix_embs.dtype)
        if state_features.shape[-1] < state_dim:
            padding = torch.zeros(
                state_features.shape[0],
                state_dim - state_features.shape[-1],
                device=state_features.device,
                dtype=state_features.dtype,
            )
            state_features = torch.cat([state_features, padding], dim=-1)
        elif state_features.shape[-1] > state_dim:
            state_features = state_features[:, :state_dim]

        router_dtype = self.phase_router_prefix_proj.weight.dtype
        prefix_pool = prefix_pool.to(dtype=router_dtype)
        state_features = state_features.to(dtype=router_dtype)

        hidden = self.phase_router_prefix_proj(prefix_pool) + self.phase_router_state_proj(state_features)
        hidden = F.silu(hidden)
        if self.phase_router_dropout > 0.0 and self.training:
            hidden = F.dropout(hidden, p=self.phase_router_dropout, training=True)

        logits = self.phase_router_out(hidden)
        return logits, hidden

    def _sample_phase_routes(
        self, phase_logits: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if phase_logits is None or self.phase_count <= 1:
            return None, None

        probs = F.softmax(phase_logits, dim=-1)
        if self.training:
            if self.phase_gating_use_gumbel_st:
                route_weights = F.gumbel_softmax(
                    phase_logits,
                    tau=self.phase_gating_temperature,
                    hard=True,
                    dim=-1,
                )
            else:
                route_weights = F.softmax(phase_logits / self.phase_gating_temperature, dim=-1)
        else:
            route_ids = torch.argmax(probs, dim=-1)
            route_weights = F.one_hot(route_ids, num_classes=self.phase_count).to(dtype=probs.dtype)

        route_ids = torch.argmax(route_weights, dim=-1)
        return route_ids, route_weights

    def _build_phase_pseudo_labels(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = actions.shape[0]
        device = actions.device
        if self.phase_count != 3:
            return (
                torch.zeros(batch_size, dtype=torch.long, device=device),
                torch.zeros(batch_size, dtype=torch.bool, device=device),
                torch.zeros(batch_size, dtype=torch.float32, device=device),
            )

        window = min(max(1, self.phase_gating_pseudo_window), actions.shape[1])
        near_actions = actions[:, :window]
        gripper_cmd = near_actions[:, :, -1].mean(dim=1)
        ee_speed = torch.norm(near_actions[:, :, :3], dim=-1).mean(dim=1)

        gripper_norm = self._normalize_for_phase_heuristics(gripper_cmd)
        speed_norm = self._normalize_for_phase_heuristics(ee_speed)

        slow_conf = torch.sigmoid(-(speed_norm - 0.25) * 2.0)
        fast_conf = 1.0 - slow_conf
        gripper_close_conf = torch.sigmoid((-gripper_norm - 0.4) * 2.2)
        gripper_open_conf = torch.sigmoid((gripper_norm - 0.4) * 2.2)
        gripper_neutral_conf = torch.exp(-torch.abs(gripper_norm))

        pre_grasp_score = gripper_close_conf * slow_conf
        pre_place_score = gripper_open_conf * slow_conf
        other_score = 0.7 * fast_conf + 0.3 * gripper_neutral_conf

        phase_scores = torch.stack([pre_grasp_score, pre_place_score, other_score], dim=-1)
        phase_probs = phase_scores / phase_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        top2 = torch.topk(phase_probs, k=2, dim=-1).values
        pseudo_phase = torch.argmax(phase_probs, dim=-1)
        pseudo_conf = top2[:, 0]
        pseudo_margin = top2[:, 0] - top2[:, 1]
        pseudo_mask = (pseudo_conf >= self.phase_gating_confidence_threshold) & (pseudo_margin >= 0.15)
        return pseudo_phase, pseudo_mask, pseudo_conf

    def _js_divergence(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        probs_a = F.softmax(logits_a, dim=-1)
        probs_b = F.softmax(logits_b, dim=-1)
        mean_probs = 0.5 * (probs_a + probs_b)

        kl_a = torch.sum(probs_a * (torch.log(probs_a + eps) - torch.log(mean_probs + eps)), dim=-1)
        kl_b = torch.sum(probs_b * (torch.log(probs_b + eps) - torch.log(mean_probs + eps)), dim=-1)
        return 0.5 * (kl_a + kl_b).mean()

    def _compute_phase_gating_regularizer(
        self,
        phase_logits: torch.Tensor,
        phase_hidden: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        phase_probs = F.softmax(phase_logits, dim=-1)
        zero = phase_logits.new_tensor(0.0)
        eps = 1e-6

        pseudo_phase, pseudo_mask, pseudo_conf = self._build_phase_pseudo_labels(actions)
        if pseudo_mask.any():
            ce_raw = F.cross_entropy(phase_logits[pseudo_mask], pseudo_phase[pseudo_mask], reduction="none")
            ce_weights = pseudo_conf[pseudo_mask].detach().clamp_min(1e-3)
            ce_loss = torch.sum(ce_raw * ce_weights) / torch.sum(ce_weights)
            pseudo_conf_mean = pseudo_conf[pseudo_mask].mean()
        else:
            ce_loss = zero
            pseudo_conf_mean = zero

        mean_phase_probs = phase_probs.mean(dim=0)
        uniform_probs = torch.full_like(mean_phase_probs, 1.0 / float(self.phase_count))
        balance_loss = torch.sum(
            mean_phase_probs * (torch.log(mean_phase_probs + eps) - torch.log(uniform_probs + eps))
        )

        phase_entropy = -torch.sum(phase_probs * torch.log(phase_probs + eps), dim=-1).mean()
        norm_entropy = phase_entropy / max(math.log(float(self.phase_count)), eps)
        entropy_loss = F.relu(self.phase_gating_entropy_target - norm_entropy)

        if self.phase_gating_noise_std > 0.0:
            noisy_hidden = phase_hidden + torch.randn_like(phase_hidden) * self.phase_gating_noise_std
            noisy_logits = self.phase_router_out(noisy_hidden)
            consistency_loss = self._js_divergence(phase_logits, noisy_logits)
        else:
            consistency_loss = zero

        regularizer = (
            self.phase_gating_ce_weight * ce_loss
            + self.phase_gating_balance_weight * balance_loss
            + self.phase_gating_entropy_weight * entropy_loss
            + self.phase_gating_consistency_weight * consistency_loss
        )

        components = {
            "phase_ce_loss": ce_loss,
            "phase_balance_loss": balance_loss,
            "phase_entropy_loss": entropy_loss,
            "phase_consistency_loss": consistency_loss,
            "phase_entropy": norm_entropy,
            "phase_pseudo_coverage": pseudo_mask.to(dtype=torch.float32).mean(),
            "phase_pseudo_confidence": pseudo_conf_mean,
            "phase_regularizer": regularizer,
        }
        for phase_idx in range(self.phase_count):
            components[f"phase_usage_{phase_idx}"] = mean_phase_probs[phase_idx]
        return components, regularizer

    def _predict_vector_field(
        self,
        prefix_embs: torch.Tensor,
        suffix_embs: torch.Tensor,
        att_2d_masks_4d: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_cond: torch.Tensor | None,
        *,
        use_checkpoint: bool,
        lora_phase_ids: torch.Tensor | None = None,
        lora_phase_weights: torch.Tensor | None = None,
        lora_route_pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            with LoRALinear.lora_routing(
                route_ids=lora_phase_ids,
                route_weights=lora_phase_weights,
                route_pe=lora_route_pe,
            ):
                (_, suffix_out), _ = self.paligemma_with_expert.forward(
                    attention_mask=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond],
                )
            return suffix_out

        if use_checkpoint:
            suffix_out = self._apply_checkpoint(
                forward_func,
                prefix_embs,
                suffix_embs,
                att_2d_masks_4d,
                position_ids,
                adarms_cond,
            )
        else:
            suffix_out = forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond)

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        if use_checkpoint:
            v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        else:
            v_t = action_out_proj_func(suffix_out)
        return v_t.to(dtype=torch.float32)

    def _compute_consistency_losses(
        self,
        v_base: torch.Tensor,
        v_lora: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = self.consistency_loss_eps
        v_base_flat = v_base.reshape(v_base.shape[0], -1)
        v_lora_flat = v_lora.reshape(v_lora.shape[0], -1)

        cosine = F.cosine_similarity(v_base_flat, v_lora_flat, dim=-1, eps=eps)
        direction_loss_per_sample = 1.0 - cosine

        v_base_norm = torch.norm(v_base_flat, dim=-1)
        v_lora_norm = torch.norm(v_lora_flat, dim=-1)
        magnitude_loss_per_sample = (torch.log(v_lora_norm + eps) - torch.log(v_base_norm + eps)) ** 2

        if self.consistency_time_weighting:
            sample_weights = (4.0 * time * (1.0 - time)).to(dtype=direction_loss_per_sample.dtype)
        else:
            sample_weights = torch.ones_like(direction_loss_per_sample)

        normalizer = torch.clamp(sample_weights.sum(), min=eps)
        direction_loss = torch.sum(direction_loss_per_sample * sample_weights) / normalizer
        magnitude_loss = torch.sum(magnitude_loss_per_sample * sample_weights) / normalizer
        return direction_loss, magnitude_loss

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def zero_action_input(self, reference_actions: torch.Tensor) -> torch.Tensor:
        """Create the zero-initialized action suffix used by regression training/inference."""
        return torch.zeros_like(reference_actions, dtype=torch.float32)

    def zero_time(self, bsize: int, device) -> torch.Tensor:
        """Create the zero timestep used by regression training/inference."""
        return torch.zeros(bsize, dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, timestep=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            if self.timestamp_conditioned_image_blur and timestep is not None:
                img_emb = self._apply_timestamp_conditioned_feature_blur(img_emb, timestep)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def _apply_timestamp_conditioned_feature_blur(
        self, image_tokens: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        if image_tokens.ndim != 3:
            return image_tokens
        if timestep.ndim != 1:
            return image_tokens

        bsize, num_tokens, emb_dim = image_tokens.shape
        if bsize == 0 or num_tokens == 0 or emb_dim == 0:
            return image_tokens

        min_ratio = max(0.0, min(1.0, self.image_blur_min_cutoff_ratio))
        max_ratio = max(min_ratio, min(1.0, self.image_blur_max_cutoff_ratio))

        # Current flow setup denoises from t=1 to t=0, so use (1 - t) to release high frequencies over time.
        t = torch.clamp(timestep.to(dtype=torch.float32, device=image_tokens.device), 0.0, 1.0)
        cutoff_ratio = min_ratio + (max_ratio - min_ratio) * (1.0 - t)

        grid_size = int(math.sqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            return image_tokens

        x = image_tokens.to(torch.float32).reshape(bsize, grid_size, grid_size, emb_dim).permute(0, 3, 1, 2)
        freq = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))

        yy = torch.arange(grid_size, device=image_tokens.device, dtype=torch.float32)
        xx = torch.arange(grid_size, device=image_tokens.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        cy = (grid_size - 1) / 2.0
        cx = (grid_size - 1) / 2.0
        radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        radius_limit = 0.5 * float(grid_size)
        cutoff_radius = torch.clamp(cutoff_ratio * radius_limit, min=1.0).view(bsize, 1, 1)
        mask = (radius.unsqueeze(0) <= cutoff_radius).to(freq.dtype).unsqueeze(1)

        filtered = torch.real(torch.fft.ifft2(torch.fft.ifftshift(freq * mask, dim=(-2, -1)), dim=(-2, -1)))
        filtered = filtered.permute(0, 2, 3, 1).reshape(bsize, num_tokens, emb_dim)
        return filtered.to(image_tokens.dtype)

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        route_label = getattr(observation, "route_label", None)
        route_label_p = getattr(observation, "route_label_p", None)
        route_label_e = getattr(observation, "route_label_e", None)
        route_label_p_left = getattr(observation, "route_label_p_left", None)
        route_label_e_left = getattr(observation, "route_label_e_left", None)
        route_label_p_right = getattr(observation, "route_label_p_right", None)
        route_label_e_right = getattr(observation, "route_label_e_right", None)
        route_label_coordination = getattr(observation, "route_label_coordination", None)
        router_action_history = getattr(observation, "router_action_history", None)
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if self.action_training_type == "regression":
            time = self.zero_time(actions.shape[0], actions.device)
            x_t = self.zero_action_input(actions)
            target = actions
        else:
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)

            if time is None:
                time = self.sample_time(actions.shape[0], actions.device)

            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            target = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, timestep=time
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        phase_logits = None
        phase_hidden = None
        phase_ids = None
        phase_weights = None
        route_pe = None
        coarse_fine_components: dict[str, Tensor] = {}
        router_components: dict[str, Tensor] = {}
        router_supervision_loss = actions.new_tensor(0.0)

        if self.coarse_fine_rank_gating:
            route_pe, router_components, router_supervision_loss = self._compute_rank_gating_pe_train(
                actions,
                prefix_embs,
                prefix_pad_masks,
                state,
                router_action_history=router_action_history,
                route_label=route_label,
                route_label_p=route_label_p,
                route_label_e=route_label_e,
                route_label_p_left=route_label_p_left,
                route_label_e_left=route_label_e_left,
                route_label_p_right=route_label_p_right,
                route_label_e_right=route_label_e_right,
                route_label_coordination=route_label_coordination,
            )
        elif self.coarse_fine_shared_routing:
            phase_ids, phase_weights, coarse_fine_components = self._compute_coarse_fine_routes_train(
                actions,
                route_label=route_label,
            )
        else:
            phase_logits, phase_hidden = self._compute_phase_router_logits(prefix_embs, prefix_pad_masks, state)
            phase_ids, phase_weights = self._sample_phase_routes(phase_logits)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        if self.lora_sp_enabled:
            LoRALinear.reset_lora_sp_stats()

        v_t = self._predict_vector_field(
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
            use_checkpoint=True,
            lora_phase_ids=phase_ids,
            lora_phase_weights=phase_weights,
            lora_route_pe=route_pe,
        )

        lora_sp_stats = LoRALinear.pop_lora_sp_stats() if self.lora_sp_enabled else []

        if self.action_training_type == "regression":
            losses = F.l1_loss(v_t, target, reduction="none")
        else:
            losses = F.mse_loss(target, v_t, reduction="none")

        main_loss = losses.mean()
        loss_components: dict[str, Tensor] = {
            "main_loss": main_loss,
        }
        if coarse_fine_components:
            loss_components.update(coarse_fine_components)
        if router_components:
            loss_components.update(router_components)

        if self.lora_sp_enabled:
            lora_sp_components, spec_loss, router_loss = self._aggregate_lora_sp_stats(
                lora_sp_stats,
                device=losses.device,
                dtype=losses.dtype,
            )
            if lora_sp_components:
                loss_components.update(lora_sp_components)

            if self.lora_sp_spec_loss_weight > 0.0:
                weighted_spec_loss = self.lora_sp_spec_loss_weight * spec_loss
                losses = losses + weighted_spec_loss
                loss_components["lora_sp_weighted_spec_loss"] = weighted_spec_loss
            if self.lora_sp_router_loss_weight > 0.0:
                weighted_router_loss = self.lora_sp_router_loss_weight * router_loss
                losses = losses + weighted_router_loss
                loss_components["lora_sp_weighted_router_loss"] = weighted_router_loss

        consistency_enabled = (
            self.training
            and self.action_training_type == "flow_matching"
            and time is not None
            and (self.direction_consistency_weight > 0.0 or self.magnitude_consistency_weight > 0.0)
        )
        if consistency_enabled:
            with torch.no_grad(), LoRALinear.lora_disabled():
                v_base = self._predict_vector_field(
                    prefix_embs,
                    suffix_embs,
                    att_2d_masks_4d,
                    position_ids,
                    adarms_cond,
                    use_checkpoint=False,
                    lora_phase_ids=phase_ids,
                    lora_phase_weights=phase_weights,
                    lora_route_pe=route_pe,
                )

            direction_loss, magnitude_loss = self._compute_consistency_losses(v_base, v_t, time)
            consistency_regularizer = (
                self.direction_consistency_weight * direction_loss + self.magnitude_consistency_weight * magnitude_loss
            )
            losses = losses + consistency_regularizer

            loss_components["direction_consistency_loss"] = direction_loss
            loss_components["magnitude_consistency_loss"] = magnitude_loss
            loss_components["consistency_regularizer"] = consistency_regularizer

        if phase_logits is not None:
            phase_probs = F.softmax(phase_logits, dim=-1)
            phase_usage = phase_probs.mean(dim=0)
            for phase_idx in range(self.phase_count):
                loss_components[f"phase_usage_{phase_idx}"] = phase_usage[phase_idx]

            if self.phase_gating_loss_weight > 0.0 and phase_hidden is not None:
                phase_components, phase_regularizer = self._compute_phase_gating_regularizer(
                    phase_logits,
                    phase_hidden,
                    actions,
                )
                losses = losses + self.phase_gating_loss_weight * phase_regularizer
                loss_components.update(phase_components)
                loss_components["phase_weighted_regularizer"] = self.phase_gating_loss_weight * phase_regularizer

        if self.coarse_fine_rank_gating and router_supervision_loss is not None:
            losses = losses + router_supervision_loss
            loss_components["router_weighted_supervision"] = router_supervision_loss

        loss_components["total_loss"] = losses.mean()
        self.latest_loss_components = {name: value.detach() for name, value in loss_components.items()}
        return losses

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        zero_time = self.zero_time(bsize, device)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            timestep=zero_time if self.timestamp_conditioned_image_blur else None,
        )
        phase_ids = None
        phase_weights = None
        route_pe = None
        router_components: dict[str, Tensor] = {}
        if self.coarse_fine_rank_gating:
            route_pe, router_components = self._compute_rank_gating_pe_inference(prefix_embs, prefix_pad_masks, state)
        elif self.coarse_fine_shared_routing:
            phase_ids, phase_weights = self._compute_coarse_fine_routes_inference(
                bsize,
                device=state.device,
                dtype=prefix_embs.dtype,
            )
        else:
            phase_logits, _ = self._compute_phase_router_logits(prefix_embs, prefix_pad_masks, state)
            phase_ids, phase_weights = self._sample_phase_routes(phase_logits)
        self.latest_router_components = {name: value.detach() for name, value in router_components.items()}

        if self.action_training_type == "regression":
            zero_actions = self.zero_action_input(
                torch.zeros(
                    (bsize, self.config.action_horizon, self.config.action_dim),
                    dtype=torch.float32,
                    device=device,
                )
            )
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state, zero_actions, zero_time
            )

            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            predicted_actions = self._predict_vector_field(
                prefix_embs,
                suffix_embs,
                att_2d_masks_4d,
                position_ids,
                adarms_cond,
                use_checkpoint=False,
                lora_phase_ids=phase_ids,
                lora_phase_weights=phase_weights,
                lora_route_pe=route_pe,
            )
            if self.coarse_fine_shared_routing:
                self._update_coarse_fine_inference_history(predicted_actions)
            elif self.coarse_fine_rank_gating:
                self._update_coarse_fine_inference_history(
                    predicted_actions,
                    prefix_steps=self.coarse_fine_label_prefix_steps,
                )
            return predicted_actions

        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        past_key_values = None
        if not self.timestamp_conditioned_image_blur:
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            # Compute image and language key value cache once.
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

            with LoRALinear.lora_routing(route_ids=phase_ids, route_weights=phase_weights, route_pe=route_pe):
                _, past_key_values = self.paligemma_with_expert.forward(
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            if self.timestamp_conditioned_image_blur:
                v_t = self.denoise_step_dynamic_prefix(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                    x_t,
                    expanded_time,
                    phase_ids=phase_ids,
                    phase_weights=phase_weights,
                    route_pe=route_pe,
                )
            else:
                v_t = self.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    expanded_time,
                    phase_ids=phase_ids,
                    phase_weights=phase_weights,
                    route_pe=route_pe,
                )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt

        if self.coarse_fine_shared_routing:
            self._update_coarse_fine_inference_history(x_t)
        elif self.coarse_fine_rank_gating:
            self._update_coarse_fine_inference_history(x_t, prefix_steps=self.coarse_fine_label_prefix_steps)
        return x_t

    def denoise_step_dynamic_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        x_t,
        timestep,
        phase_ids: torch.Tensor | None = None,
        phase_weights: torch.Tensor | None = None,
        route_pe: torch.Tensor | None = None,
    ):
        """Apply one denoising step while recomputing timestamp-conditioned visual prefix."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            timestep=timestep,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        # Keep dtype behavior consistent with the standard inference path.
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        with LoRALinear.lora_routing(route_ids=phase_ids, route_weights=phase_weights, route_pe=route_pe):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        phase_ids: torch.Tensor | None = None,
        phase_weights: torch.Tensor | None = None,
        route_pe: torch.Tensor | None = None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        with LoRALinear.lora_routing(route_ids=phase_ids, route_weights=phase_weights, route_pe=route_pe):
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
