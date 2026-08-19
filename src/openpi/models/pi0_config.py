import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    # PyTorch-only action training objective. `flow_matching` preserves the current denoising setup,
    # while `regression` predicts actions directly from a zero-initialized action suffix.
    action_training_type: Literal["flow_matching", "regression"] = "flow_matching"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # If true, apply denoising-timestamp-conditioned low-pass filtering to vision encoder features.
    timestamp_conditioned_image_blur: bool = False
    # Minimum low-pass cutoff ratio used at the beginning of denoising.
    image_blur_min_cutoff_ratio: float = 0.08
    # Maximum low-pass cutoff ratio used near the end of denoising.
    image_blur_max_cutoff_ratio: float = 1.0
    # If true, enable learned phase routing and multi-bank LoRA selection in the PyTorch PI0.5 model.
    phase_gating: bool = False
    # If true, use deterministic coarse/fine routing with a shared always-on LoRA bank (PyTorch only).
    # This mode does not train the learned phase router.
    coarse_fine_shared_routing: bool = False
    # Number of LoRA banks to instantiate in PyTorch Gemma blocks.
    # For coarse/fine/shared routing this should be >= 3.
    lora_bank_count: int = 1
    # Optional per-branch overrides for PyTorch only.
    # If None, each branch falls back to lora_bank_count for backward compatibility.
    vlm_lora_bank_count: int | None = None
    action_expert_lora_bank_count: int | None = None
    # Number of routing phases / LoRA banks. Three corresponds to pre-grasp, pre-place, and other.
    phase_count: int = 3
    # Hidden width for the lightweight phase router.
    phase_router_hidden_dim: int = 512
    # Dropout used in router hidden activations.
    phase_router_dropout: float = 0.1
    # Number of historical windows used by the coarse/fine quantile rule.
    coarse_fine_history_window: int = 64
    # Quantile threshold q in (0, 1): window_motion <= quantile(history, q) => fine.
    coarse_fine_quantile: float = 0.3
    # Minimum number of historical windows before enabling quantile-based fine decisions.
    coarse_fine_min_history: int = 16
    # Hysteresis windows required before switching coarse<->fine in inference.
    coarse_fine_hysteresis: int = 2
    # Weights in motion scalar: S = L_trans + w_rot * L_rot + w_gripper * L_gripper.
    coarse_fine_rot_weight: float = 0.5
    coarse_fine_gripper_weight: float = 0.1
    # If true, replace multi-bank coarse/fine routing with a single-bank rank-gated LoRA path
    # in the action expert (PyTorch only).
    coarse_fine_rank_gating: bool = False
    # Controls coefficient mixing for PE-routed single-bank action-expert LoRA.
    # default: Bbase + p*Bp + e*Be + (p*e)*Bpe
    # bp_e_ebpe: Bbase + Bp + e*Be + e*Bpe
    # pbp_be_pbpe: Bbase + p*Bp + Be + p*Bpe
    # pbp_ebe_bpe: Bbase + p*Bp + e*Be + Bpe
    action_expert_pe_route_mode: Literal[
        "default",
        "bp_e_ebpe",
        "pbp_be_pbpe",
        "pbp_ebe_bpe",
    ] = "default"
    # If true, route_label is interpreted as a continuous score in [0, 1] rather than binary.
    coarse_fine_score_label: bool = False
    # Number of prefix actions used by motion-based label/statistics alignment.
    coarse_fine_label_prefix_steps: int = 5
    # GRU hidden width for the factorized P/E router trunk.
    coarse_fine_router_hidden_dim: int = 128
    # Number of action-history steps used by the router trunk.
    coarse_fine_router_history_steps: int = 6
    # Executed action count per historical chunk for router inputs (replan interval).
    coarse_fine_router_chunk_prefix_steps: int = 5
    # Physical action layout consumed by the PhaseLoRA router. This does not change
    # the policy action dimension; it only prevents padded actions (and, for ALOHA,
    # the second arm) from being silently discarded by the temporal encoder.
    coarse_fine_action_layout: Literal["libero_7d", "aloha_bimanual_14d"] = "libero_7d"
    # Bimanual PhaseLoRA predicts (P_l, E_l, P_r, E_r, C) and uses separate
    # left-arm, right-arm, and coordination adapter components. This preserves
    # the original single-arm P/E/PE parameterization for each arm without
    # assuming that both arms enter the same regime simultaneously.
    coarse_fine_bimanual_routing: bool = False
    # Auxiliary P/E supervision weights.
    coarse_fine_p_loss_weight: float = 0.5
    coarse_fine_e_loss_weight: float = 0.2
    coarse_fine_c_loss_weight: float = 0.2
    # Huber delta for P supervision.
    coarse_fine_p_huber_delta: float = 0.1

    # If true, replace standard LoRA delta BA with input-conditioned spectral LoRA (PyTorch only).
    lora_sp_enabled: bool = False
    # Initial adapter rank r for LoRA-SP. If None, reuse each layer's original LoRA rank from the model variant.
    lora_sp_rank: int | None = None
    # Cumulative energy threshold eta for dynamic active-rank selection.
    lora_sp_energy_threshold: float = 0.9
    # Hidden width of the 2-layer LoRA-SP router MLP.
    lora_sp_router_hidden_dim: int = 256
    # Hidden activation for the LoRA-SP router MLP.
    lora_sp_router_activation: Literal["silu", "gelu", "relu"] = "silu"
    # Nonnegative function used to map router outputs to gating scores.
    lora_sp_router_nonnegative: Literal["softplus", "relu", "exp"] = "softplus"
    # Numerical stability epsilon for energy normalization.
    lora_sp_eps: float = 1e-8
    # Weight for spectral auxiliary loss mean(1 - E_k).
    lora_sp_spec_loss_weight: float = 1e-2
    # Weight for router auxiliary loss.
    lora_sp_router_loss_weight: float = 1e-3
    # Internal balance term weight in LoRA-SP router loss.
    lora_sp_router_balance_weight: float = 1.0
    # Internal z-loss weight in LoRA-SP router loss.
    lora_sp_router_z_loss_weight: float = 1.0
    # If true, inference computes adapter output using only selected active ranks.
    lora_sp_inference_prune: bool = True

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.phase_gating and self.coarse_fine_shared_routing:
            raise ValueError("phase_gating and coarse_fine_shared_routing cannot both be enabled")
        if self.coarse_fine_rank_gating and (self.phase_gating or self.coarse_fine_shared_routing):
            raise ValueError(
                "coarse_fine_rank_gating is mutually exclusive with phase_gating and coarse_fine_shared_routing"
            )
        if self.lora_bank_count < 1:
            raise ValueError(f"lora_bank_count must be >= 1, got {self.lora_bank_count}")
        vlm_bank_count = self.lora_bank_count if self.vlm_lora_bank_count is None else self.vlm_lora_bank_count
        action_bank_count = (
            self.lora_bank_count if self.action_expert_lora_bank_count is None else self.action_expert_lora_bank_count
        )
        if vlm_bank_count < 1:
            raise ValueError(f"vlm_lora_bank_count must be >= 1, got {vlm_bank_count}")
        if action_bank_count < 1:
            raise ValueError(f"action_expert_lora_bank_count must be >= 1, got {action_bank_count}")

        if self.coarse_fine_shared_routing and action_bank_count < 3:
            raise ValueError(
                "coarse_fine_shared_routing requires action_expert_lora_bank_count >= 3 (fine, coarse, shared)"
            )
        if self.coarse_fine_shared_routing and vlm_bank_count not in {1, action_bank_count}:
            raise ValueError(
                "coarse_fine_shared_routing requires vlm_lora_bank_count to be either 1 "
                f"or action_expert_lora_bank_count ({action_bank_count}), got {vlm_bank_count}"
            )
        if self.phase_count < 1:
            raise ValueError(f"phase_count must be >= 1, got {self.phase_count}")
        if self.phase_gating:
            if vlm_bank_count > 1 and vlm_bank_count != self.phase_count:
                raise ValueError(
                    "phase_gating requires vlm_lora_bank_count to be either 1 "
                    f"or phase_count ({self.phase_count}), got {vlm_bank_count}"
                )
            if action_bank_count > 1 and action_bank_count != self.phase_count:
                raise ValueError(
                    "phase_gating requires action_expert_lora_bank_count to be either 1 "
                    f"or phase_count ({self.phase_count}), got {action_bank_count}"
                )
        if self.phase_router_hidden_dim < 8:
            raise ValueError(f"phase_router_hidden_dim must be >= 8, got {self.phase_router_hidden_dim}")
        if not (0.0 < self.coarse_fine_quantile < 1.0):
            raise ValueError(f"coarse_fine_quantile must be in (0, 1), got {self.coarse_fine_quantile}")
        if self.coarse_fine_history_window < 1:
            raise ValueError(f"coarse_fine_history_window must be >= 1, got {self.coarse_fine_history_window}")
        if self.coarse_fine_min_history < 1:
            raise ValueError(f"coarse_fine_min_history must be >= 1, got {self.coarse_fine_min_history}")
        if self.coarse_fine_min_history > self.coarse_fine_history_window:
            raise ValueError(
                "coarse_fine_min_history cannot exceed coarse_fine_history_window: "
                f"{self.coarse_fine_min_history} > {self.coarse_fine_history_window}"
            )
        if self.coarse_fine_hysteresis < 1:
            raise ValueError(f"coarse_fine_hysteresis must be >= 1, got {self.coarse_fine_hysteresis}")
        if self.coarse_fine_rot_weight < 0.0:
            raise ValueError(f"coarse_fine_rot_weight must be >= 0, got {self.coarse_fine_rot_weight}")
        if self.coarse_fine_gripper_weight < 0.0:
            raise ValueError(f"coarse_fine_gripper_weight must be >= 0, got {self.coarse_fine_gripper_weight}")
        if self.coarse_fine_label_prefix_steps < 1:
            raise ValueError(f"coarse_fine_label_prefix_steps must be >= 1, got {self.coarse_fine_label_prefix_steps}")
        if self.coarse_fine_router_hidden_dim < 8:
            raise ValueError(f"coarse_fine_router_hidden_dim must be >= 8, got {self.coarse_fine_router_hidden_dim}")
        if self.coarse_fine_router_history_steps < 2:
            raise ValueError(
                f"coarse_fine_router_history_steps must be >= 2, got {self.coarse_fine_router_history_steps}"
            )
        if self.coarse_fine_router_chunk_prefix_steps < 1:
            raise ValueError(
                f"coarse_fine_router_chunk_prefix_steps must be >= 1, got {self.coarse_fine_router_chunk_prefix_steps}"
            )
        if self.coarse_fine_action_layout not in {"libero_7d", "aloha_bimanual_14d"}:
            raise ValueError(
                "coarse_fine_action_layout must be 'libero_7d' or 'aloha_bimanual_14d', got "
                f"{self.coarse_fine_action_layout!r}"
            )
        if self.coarse_fine_bimanual_routing:
            if not self.coarse_fine_rank_gating:
                raise ValueError("coarse_fine_bimanual_routing requires coarse_fine_rank_gating")
            if self.coarse_fine_action_layout != "aloha_bimanual_14d":
                raise ValueError("coarse_fine_bimanual_routing requires coarse_fine_action_layout='aloha_bimanual_14d'")
        if self.coarse_fine_p_loss_weight < 0.0:
            raise ValueError(f"coarse_fine_p_loss_weight must be >= 0, got {self.coarse_fine_p_loss_weight}")
        if self.coarse_fine_e_loss_weight < 0.0:
            raise ValueError(f"coarse_fine_e_loss_weight must be >= 0, got {self.coarse_fine_e_loss_weight}")
        if self.coarse_fine_c_loss_weight < 0.0:
            raise ValueError(f"coarse_fine_c_loss_weight must be >= 0, got {self.coarse_fine_c_loss_weight}")
        if self.coarse_fine_p_huber_delta <= 0.0:
            raise ValueError(f"coarse_fine_p_huber_delta must be > 0, got {self.coarse_fine_p_huber_delta}")
        if self.coarse_fine_rank_gating:
            if action_bank_count != 1:
                raise ValueError(
                    f"coarse_fine_rank_gating requires action_expert_lora_bank_count == 1, got {action_bank_count}"
                )
            if "lora" not in str(self.action_expert_variant):
                raise ValueError("coarse_fine_rank_gating requires a LoRA action expert variant")
            valid_route_modes = {"default", "bp_e_ebpe", "pbp_be_pbpe", "pbp_ebe_bpe"}
            if self.action_expert_pe_route_mode not in valid_route_modes:
                raise ValueError(
                    "action_expert_pe_route_mode must be one of "
                    f"{sorted(valid_route_modes)}, got {self.action_expert_pe_route_mode}"
                )

        if self.lora_sp_enabled:
            if self.phase_gating or self.coarse_fine_shared_routing or self.coarse_fine_rank_gating:
                raise ValueError("lora_sp_enabled is mutually exclusive with phase_gating/coarse_fine routing modes")
            if vlm_bank_count != 1 or action_bank_count != 1:
                raise ValueError(
                    "lora_sp_enabled requires vlm_lora_bank_count == 1 and action_expert_lora_bank_count == 1"
                )
            if "lora" not in str(self.paligemma_variant):
                raise ValueError("lora_sp_enabled requires a LoRA paligemma variant")
            if "lora" not in str(self.action_expert_variant):
                raise ValueError("lora_sp_enabled requires a LoRA action expert variant")

        if self.lora_sp_rank is not None and self.lora_sp_rank < 1:
            raise ValueError(f"lora_sp_rank must be >= 1 when set, got {self.lora_sp_rank}")
        if not (0.0 < self.lora_sp_energy_threshold <= 1.0):
            raise ValueError(f"lora_sp_energy_threshold must be in (0, 1], got {self.lora_sp_energy_threshold}")
        if self.lora_sp_router_hidden_dim < 8:
            raise ValueError(f"lora_sp_router_hidden_dim must be >= 8, got {self.lora_sp_router_hidden_dim}")
        if self.lora_sp_eps <= 0.0:
            raise ValueError(f"lora_sp_eps must be > 0, got {self.lora_sp_eps}")
        if self.lora_sp_spec_loss_weight < 0.0:
            raise ValueError(f"lora_sp_spec_loss_weight must be >= 0, got {self.lora_sp_spec_loss_weight}")
        if self.lora_sp_router_loss_weight < 0.0:
            raise ValueError(f"lora_sp_router_loss_weight must be >= 0, got {self.lora_sp_router_loss_weight}")
        if self.lora_sp_router_balance_weight < 0.0:
            raise ValueError(f"lora_sp_router_balance_weight must be >= 0, got {self.lora_sp_router_balance_weight}")
        if self.lora_sp_router_z_loss_weight < 0.0:
            raise ValueError(f"lora_sp_router_z_loss_weight must be >= 0, got {self.lora_sp_router_z_loss_weight}")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
