# ruff: noqa: SLF001

from contextlib import contextmanager
import dataclasses
from typing import Literal

import pytest
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter on top of a frozen base linear layer."""

    _global_lora_enabled = True
    _global_lora_route_ids: torch.Tensor | None = None
    _global_lora_route_weights: torch.Tensor | None = None
    _global_lora_route_pe: torch.Tensor | None = None
    _global_lora_sp_stats: list[dict[str, torch.Tensor]] | None = None

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.lora_a = nn.Linear(base_linear.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base_linear.out_features, bias=False)

        nn.init.normal_(self.lora_a.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_linear(x)
        if not type(self)._global_lora_enabled:
            return base
        lora = self.lora_b(self.lora_a(x)) * self.scaling
        return base + lora

    @classmethod
    @contextmanager
    def lora_disabled(cls):
        prev = cls._global_lora_enabled
        cls._global_lora_enabled = False
        try:
            yield
        finally:
            cls._global_lora_enabled = prev

    @classmethod
    @contextmanager
    def lora_routing(
        cls,
        route_ids: torch.Tensor | None = None,
        route_weights: torch.Tensor | None = None,
        route_pe: torch.Tensor | None = None,
    ):
        prev_ids = cls._global_lora_route_ids
        prev_weights = cls._global_lora_route_weights
        prev_pe = cls._global_lora_route_pe
        cls._global_lora_route_ids = route_ids
        cls._global_lora_route_weights = route_weights
        cls._global_lora_route_pe = route_pe
        try:
            yield
        finally:
            cls._global_lora_route_ids = prev_ids
            cls._global_lora_route_weights = prev_weights
            cls._global_lora_route_pe = prev_pe

    @classmethod
    def reset_lora_sp_stats(cls) -> None:
        cls._global_lora_sp_stats = []

    @classmethod
    def pop_lora_sp_stats(cls) -> list[dict[str, torch.Tensor]]:
        stats = cls._global_lora_sp_stats or []
        cls._global_lora_sp_stats = None
        return stats

    @classmethod
    def record_lora_sp_stats(cls, stats: dict[str, torch.Tensor]) -> None:
        if cls._global_lora_sp_stats is not None:
            cls._global_lora_sp_stats.append(stats)

    @property
    def weight(self):
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias


@dataclasses.dataclass(frozen=True)
class LoRASPConfig:
    enabled: bool = False
    rank: int | None = None
    energy_threshold: float = 0.9
    router_hidden_dim: int = 256
    router_activation: Literal["silu", "gelu", "relu"] = "silu"
    router_nonnegative: Literal["softplus", "relu", "exp"] = "softplus"
    eps: float = 1e-8
    router_balance_weight: float = 1.0
    router_z_loss_weight: float = 1.0
    inference_prune: bool = True


class SpectralLoRALinear(nn.Module):
    """Input-conditioned SVD-style LoRA: DeltaW(x) = U diag(s(x)) V."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        *,
        energy_threshold: float,
        router_hidden_dim: int,
        router_activation: Literal["silu", "gelu", "relu"],
        router_nonnegative: Literal["softplus", "relu", "exp"],
        eps: float,
        router_balance_weight: float,
        router_z_loss_weight: float,
        inference_prune: bool,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA-SP rank must be positive, got {rank}")
        if not (0.0 < energy_threshold <= 1.0):
            raise ValueError(f"energy_threshold must be in (0, 1], got {energy_threshold}")
        if router_hidden_dim <= 0:
            raise ValueError(f"router_hidden_dim must be positive, got {router_hidden_dim}")

        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.energy_threshold = energy_threshold
        self.router_activation = router_activation
        self.router_nonnegative = router_nonnegative
        self.eps = max(1e-12, float(eps))
        self.router_balance_weight = max(0.0, float(router_balance_weight))
        self.router_z_loss_weight = max(0.0, float(router_z_loss_weight))
        self.inference_prune = bool(inference_prune)

        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.lora_v = nn.Linear(base_linear.in_features, rank, bias=False)
        self.lora_u = nn.Linear(rank, base_linear.out_features, bias=False)
        self.router_fc1 = nn.Linear(base_linear.in_features, router_hidden_dim)
        self.router_fc2 = nn.Linear(router_hidden_dim, rank)

        nn.init.normal_(self.lora_v.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_u.weight)
        nn.init.xavier_uniform_(self.router_fc1.weight)
        nn.init.zeros_(self.router_fc1.bias)
        nn.init.zeros_(self.router_fc2.weight)
        nn.init.zeros_(self.router_fc2.bias)

    def _apply_router_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.router_activation == "silu":
            return F.silu(x)
        if self.router_activation == "gelu":
            return F.gelu(x)
        if self.router_activation == "relu":
            return F.relu(x)
        raise ValueError(f"Unsupported router activation: {self.router_activation}")

    def _apply_nonnegative(self, x: torch.Tensor) -> torch.Tensor:
        if self.router_nonnegative == "softplus":
            return F.softplus(x)
        if self.router_nonnegative == "relu":
            return F.relu(x)
        if self.router_nonnegative == "exp":
            return torch.exp(torch.clamp(x, max=20.0))
        raise ValueError(f"Unsupported nonnegative mapping: {self.router_nonnegative}")

    def _compute_router_scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._apply_router_activation(self.router_fc1(x))
        router_logits = self.router_fc2(hidden)
        scores = self._apply_nonnegative(router_logits)
        return scores, router_logits

    def _energy_prune(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy = scores.square()
        sorted_energy, sorted_idx = torch.sort(energy, dim=-1, descending=True)
        energy_sum = sorted_energy.sum(dim=-1, keepdim=True)
        safe_sum = energy_sum.clamp_min(self.eps)
        cumulative_energy = torch.cumsum(sorted_energy, dim=-1) / safe_sum

        meets_threshold = cumulative_energy >= self.energy_threshold
        first_idx = torch.argmax(meets_threshold.to(dtype=torch.int64), dim=-1)
        has_meet = meets_threshold.any(dim=-1)
        fallback_idx = torch.full_like(first_idx, self.rank - 1)
        k_idx = torch.where(has_meet, first_idx, fallback_idx)

        zero_energy = energy_sum.squeeze(-1) <= self.eps
        k_idx = torch.where(zero_energy, torch.zeros_like(k_idx), k_idx)

        rank_positions = torch.arange(self.rank, device=scores.device)
        view_shape = [1] * (scores.ndim - 1) + [self.rank]
        rank_positions = rank_positions.view(*view_shape)
        sorted_mask = rank_positions <= k_idx.unsqueeze(-1)

        mask = torch.zeros_like(scores)
        mask.scatter_(-1, sorted_idx, sorted_mask.to(dtype=scores.dtype))

        energy_at_k = torch.gather(cumulative_energy, -1, k_idx.unsqueeze(-1)).squeeze(-1)
        energy_at_k = torch.where(zero_energy, torch.ones_like(energy_at_k), energy_at_k)
        active_rank = (k_idx + 1).to(dtype=scores.dtype)
        return mask, energy_at_k, active_rank

    def _sparse_inference_projection(
        self,
        latent: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        out_dim = self.base_linear.out_features
        flat_latent = latent.reshape(-1, self.rank)
        flat_scores = scores.reshape(-1, self.rank)
        flat_mask = mask.reshape(-1, self.rank).to(dtype=torch.bool)

        active_counts = flat_mask.sum(dim=-1)
        max_active = int(active_counts.max().item()) if active_counts.numel() > 0 else 0
        if max_active == 0:
            return torch.zeros(*latent.shape[:-1], out_dim, device=latent.device, dtype=latent.dtype)

        # Fast-path: if all ranks are active, use dense projection.
        if max_active >= self.rank:
            dense_out = self.lora_u(flat_latent * flat_scores)
            return dense_out.reshape(*latent.shape[:-1], out_dim)

        # Gather active ranks per token and perform a batched projected sum.
        topk_idx = torch.topk(flat_mask.to(dtype=torch.float32), k=max_active, dim=-1).indices
        selected_valid = torch.gather(flat_mask, dim=-1, index=topk_idx)
        selected_latent = torch.gather(flat_latent, dim=-1, index=topk_idx)
        selected_scores = torch.gather(flat_scores, dim=-1, index=topk_idx)
        weighted_latent = selected_latent * selected_scores * selected_valid.to(dtype=latent.dtype)

        u_weight_t = self.lora_u.weight.to(dtype=latent.dtype).t()  # [rank, out_dim]
        selected_u = u_weight_t[topk_idx]  # [tokens, max_active, out_dim]
        out_flat = torch.bmm(weighted_latent.unsqueeze(1), selected_u).squeeze(1)
        return out_flat.reshape(*latent.shape[:-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_linear(x)
        if not LoRALinear._global_lora_enabled:
            return base

        latent = self.lora_v(x)
        scores, router_logits = self._compute_router_scores(x)
        mask, energy_at_k, active_rank = self._energy_prune(scores)
        scores_pruned = scores * mask

        if self.training or not self.inference_prune:
            lora = self.lora_u(latent * scores_pruned)
        else:
            lora = self._sparse_inference_projection(latent, scores, mask)
        lora = lora * self.scaling

        spec_loss = 1.0 - energy_at_k

        energy = scores.square()
        norm_energy = energy / energy.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        reduce_dims = tuple(range(norm_energy.ndim - 1))
        rank_usage = norm_energy.mean(dim=reduce_dims) if reduce_dims else norm_energy
        uniform = torch.full_like(rank_usage, 1.0 / float(self.rank))
        router_balance_loss = torch.mean((rank_usage - uniform) ** 2)
        router_z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)
        router_loss = self.router_balance_weight * router_balance_loss + self.router_z_loss_weight * router_z_loss

        LoRALinear.record_lora_sp_stats(
            {
                "spec_loss": spec_loss.mean(),
                "router_loss": router_loss,
                "router_balance_loss": router_balance_loss,
                "router_z_loss": router_z_loss,
                "active_rank_mean": active_rank.mean(),
                "energy_at_k_mean": energy_at_k.mean(),
            }
        )

        return base + lora

    @property
    def weight(self):
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias


class MultiBankLoRALinear(nn.Module):
    """Linear layer with multiple LoRA banks, selected by global routing state."""

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float, bank_count: int):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if bank_count <= 1:
            raise ValueError(f"bank_count must be > 1 for MultiBankLoRALinear, got {bank_count}")

        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.bank_count = bank_count
        self.scaling = alpha / rank

        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.lora_a = nn.ModuleList([nn.Linear(base_linear.in_features, rank, bias=False) for _ in range(bank_count)])
        self.lora_b = nn.ModuleList([nn.Linear(rank, base_linear.out_features, bias=False) for _ in range(bank_count)])

        for lora_a, lora_b in zip(self.lora_a, self.lora_b, strict=True):
            nn.init.normal_(lora_a.weight, mean=0.0, std=0.01)
            nn.init.zeros_(lora_b.weight)

    def _resolve_route_weights(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        route_weights = LoRALinear._global_lora_route_weights
        if route_weights is not None:
            if route_weights.shape[-1] != self.bank_count:
                raise ValueError(
                    f"Route weight bank dimension mismatch: expected {self.bank_count}, got {route_weights.shape[-1]}"
                )
            if route_weights.ndim == 1:
                route_weights = route_weights.unsqueeze(0)
            if route_weights.shape[0] == 1 and x.shape[0] > 1:
                route_weights = route_weights.expand(x.shape[0], -1)
            if route_weights.shape[0] != x.shape[0]:
                raise ValueError(
                    f"Route weight batch dimension mismatch: expected {x.shape[0]}, got {route_weights.shape[0]}"
                )
            return route_weights.to(device=x.device, dtype=dtype)

        route_ids = LoRALinear._global_lora_route_ids
        if route_ids is None:
            route_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if route_ids.ndim == 0:
            route_ids = route_ids.unsqueeze(0)
        if route_ids.shape[0] == 1 and x.shape[0] > 1:
            route_ids = route_ids.expand(x.shape[0])
        if route_ids.shape[0] != x.shape[0]:
            raise ValueError(f"Route id batch dimension mismatch: expected {x.shape[0]}, got {route_ids.shape[0]}")
        route_ids = route_ids.to(device=x.device, dtype=torch.long).clamp(min=0, max=self.bank_count - 1)
        return F.one_hot(route_ids, num_classes=self.bank_count).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_linear(x)
        if not LoRALinear._global_lora_enabled:
            return base

        route_weights = self._resolve_route_weights(x, base.dtype)

        lora_outputs = []
        for lora_a, lora_b in zip(self.lora_a, self.lora_b, strict=True):
            lora_outputs.append(lora_b(lora_a(x)) * self.scaling)

        stacked = torch.stack(lora_outputs, dim=1)
        while route_weights.ndim < stacked.ndim:
            route_weights = route_weights.unsqueeze(-1)
        lora = torch.sum(stacked * route_weights, dim=1)
        return base + lora

    @property
    def weight(self):
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias


class PERoutedLoRALinear(nn.Module):
    """Single-bank LoRA with PE-conditioned B matrix: DeltaW(E, P) = B(E, P)A."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        route_mode: Literal["default", "bp_e_ebpe", "pbp_be_pbpe", "pbp_ebe_bpe"] = "default",
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.route_mode = route_mode

        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.lora_a = nn.Linear(base_linear.in_features, rank, bias=False)
        self.lora_b_base = nn.Linear(rank, base_linear.out_features, bias=False)
        self.lora_b_p = nn.Linear(rank, base_linear.out_features, bias=False)
        self.lora_b_e = nn.Linear(rank, base_linear.out_features, bias=False)
        self.lora_b_pe = nn.Linear(rank, base_linear.out_features, bias=False)

        nn.init.normal_(self.lora_a.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_b_base.weight)
        nn.init.zeros_(self.lora_b_p.weight)
        nn.init.zeros_(self.lora_b_e.weight)
        nn.init.zeros_(self.lora_b_pe.weight)

    def _compute_route_coeffs(
        self,
        p: torch.Tensor,
        e: torch.Tensor,
        pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.route_mode == "default":
            return p, e, pe
        if self.route_mode == "bp_e_ebpe":
            return torch.ones_like(p), e, e
        if self.route_mode == "pbp_be_pbpe":
            return p, torch.ones_like(e), p
        if self.route_mode == "pbp_ebe_bpe":
            return p, e, torch.ones_like(pe)
        raise ValueError(f"Unsupported PE route mode: {self.route_mode}")

    def _resolve_route_pe(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        route_pe = LoRALinear._global_lora_route_pe
        if route_pe is None:
            route_pe = torch.full((x.shape[0], 2), 0.5, device=x.device, dtype=torch.float32)
        elif not isinstance(route_pe, torch.Tensor):
            route_pe = torch.as_tensor(route_pe, device=x.device, dtype=torch.float32)
        else:
            route_pe = route_pe.to(device=x.device, dtype=torch.float32)

        if route_pe.ndim == 0:
            route_pe = route_pe.reshape(1, 1)
        if route_pe.ndim == 1:
            if route_pe.shape[0] == 2:
                route_pe = route_pe.unsqueeze(0)
            else:
                raise ValueError(f"Route PE must contain 2 values per sample, got shape {tuple(route_pe.shape)}")
        if route_pe.ndim > 2:
            route_pe = route_pe.reshape(route_pe.shape[0], -1)
        if route_pe.shape[-1] < 2:
            raise ValueError(f"Route PE feature mismatch: expected at least 2 features, got {route_pe.shape[-1]}")
        if route_pe.shape[-1] > 2:
            route_pe = route_pe[:, :2]
        if route_pe.shape[0] == 1 and x.shape[0] > 1:
            route_pe = route_pe.expand(x.shape[0], -1)
        if route_pe.shape[0] != x.shape[0]:
            raise ValueError(f"Route PE batch mismatch: expected {x.shape[0]}, got {route_pe.shape[0]}")
        return route_pe.clamp(0.0, 1.0).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_linear(x)
        if not LoRALinear._global_lora_enabled:
            return base

        route_pe = self._resolve_route_pe(x, base.dtype)
        p = route_pe[:, 0]
        e = route_pe[:, 1]
        pe = p * e
        c_bp, c_be, c_bpe = self._compute_route_coeffs(p, e, pe)

        a_out = self.lora_a(x)
        lora = self.lora_b_base(a_out)

        while c_bp.ndim < lora.ndim:
            c_bp = c_bp.unsqueeze(-1)
        while c_be.ndim < lora.ndim:
            c_be = c_be.unsqueeze(-1)
        while c_bpe.ndim < lora.ndim:
            c_bpe = c_bpe.unsqueeze(-1)

        lora = lora + c_bp * self.lora_b_p(a_out) + c_be * self.lora_b_e(a_out) + c_bpe * self.lora_b_pe(a_out)
        lora = lora * self.scaling
        return base + lora

    @property
    def weight(self):
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias


class BimanualPERoutedLoRALinear(nn.Module):
    """PhaseLoRA conditioned by arm-specific P/E and coordination descriptors.

    Delta W = (B0
               + P_l B_Pl + E_l B_El + P_l E_l B_PEl
               + P_r B_Pr + E_r B_Er + P_r E_r B_PEr
               + C B_C) A.

    A remains shared, matching the original PhaseLoRA factorization, while the
    two arms no longer have to share the same regime.
    """

    ROUTE_DIM = 5

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        for param in self.base_linear.parameters():
            param.requires_grad = False

        self.lora_a = nn.Linear(base_linear.in_features, rank, bias=False)
        self.lora_b_base = nn.Linear(rank, base_linear.out_features, bias=False)
        self.lora_b_p_left = nn.Linear(rank, base_linear.out_features, bias=False)
        self.lora_b_e_left = nn.Linear(rank, base_linear.out_features, bias=False)
        self.lora_b_pe_left = nn.Linear(rank, base_linear.out_features, bias=False)
        self.lora_b_p_right = nn.Linear(rank, base_linear.out_features, bias=False)
        self.lora_b_e_right = nn.Linear(rank, base_linear.out_features, bias=False)
        self.lora_b_pe_right = nn.Linear(rank, base_linear.out_features, bias=False)
        self.lora_b_coordination = nn.Linear(rank, base_linear.out_features, bias=False)

        nn.init.normal_(self.lora_a.weight, mean=0.0, std=0.01)
        for module in (
            self.lora_b_base,
            self.lora_b_p_left,
            self.lora_b_e_left,
            self.lora_b_pe_left,
            self.lora_b_p_right,
            self.lora_b_e_right,
            self.lora_b_pe_right,
            self.lora_b_coordination,
        ):
            nn.init.zeros_(module.weight)

    def _resolve_route(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        route = LoRALinear._global_lora_route_pe
        if route is None:
            route = torch.full(
                (x.shape[0], self.ROUTE_DIM),
                0.5,
                device=x.device,
                dtype=torch.float32,
            )
        elif not isinstance(route, torch.Tensor):
            route = torch.as_tensor(route, device=x.device, dtype=torch.float32)
        else:
            route = route.to(device=x.device, dtype=torch.float32)

        if route.ndim == 1:
            route = route.unsqueeze(0)
        if route.ndim > 2:
            route = route.reshape(route.shape[0], -1)
        if route.shape[-1] != self.ROUTE_DIM:
            raise ValueError(
                f"Bimanual route must contain {self.ROUTE_DIM} values "
                f"(P_l,E_l,P_r,E_r,C), got shape {tuple(route.shape)}"
            )
        if route.shape[0] == 1 and x.shape[0] > 1:
            route = route.expand(x.shape[0], -1)
        if route.shape[0] != x.shape[0]:
            raise ValueError(f"Route batch mismatch: expected {x.shape[0]}, got {route.shape[0]}")
        return route.clamp(0.0, 1.0).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_linear(x)
        if not LoRALinear._global_lora_enabled:
            return base

        route = self._resolve_route(x, base.dtype)
        p_left, e_left, p_right, e_right, coordination = route.unbind(dim=-1)
        coeffs = (
            p_left,
            e_left,
            p_left * e_left,
            p_right,
            e_right,
            p_right * e_right,
            coordination,
        )
        modules = (
            self.lora_b_p_left,
            self.lora_b_e_left,
            self.lora_b_pe_left,
            self.lora_b_p_right,
            self.lora_b_e_right,
            self.lora_b_pe_right,
            self.lora_b_coordination,
        )

        a_out = self.lora_a(x)
        lora = self.lora_b_base(a_out)
        for coefficient, module in zip(coeffs, modules, strict=True):
            expanded_coefficient = coefficient
            while expanded_coefficient.ndim < lora.ndim:
                expanded_coefficient = expanded_coefficient.unsqueeze(-1)
            lora = lora + expanded_coefficient * module(a_out)
        return base + lora * self.scaling

    @property
    def weight(self):
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias


def _make_lora_linear(
    base_linear: nn.Linear,
    rank: int,
    alpha: float,
    bank_count: int,
    *,
    pe_routed: bool = False,
    bimanual_pe_routed: bool = False,
    pe_route_mode: Literal["default", "bp_e_ebpe", "pbp_be_pbpe", "pbp_ebe_bpe"] = "default",
    lora_sp_config: LoRASPConfig | None = None,
) -> nn.Module:
    if pe_routed and bimanual_pe_routed:
        raise ValueError("pe_routed and bimanual_pe_routed are mutually exclusive")
    if lora_sp_config is not None and lora_sp_config.enabled:
        if bank_count > 1:
            raise ValueError("LoRA-SP does not support multi-bank LoRA routing")
        if pe_routed or bimanual_pe_routed:
            raise ValueError("LoRA-SP cannot be combined with PE-routed LoRA")
        effective_rank = rank if lora_sp_config.rank is None else lora_sp_config.rank
        return SpectralLoRALinear(
            base_linear,
            rank=effective_rank,
            alpha=float(effective_rank),
            energy_threshold=lora_sp_config.energy_threshold,
            router_hidden_dim=lora_sp_config.router_hidden_dim,
            router_activation=lora_sp_config.router_activation,
            router_nonnegative=lora_sp_config.router_nonnegative,
            eps=lora_sp_config.eps,
            router_balance_weight=lora_sp_config.router_balance_weight,
            router_z_loss_weight=lora_sp_config.router_z_loss_weight,
            inference_prune=lora_sp_config.inference_prune,
        )
    if (pe_routed or bimanual_pe_routed) and bank_count > 1:
        raise ValueError("PE-routed LoRA only supports single-bank mode")
    if bank_count > 1:
        return MultiBankLoRALinear(base_linear, rank=rank, alpha=alpha, bank_count=bank_count)
    if pe_routed:
        return PERoutedLoRALinear(base_linear, rank=rank, alpha=alpha, route_mode=pe_route_mode)
    if bimanual_pe_routed:
        return BimanualPERoutedLoRALinear(base_linear, rank=rank, alpha=alpha)
    return LoRALinear(base_linear, rank=rank, alpha=alpha)


def _apply_lora_to_gemma_language_model(
    language_model: nn.Module,
    lora_configs: dict,
    *,
    bank_count: int = 1,
    pe_routed: bool = False,
    bimanual_pe_routed: bool = False,
    pe_route_mode: Literal["default", "bp_e_ebpe", "pbp_be_pbpe", "pbp_ebe_bpe"] = "default",
    lora_sp_config: LoRASPConfig | None = None,
) -> None:
    """Apply LoRA adapters to Gemma language model layers.

    Applies LoRA to attention projections (q/k/v/o) and MLP projections (gate/up/down)
    when corresponding lora configs are provided.
    """
    attn_lora = lora_configs.get("attn") if lora_configs else None
    ffn_lora = lora_configs.get("ffn") if lora_configs else None

    if attn_lora is None and ffn_lora is None:
        return

    for layer in language_model.layers:
        if attn_lora is not None:
            layer.self_attn.q_proj = _make_lora_linear(
                layer.self_attn.q_proj,
                rank=attn_lora.rank,
                alpha=attn_lora.alpha,
                bank_count=bank_count,
                pe_routed=pe_routed,
                bimanual_pe_routed=bimanual_pe_routed,
                pe_route_mode=pe_route_mode,
                lora_sp_config=lora_sp_config,
            )
            layer.self_attn.k_proj = _make_lora_linear(
                layer.self_attn.k_proj,
                rank=attn_lora.rank,
                alpha=attn_lora.alpha,
                bank_count=bank_count,
                pe_routed=pe_routed,
                bimanual_pe_routed=bimanual_pe_routed,
                pe_route_mode=pe_route_mode,
                lora_sp_config=lora_sp_config,
            )
            layer.self_attn.v_proj = _make_lora_linear(
                layer.self_attn.v_proj,
                rank=attn_lora.rank,
                alpha=attn_lora.alpha,
                bank_count=bank_count,
                pe_routed=pe_routed,
                bimanual_pe_routed=bimanual_pe_routed,
                pe_route_mode=pe_route_mode,
                lora_sp_config=lora_sp_config,
            )
            layer.self_attn.o_proj = _make_lora_linear(
                layer.self_attn.o_proj,
                rank=attn_lora.rank,
                alpha=attn_lora.alpha,
                bank_count=bank_count,
                pe_routed=pe_routed,
                bimanual_pe_routed=bimanual_pe_routed,
                pe_route_mode=pe_route_mode,
                lora_sp_config=lora_sp_config,
            )

        if ffn_lora is not None:
            layer.mlp.gate_proj = _make_lora_linear(
                layer.mlp.gate_proj,
                rank=ffn_lora.rank,
                alpha=ffn_lora.alpha,
                bank_count=bank_count,
                pe_routed=pe_routed,
                bimanual_pe_routed=bimanual_pe_routed,
                pe_route_mode=pe_route_mode,
                lora_sp_config=lora_sp_config,
            )
            layer.mlp.up_proj = _make_lora_linear(
                layer.mlp.up_proj,
                rank=ffn_lora.rank,
                alpha=ffn_lora.alpha,
                bank_count=bank_count,
                pe_routed=pe_routed,
                bimanual_pe_routed=bimanual_pe_routed,
                pe_route_mode=pe_route_mode,
                lora_sp_config=lora_sp_config,
            )
            layer.mlp.down_proj = _make_lora_linear(
                layer.mlp.down_proj,
                rank=ffn_lora.rank,
                alpha=ffn_lora.alpha,
                bank_count=bank_count,
                pe_routed=pe_routed,
                bimanual_pe_routed=bimanual_pe_routed,
                pe_route_mode=pe_route_mode,
                lora_sp_config=lora_sp_config,
            )


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        vlm_lora_bank_count: int = 1,
        action_expert_lora_bank_count: int = 1,
        *,
        action_expert_pe_routed: bool = False,
        action_expert_bimanual_pe_routed: bool = False,
        action_expert_pe_route_mode: Literal["default", "bp_e_ebpe", "pbp_be_pbpe", "pbp_ebe_bpe"] = "default",
        lora_sp_config: LoRASPConfig | None = None,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        _apply_lora_to_gemma_language_model(
            self.paligemma.language_model,
            vlm_config.lora_configs,
            bank_count=vlm_lora_bank_count,
            lora_sp_config=lora_sp_config,
        )
        _apply_lora_to_gemma_language_model(
            self.gemma_expert.model,
            action_expert_config.lora_configs,
            bank_count=action_expert_lora_bank_count,
            pe_routed=action_expert_pe_routed,
            bimanual_pe_routed=action_expert_bimanual_pe_routed,
            pe_route_mode=action_expert_pe_route_mode,
            lora_sp_config=lora_sp_config,
        )

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Snapshot LoRA routing context so gradient-checkpoint recomputation uses
            # the same route assignment as the original forward pass.
            checkpoint_route_ids = LoRALinear._global_lora_route_ids
            checkpoint_route_weights = LoRALinear._global_lora_route_weights
            checkpoint_route_pe = LoRALinear._global_lora_route_pe

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]
                with LoRALinear.lora_routing(
                    route_ids=checkpoint_route_ids,
                    route_weights=checkpoint_route_weights,
                    route_pe=checkpoint_route_pe,
                ):
                    query_states = []
                    key_states = []
                    value_states = []
                    gates = []
                    for i, hidden_states in enumerate(inputs_embeds):
                        layer = models[i].layers[layer_idx]
                        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                        gates.append(gate)

                        input_shape = hidden_states.shape[:-1]
                        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                        query_states.append(query_state)
                        key_states.append(key_state)
                        value_states.append(value_state)

                    # Concatenate and process attention
                    query_states = torch.cat(query_states, dim=2)
                    key_states = torch.cat(key_states, dim=2)
                    value_states = torch.cat(value_states, dim=2)

                    dummy_tensor = torch.zeros(
                        query_states.shape[0],
                        query_states.shape[2],
                        query_states.shape[-1],
                        device=query_states.device,
                        dtype=query_states.dtype,
                    )
                    cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                        query_states, key_states, cos, sin, unsqueeze_dim=1
                    )

                    batch_size = query_states.shape[0]
                    scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                    # Attention computation
                    att_output, _ = modeling_gemma.eager_attention_forward(
                        self.paligemma.language_model.layers[layer_idx].self_attn,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                        scaling,
                    )
                    # Get head_dim from the current layer, not from the model
                    head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                    # Process layer outputs
                    outputs_embeds = []
                    start_pos = 0
                    for i, hidden_states in enumerate(inputs_embeds):
                        layer = models[i].layers[layer_idx]
                        end_pos = start_pos + hidden_states.shape[1]

                        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                        # first residual
                        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])
                        after_first_residual = out_emb.clone()
                        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                            out_emb = out_emb.to(dtype=torch.bfloat16)

                        out_emb = layer.mlp(out_emb)
                        # second residual
                        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)
                        outputs_embeds.append(out_emb)
                        start_pos = end_pos

                    return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
