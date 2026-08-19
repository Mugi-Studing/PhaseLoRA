# ruff: noqa: SLF001

import math

import torch
from torch import nn

from openpi.models_pytorch.gemma_pytorch import BimanualPERoutedLoRALinear
from openpi.models_pytorch.gemma_pytorch import LoRALinear
from openpi.models_pytorch.gemma_pytorch import LoRASPConfig
from openpi.models_pytorch.gemma_pytorch import SpectralLoRALinear
from openpi.models_pytorch.gemma_pytorch import _make_lora_linear


def _build_spectral_layer(*, rank: int = 4, in_dim: int = 6, out_dim: int = 5) -> SpectralLoRALinear:
    base = nn.Linear(in_dim, out_dim, bias=False)
    return SpectralLoRALinear(
        base,
        rank=rank,
        alpha=float(rank),
        energy_threshold=0.9,
        router_hidden_dim=16,
        router_activation="silu",
        router_nonnegative="softplus",
        eps=1e-8,
        router_balance_weight=1.0,
        router_z_loss_weight=1.0,
        inference_prune=True,
    )


def test_spectral_lora_router_scores_are_nonnegative():
    layer = _build_spectral_layer(rank=5)
    x = torch.randn(2, 3, 6)
    scores, _ = layer._compute_router_scores(x)
    assert torch.all(scores >= 0.0)


def test_spectral_lora_energy_prune_selects_min_rank_by_threshold():
    layer = _build_spectral_layer(rank=4)
    scores = torch.tensor([[3.0, 2.0, 1.0, 0.5]], dtype=torch.float32)

    mask, energy_at_k, active_rank = layer._energy_prune(scores)

    expected_energy_at_k = (3.0**2 + 2.0**2) / (3.0**2 + 2.0**2 + 1.0**2 + 0.5**2)
    assert int(active_rank.item()) == 2
    assert torch.allclose(energy_at_k, torch.tensor([expected_energy_at_k], dtype=torch.float32), atol=1e-6)
    assert torch.equal(mask, torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32))


def test_spectral_lora_records_spec_loss_from_energy():
    layer = _build_spectral_layer(rank=4, in_dim=4, out_dim=3)
    x = torch.randn(1, 4)

    fixed_scores = torch.tensor([[3.0, 2.0, 1.0, 0.5]], dtype=torch.float32)
    fixed_logits = fixed_scores.log()

    def _fixed_router(_x: torch.Tensor):
        return fixed_scores.to(device=_x.device, dtype=_x.dtype), fixed_logits.to(device=_x.device, dtype=_x.dtype)

    layer._compute_router_scores = _fixed_router  # type: ignore[method-assign]

    LoRALinear.reset_lora_sp_stats()
    _ = layer(x)
    stats = LoRALinear.pop_lora_sp_stats()

    expected_energy_at_k = (3.0**2 + 2.0**2) / (3.0**2 + 2.0**2 + 1.0**2 + 0.5**2)
    expected_spec = 1.0 - expected_energy_at_k

    assert len(stats) == 1
    assert math.isclose(float(stats[0]["spec_loss"].item()), expected_spec, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(stats[0]["active_rank_mean"].item()), 2.0, rel_tol=0.0, abs_tol=1e-6)


def test_spectral_lora_sparse_inference_matches_dense_path():
    torch.manual_seed(7)
    layer = _build_spectral_layer(rank=4, in_dim=4, out_dim=3)
    nn.init.normal_(layer.lora_u.weight, mean=0.0, std=0.1)
    nn.init.normal_(layer.lora_v.weight, mean=0.0, std=0.1)

    x = torch.randn(2, 3, 4)

    fixed_scores = torch.tensor([[[4.0, 2.0, 1.0, 0.1]]], dtype=torch.float32)
    fixed_logits = fixed_scores.log()

    def _fixed_router(_x: torch.Tensor):
        score = fixed_scores.to(device=_x.device, dtype=_x.dtype).expand(_x.shape[0], _x.shape[1], -1)
        logits = fixed_logits.to(device=_x.device, dtype=_x.dtype).expand(_x.shape[0], _x.shape[1], -1)
        return score, logits

    layer._compute_router_scores = _fixed_router  # type: ignore[method-assign]

    layer.train()
    dense_out = layer(x)

    layer.eval()
    sparse_out = layer(x)

    assert torch.allclose(dense_out, sparse_out, atol=1e-5, rtol=1e-5)


def test_make_lora_linear_keeps_legacy_when_lora_sp_disabled():
    base = nn.Linear(4, 3, bias=False)
    linear = _make_lora_linear(
        base,
        rank=4,
        alpha=4.0,
        bank_count=1,
        pe_routed=False,
        lora_sp_config=LoRASPConfig(enabled=False),
    )
    assert isinstance(linear, LoRALinear)


def test_make_lora_linear_uses_spectral_when_lora_sp_enabled():
    base = nn.Linear(4, 3, bias=False)
    linear = _make_lora_linear(
        base,
        rank=4,
        alpha=4.0,
        bank_count=1,
        pe_routed=False,
        lora_sp_config=LoRASPConfig(enabled=True, rank=128, energy_threshold=0.9),
    )
    assert isinstance(linear, SpectralLoRALinear)
    assert linear.rank == 128


def test_make_lora_linear_uses_source_rank_when_lora_sp_rank_unset():
    base = nn.Linear(4, 3, bias=False)
    linear = _make_lora_linear(
        base,
        rank=96,
        alpha=96.0,
        bank_count=1,
        pe_routed=False,
        lora_sp_config=LoRASPConfig(enabled=True, rank=None, energy_threshold=0.9),
    )
    assert isinstance(linear, SpectralLoRALinear)
    assert linear.rank == 96


def test_bimanual_pe_routing_keeps_left_and_right_adapters_independent():
    base = nn.Linear(2, 1, bias=False)
    nn.init.zeros_(base.weight)
    layer = BimanualPERoutedLoRALinear(base, rank=2, alpha=2.0)
    with torch.no_grad():
        layer.lora_a.weight.copy_(torch.eye(2))
        layer.lora_b_p_left.weight.copy_(torch.tensor([[1.0, 0.0]]))
        layer.lora_b_p_right.weight.copy_(torch.tensor([[0.0, 1.0]]))

    inputs = torch.tensor([[2.0, 3.0]])
    with LoRALinear.lora_routing(route_pe=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])):
        left_output = layer(inputs)
    with LoRALinear.lora_routing(route_pe=torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])):
        right_output = layer(inputs)

    assert torch.allclose(left_output, torch.tensor([[2.0]]))
    assert torch.allclose(right_output, torch.tensor([[3.0]]))


def test_make_lora_linear_uses_bimanual_pe_router():
    linear = _make_lora_linear(
        nn.Linear(4, 3, bias=False),
        rank=4,
        alpha=4.0,
        bank_count=1,
        bimanual_pe_routed=True,
    )
    assert isinstance(linear, BimanualPERoutedLoRALinear)
