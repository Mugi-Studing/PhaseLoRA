#!/usr/bin/env python3
"""Benchmark router parameter counts, policy latency, and peak GPU memory for trained checkpoints."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from statistics import mean
from statistics import pstdev
import time

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(f"PyTorch is required for this benchmark: {exc}") from exc

from openpi.policies import policy_config
from openpi.training import config as training_config

ROUTER_NAME_HINTS = ("tau_router", "phase_router", "lora_sp_router")


def _build_example(prompt: str, *, state_dim: int) -> dict[str, object]:
    rng = np.random.default_rng(0)
    base_image = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    wrist_image = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    state = np.zeros((state_dim,), dtype=np.float32)
    return {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": prompt,
    }


def _count_router_params(model: torch.nn.Module) -> dict[str, int]:
    totals = dict.fromkeys(ROUTER_NAME_HINTS, 0)
    total = 0
    for name, param in model.named_parameters():
        if any(hint in name for hint in ROUTER_NAME_HINTS):
            n = int(param.numel())
            total += n
            for hint in ROUTER_NAME_HINTS:
                if hint in name:
                    totals[hint] += n
    totals["router_total"] = total
    return totals


def _benchmark_policy(policy, example: dict[str, object], *, warmup: int, runs: int) -> dict[str, float]:
    if warmup < 0 or runs < 1:
        raise ValueError("warmup must be >= 0 and runs must be >= 1")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    policy.reset()

    for _ in range(warmup):
        _ = policy.infer(example)

    latencies_ms: list[float] = []
    peak_allocated_bytes: list[int] = []
    peak_reserved_bytes: list[int] = []

    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = policy.infer(example)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

        if torch.cuda.is_available():
            peak_allocated_bytes.append(torch.cuda.max_memory_allocated())
            peak_reserved_bytes.append(torch.cuda.max_memory_reserved())

    results = {
        "latency_mean_ms": float(mean(latencies_ms)),
        "latency_std_ms": float(pstdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0,
    }
    if peak_allocated_bytes:
        results["peak_inference_gpu_memory_allocated_mb"] = float(max(peak_allocated_bytes) / (1024.0 * 1024.0))
        results["peak_inference_gpu_memory_reserved_mb"] = float(max(peak_reserved_bytes) / (1024.0 * 1024.0))
    return results


def _load_policy(config_name: str, checkpoint_dir: pathlib.Path):
    train_cfg = training_config.get_config(config_name)
    data_cfg = train_cfg.data
    if dataclasses.is_dataclass(data_cfg) and hasattr(data_cfg, "coarse_fine_label_path"):
        data_cfg = dataclasses.replace(data_cfg, coarse_fine_label_path=None)
        train_cfg = dataclasses.replace(train_cfg, data=data_cfg)
    return policy_config.create_trained_policy(
        train_cfg,
        checkpoint_dir,
        pytorch_device="cuda" if torch.cuda.is_available() else "cpu",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark router params and inference latency for checkpoints")
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=3,
        metavar=("CONFIG_NAME", "CHECKPOINT_DIR", "LABEL"),
        help="Repeatable. Each entry is CONFIG_NAME CHECKPOINT_DIR LABEL.",
    )
    parser.add_argument("--prompt", type=str, default="put the object in the basket")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    if not args.checkpoint:
        parser.error("at least one --checkpoint CONFIG_NAME CHECKPOINT_DIR LABEL is required")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the GPU memory benchmark.")

    results = []
    for config_name, checkpoint_dir, label in args.checkpoint:
        checkpoint_path = pathlib.Path(checkpoint_dir)
        policy = _load_policy(config_name, checkpoint_path)
        model = policy._model  # noqa: SLF001 - benchmark intentionally inspects the wrapped model
        router_params = _count_router_params(model)
        state_dim = 8
        example = _build_example(args.prompt, state_dim=state_dim)
        benchmark = _benchmark_policy(policy, example, warmup=args.warmup, runs=args.runs)

        row = {
            "label": label,
            "config_name": config_name,
            "checkpoint_dir": str(checkpoint_path),
            **router_params,
            **benchmark,
        }
        results.append(row)

    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
