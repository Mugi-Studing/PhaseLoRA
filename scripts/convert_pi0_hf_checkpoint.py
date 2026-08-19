#!/usr/bin/env python3
"""Convert a HF-style PI0 checkpoint into OpenPI loadable format.

This utility fixes two incompatibilities:
1) strips the `model.` prefix from state_dict keys.
2) extracts normalization buffers into OpenPI assets/norm_stats.json.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

import numpy as np
from safetensors import safe_open
import safetensors.torch

from openpi.shared import normalize as _normalize

_NORM_MAP: dict[str, tuple[str, str]] = {
    "normalize_inputs.buffer_observation_state.mean": ("state", "mean"),
    "normalize_inputs.buffer_observation_state.std": ("state", "std"),
    "normalize_targets.buffer_action.mean": ("actions", "mean"),
    "normalize_targets.buffer_action.std": ("actions", "std"),
    "unnormalize_outputs.buffer_action.mean": ("actions_unnorm", "mean"),
    "unnormalize_outputs.buffer_action.std": ("actions_unnorm", "std"),
}

_NORM_PREFIXES = (
    "normalize_inputs.",
    "normalize_targets.",
    "unnormalize_outputs.",
)

_SIDECAR_NORM_MAP: dict[str, tuple[str, str]] = {
    "observation.state.mean": ("state", "mean"),
    "observation.state.std": ("state", "std"),
    "observation.state.q01": ("state", "q01"),
    "observation.state.q99": ("state", "q99"),
    "action.mean": ("actions", "mean"),
    "action.std": ("actions", "std"),
    "action.q01": ("actions", "q01"),
    "action.q99": ("actions", "q99"),
}


def _to_numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def _fill_norm_from_sidecars(checkpoint_dir: pathlib.Path, extracted: dict[str, dict[str, np.ndarray]]) -> None:
    """Backfill norm stats from policy processor sidecar safetensors if needed."""
    sidecars = [
        *sorted(checkpoint_dir.glob("policy_preprocessor_step_*_normalizer_processor.safetensors")),
        *sorted(checkpoint_dir.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors")),
    ]

    if not sidecars:
        return

    filled_fields = 0
    for sidecar in sidecars:
        tensors = safetensors.torch.load_file(str(sidecar), device="cpu")
        for key, (group, field) in _SIDECAR_NORM_MAP.items():
            if key not in tensors:
                continue
            if field in extracted.get(group, {}):
                continue
            extracted.setdefault(group, {})[field] = _to_numpy(tensors[key])
            filled_fields += 1

    if filled_fields > 0:
        print(f"[info] backfilled {filled_fields} norm fields from policy sidecar safetensors")


def _build_norm_stats(extracted: dict[str, dict[str, np.ndarray]]) -> dict[str, _normalize.NormStats]:
    if "state" not in extracted or "mean" not in extracted["state"] or "std" not in extracted["state"]:
        raise ValueError("Missing state normalization tensors in checkpoint.")

    has_actions = "actions" in extracted and "mean" in extracted["actions"] and "std" in extracted["actions"]
    has_actions_unnorm = (
        "actions_unnorm" in extracted and "mean" in extracted["actions_unnorm"] and "std" in extracted["actions_unnorm"]
    )

    if not has_actions and not has_actions_unnorm:
        raise ValueError("Missing action normalization tensors in checkpoint.")

    actions_src = extracted["actions"] if has_actions else extracted["actions_unnorm"]

    if has_actions and has_actions_unnorm:
        same_mean = np.allclose(extracted["actions"]["mean"], extracted["actions_unnorm"]["mean"], atol=1e-6)
        same_std = np.allclose(extracted["actions"]["std"], extracted["actions_unnorm"]["std"], atol=1e-6)
        if not (same_mean and same_std):
            print("[warn] normalize_targets and unnormalize_outputs action stats differ; using normalize_targets.")

    state_src = extracted["state"]
    state_q01 = state_src.get("q01")
    state_q99 = state_src.get("q99")

    action_q01 = actions_src.get("q01")
    action_q99 = actions_src.get("q99")

    return {
        "state": _normalize.NormStats(
            mean=state_src["mean"],
            std=state_src["std"],
            q01=state_q01,
            q99=state_q99,
        ),
        "actions": _normalize.NormStats(
            mean=actions_src["mean"],
            std=actions_src["std"],
            q01=action_q01,
            q99=action_q99,
        ),
    }


def convert_checkpoint(
    checkpoint_dir: pathlib.Path,
    output_checkpoint_dir: pathlib.Path,
    *,
    asset_id: str,
    backup_original: bool,
) -> None:
    source_model_path = checkpoint_dir / "model.safetensors"
    if not source_model_path.exists():
        raise FileNotFoundError(f"model.safetensors not found: {source_model_path}")

    output_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_model_path = output_checkpoint_dir / "model.safetensors"

    print(f"[info] loading {source_model_path}")
    state_dict = safetensors.torch.load_file(str(source_model_path), device="cpu")

    metadata = {}
    with safe_open(str(source_model_path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}

    converted_state_dict = {}
    extracted_norm_tensors: dict[str, dict[str, np.ndarray]] = {}

    stripped_prefix_count = 0
    dropped_norm_keys = 0

    for key, tensor in state_dict.items():
        if key in _NORM_MAP:
            stat_group, stat_name = _NORM_MAP[key]
            extracted_norm_tensors.setdefault(stat_group, {})[stat_name] = _to_numpy(tensor)
            dropped_norm_keys += 1
            continue

        if key.startswith(_NORM_PREFIXES):
            # Drop unknown normalization buffers so model loading is strict and clean.
            dropped_norm_keys += 1
            continue

        if key.startswith("model."):
            new_key = key[len("model.") :]
            stripped_prefix_count += 1
        else:
            new_key = key
        converted_state_dict[new_key] = tensor

    print(f"[info] stripped model. prefix from {stripped_prefix_count} keys")
    print(f"[info] removed {dropped_norm_keys} normalization buffer keys")
    print(f"[info] kept {len(converted_state_dict)} model keys")

    _fill_norm_from_sidecars(checkpoint_dir, extracted_norm_tensors)
    norm_stats = _build_norm_stats(extracted_norm_tensors)
    norm_stats_dir = output_checkpoint_dir / "assets" / asset_id
    _normalize.save(norm_stats_dir, norm_stats)
    print(f"[info] wrote norm stats to {norm_stats_dir / 'norm_stats.json'}")

    # Copy sidecar files when writing to a new directory.
    if checkpoint_dir != output_checkpoint_dir:
        for name in [
            "config.json",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "policy_preprocessor_step_2_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            "train_config.json",
            "README.md",
            ".gitattributes",
        ]:
            src = checkpoint_dir / name
            dst = output_checkpoint_dir / name
            if src.exists():
                shutil.copy2(src, dst)

    if checkpoint_dir == output_checkpoint_dir and backup_original:
        backup_path = checkpoint_dir / "model.safetensors.hf_raw"
        if backup_path.exists():
            raise FileExistsError(f"Backup already exists: {backup_path}")
        source_model_path.rename(backup_path)
        print(f"[info] moved original model to {backup_path}")

    print(f"[info] writing converted model to {output_model_path}")
    safetensors.torch.save_file(converted_state_dict, str(output_model_path), metadata=metadata)
    print("[info] conversion finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=pathlib.Path,
        required=True,
        help="Source checkpoint directory containing model.safetensors",
    )
    parser.add_argument(
        "--output-checkpoint-dir",
        type=pathlib.Path,
        default=None,
        help="Output checkpoint directory. Defaults to in-place conversion.",
    )
    parser.add_argument(
        "--asset-id",
        type=str,
        default="physical-intelligence/libero",
        help="Asset id path used under assets/ for norm_stats.json",
    )
    parser.add_argument(
        "--no-backup-original",
        action="store_true",
        help="Do not keep model.safetensors.hf_raw during in-place conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_checkpoint_dir = args.output_checkpoint_dir or args.checkpoint_dir

    convert_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        output_checkpoint_dir=output_checkpoint_dir,
        asset_id=args.asset_id,
        backup_original=not args.no_backup_original,
    )


if __name__ == "__main__":
    main()
