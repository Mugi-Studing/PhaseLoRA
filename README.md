# PhaseLoRA

This repository contains the official implementation of **PhaseLoRA: Control-Regime-Conditioned Low-Rank
Adaptation for Continuous-Action Vision-Language-Action Policies**. It is built on top of
[Physical Intelligence's OpenPI](https://github.com/Physical-Intelligence/openpi) and extends the upstream π₀/π₀.₅
model and policy interfaces with phase-aware low-rank adaptation. The codebase provides reproducible PyTorch
training, policy serving, and benchmark evaluation workflows for LIBERO and ALOHA.

## Highlights

- PyTorch fine-tuning for π₀ and π₀.₅, including LoRA, full-parameter training, DDP, optional ZeRO-1 optimizer
  sharding, checkpoint resume, and safetensors checkpoints.
- Phase-aware LoRA variants: multi-bank routing, single-bank P/E-routed adapters, bimanual P/E/coordination routing,
  and input-conditioned spectral LoRA with active-rank pruning.
- Optional flow-matching consistency losses, direct action regression, action-branch reinitialization, condition
  dropout, and frequency-aware image augmentation.
- A stateful websocket inference protocol with reset support, request timeouts, reproducible sampling seeds, optional
  externally supplied diffusion noise, and checkpoint provenance in server metadata.
- LIBERO evaluation with task-range resume, per-task and aggregate metrics, JSON snapshots, optional videos, and a
  fixed-initial-state/fixed-noise diagnostic.
- ALOHA simulation evaluation with configurable episode horizon and machine-readable metrics.

PhaseLoRA is implemented on the PyTorch model path. The original OpenPI JAX training and inference entry points
remain available for upstream-compatible configurations.

## Requirements

- Ubuntu 22.04 (the tested platform)
- Python 3.11
- An NVIDIA GPU with CUDA 12 support
- [`uv`](https://docs.astral.sh/uv/)

Approximate single-GPU memory requirements depend on the chosen configuration:

| Workload | Typical memory |
| --- | ---: |
| Inference | 8+ GB |
| LoRA fine-tuning | 24+ GB |
| Full fine-tuning | 70+ GB, or multiple GPUs with optimizer sharding |

## Installation

Clone with the upstream submodules and install the workspace:

```bash
git clone --recurse-submodules https://github.com/Grinffin/PhaseLoRA.git
cd openpi

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

The PyTorch implementation currently relies on the pinned Transformers compatibility patch:

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/
```

The project pins `transformers==4.53.2`; check it with `uv pip show transformers`. See
[`docs/docker.md`](docs/docker.md) for the container workflow.

## Data preparation

The upstream combined LIBERO dataset (`physical-intelligence/libero`) works with the standard configs. To create a
LeRobot dataset from the public RLDS releases instead:

```bash
uv run --group rlds examples/libero/convert_libero_data_to_lerobot.py \
  --data-dir /path/to/libero_rlds \
  --repo-name your_hf_username/libero
```

To publish one LeRobot repository per suite, add `--separate-subsets`. The resulting repository names are
`<repo-name>_libero_spatial`, `<repo-name>_libero_object`, `<repo-name>_libero_goal`, and
`<repo-name>_libero_10`.

Several experiment configs use `your_hf_username/...` placeholders. Override them from the command line, or replace
them in [`src/openpi/training/config.py`](src/openpi/training/config.py):

```bash
--data.repo-id your_hf_username/libero_libero_spatial
```

### Optional offline labels

Supervised PhaseLoRA configurations consume a precomputed JSON label mapping through
`--data.coarse-fine-label-path`. Pass a compatible label file when running a supervised routed configuration;
unsupervised and baseline configurations do not require this argument.

The training pipeline expects labels keyed by LeRobot sample index. With strict loading enabled, a missing sample
raises an error rather than silently using a fallback label.

## Base checkpoint preparation

Download and convert the upstream JAX checkpoint once:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
  --config_name pi05_libero \
  --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
```

Make the converted checkpoint discoverable without hard-coding a machine-specific path:

```bash
export OPENPI_PI05_BASE_CHECKPOINT="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch"
export OPENPI_PI0_BASE_CHECKPOINT="$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_base_pytorch"
```

You may instead pass `--pytorch-weight-path /absolute/path/to/checkpoint` to every training command. A converted
checkpoint directory must contain `model.safetensors`.

For Hugging Face-style PI0 checkpoints whose model keys have a `model.` prefix and whose normalization buffers are
stored in safetensors, use:

```bash
uv run scripts/convert_pi0_hf_checkpoint.py \
  --checkpoint-dir /path/to/hf_checkpoint \
  --output-checkpoint-dir /path/to/openpi_checkpoint \
  --asset-id physical-intelligence/libero
```

## Training

Representative configurations are listed below; the complete registry is in
[`src/openpi/training/config.py`](src/openpi/training/config.py).

| Configuration family | Example |
| --- | --- |
| Upstream π₀.₅ baseline | `pi05_libero` |
| Standard LoRA | `pi05_libero_low_mem_finetune` |
| Full fine-tuning | `pi05_libero_spatial_full_finetune` |
| VLM LoRA + full action expert | `pi05_libero_spatial_vlm_lora_action_full_finetune` |
| Spectral LoRA | `pi05_libero_spatial_lora_sp_baseline_finetune` |
| Single-bank P/E router | `pi05_libero_spatial_single_lora_tau_gated_router_finetune` |
| Multi-bank PhaseLoRA | `pi05_libero_spatial_phaselora_no_pe_supervision_finetune` |
| Bimanual PhaseLoRA | `pi05_aloha_sim_transfer_cube_bimanual_phaselora_rank48_96_finetune` |
| Direct action regression | `pi05_libero_low_mem_finetune_regression` |

First compute normalization statistics for the exact dataset/config combination:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_libero_spatial_single_lora_tau_gated_router_finetune \
  --repo-id your_hf_username/libero_libero_spatial
```

Then launch a single-GPU run:

```bash
uv run scripts/train_pytorch.py \
  pi05_libero_spatial_single_lora_tau_gated_router_finetune \
  --exp-name spatial_router_seed42 \
  --data.repo-id your_hf_username/libero_libero_spatial \
  --data.coarse-fine-label-path /path/to/spatial_labels.json \
  --pytorch-weight-path "$OPENPI_PI05_BASE_CHECKPOINT" \
  --overwrite
```

For distributed training, keep `batch_size` as the global batch size and use `torchrun`:

```bash
uv run torchrun --standalone --nproc-per-node=4 scripts/train_pytorch.py \
  pi05_libero_spatial_full_finetune \
  --exp-name spatial_full_ft \
  --data.repo-id your_hf_username/libero_libero_spatial \
  --pytorch-weight-path "$OPENPI_PI05_BASE_CHECKPOINT" \
  --shard-optimizer \
  --overwrite
```

Checkpoints are written to `checkpoints/<config>/<exp-name>/<step>/`. Use `--resume` to continue from the latest
valid step. `--resume` and `--overwrite` are mutually exclusive. Weights & Biases logging is enabled by default; use
`--no-wandb-enabled` for an offline run.

## Evaluation

### LIBERO

Start the policy server in the main environment:

```bash
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_libero_spatial_single_lora_tau_gated_router_finetune \
  --policy.dir checkpoints/<config>/<exp-name>/<step> \
  --port 8000 \
  --inference-seed 0
```

Run the simulator in the LIBERO environment described in [`examples/libero/README.md`](examples/libero/README.md):

```bash
python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 50 \
  --args.results-out-path results/libero_spatial.json \
  --args.video-out-path data/libero/videos
```

Useful evaluation options:

- `--args.start-task-id` and `--args.end-task-id` evaluate an inclusive task range.
- `--args.no-save-videos` disables video output.
- `--args.results-out-path` writes a progress snapshot after every completed task.
- `--args.request-timeout-s` controls the websocket inference timeout.

To evaluate the four public suites sequentially:

```bash
NUM_TRIALS=50 uv run bash run_libero.sh
```

The fixed-state noise diagnostic sends explicit diffusion noise through the websocket protocol and records each
rollout's video, noise array, and summary:

```bash
python examples/libero/eval_fixed_seed_noise_sweep.py \
  --args.task-suite-name libero_10 \
  --args.task-id 0 \
  --args.init-state-id 0 \
  --args.num-rollouts 8
```

### ALOHA simulation

After starting a compatible policy server:

```bash
uv run examples/aloha_sim/main.py \
  --num-episodes 10 \
  --max-episode-steps 400 \
  --metrics-path results/aloha_transfer_cube.json
```

The evaluator records per-episode seeds, rewards, steps, success flags, and an aggregate success rate.

### Router latency and memory

```bash
uv run scripts/benchmark_checkpoint_router_stats.py \
  --checkpoint CONFIG_NAME /path/to/checkpoint experiment_label \
  --warmup 5 \
  --runs 20
```

## Tests and style

```bash
uv run ruff check src scripts examples packages
uv run pytest -q -m "not manual"
```

The full upstream suite instantiates multi-billion-parameter models and needs substantial accelerator memory. The
portable CI check focuses on the new LoRA, transform, and client code:

```bash
uv run pytest -q \
  src/openpi/models_pytorch/gemma_pytorch_test.py \
  src/openpi/transforms_test.py
```

## Repository layout

```text
src/openpi/models_pytorch/   PyTorch model, LoRA, router, and spectral adapter implementations
src/openpi/training/         Training/data configuration and data loading
scripts/train_pytorch.py     Single-GPU and distributed PyTorch trainer
scripts/serve_policy.py      Policy server entry point
examples/libero/             LIBERO conversion and evaluation
examples/aloha_sim/          ALOHA simulation evaluation
packages/openpi-client/      Websocket client protocol
```

Keep normalization statistics with the checkpoint used for evaluation so that the policy and training data
transforms stay consistent.

## Acknowledgements and license

PhaseLoRA is built on the OpenPI codebase and model releases from
[Physical Intelligence](https://www.physicalintelligence.company/), plus LIBERO, LeRobot, Gemma, and the other
upstream projects listed in the dependency files and submodules.

OpenPI code is distributed under the terms in [`LICENSE`](LICENSE). Gemma components are covered by
[`LICENSE_GEMMA.txt`](LICENSE_GEMMA.txt). Third-party submodules retain their own licenses.
