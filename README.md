# CAVIO

Cross-Attention Visual-Inertial Odometry (CAVIO): a PyTorch Lightning + Hydra codebase for training and evaluating latent-space VIO models.

## Acknowledgement

This repository is heavily influenced by the VIFT repository design and workflow.  
Many project patterns (training/eval structure, configuration style, and parts of the VIO pipeline) follow that prior codebase and are adapted here for CAVIO experiments.

VIFT repository: [https://github.com/ybkurt/vift](https://github.com/ybkurt/vift)

## Where to review the project work (CAVIO-specific)

The training stack, data loading pattern, pretrained Visual-Selective-VIO encoder, and much of the KITTI/latent plumbing follow the VIFT-style layout above. The items below are the most direct locus of the **cross-attention VIO** work and related experiments—good starting points for code review or grading.

### Cross-attention and ablation models

| Path | What it is |
|------|------------|
| `src/models/components/cavio.py` | `CAVIOPoseTransformer`: IMU queries visual latents, then causal self-attention (main CAVIO architecture). |
| `src/models/components/cavio_gated.py` | Gated residuals on cross-attention and FFN branches. |
| `src/models/components/cavio_visual_residual.py` | Extra learned visual residual into the IMU stream. |
| `src/models/components/imu_only.py` | IMU-only self-attention ablation (no cross-attention to vision). |

### Losses tied to those experiments

| Path | What it is |
|------|------------|
| `src/losses/weighted_loss.py` | RPMG pose losses; **per-axis / configurable** `CustomWeightedPoseLoss` and related classes used in sweeps. |

### Hydra configs for models and report experiments

| Path | What it is |
|------|------------|
| `configs/model/cavio.yaml` | Default Lightning module + `CAVIOPoseTransformer` + tester wiring. |
| `configs/model/imu_only.yaml` | IMU-only model group. |
| `configs/model/cavio_visual_residual.yaml` | Visual-residual model group. |
| `configs/experiment/` | Experiment presets (e.g. `cavio_baseline`, `cross_attn_d512_*`, dropout, `imu_only`, alpha sweeps via CLI). |
| `scripts/schedule_report.sh` | Batch script aligned with the course report experiment table (optional). |

### Plotting and evaluation outputs

| Path | What it is |
|------|------------|
| `src/utils/plotting/plot_train_losses.py` | Train/val loss and component-loss figures from `metrics.csv` (written at end of fit). |
| `src/utils/plotting/kitti_traj_plotting.py` | KITTI trajectory and speed plots during latent eval. |
| `src/testers/latent_kitti_eval_runner.py` | Encoder wrapper + windowed inference + metrics for test-time evaluation. |
| `src/testers/latent_kitti_eval_harness.py` | Tester object used by the Lightning module: calls the runner, writes `error_metrics.json`, optional pose dumps, plot orchestration. |

### Other refactors worth skimming (supporting code, not the core architecture)

| Path | What it is |
|------|------------|
| `src/metrics/kitti_metrics.py` | KITTI `t_rel` / `r_rel` / RMSE (logic parallel to VIFT `kitti_eval`, factored here). |
| `src/utils/lit_hydra.py` | Shared Hydra instantiation for `train.py` and `eval.py`. |
| `src/utils/safe_globals.py` | PyTorch 2 safe globals for checkpoint unpickling. |
| `src/data/latent_caching.py` | One-shot latent caching for train/val splits. |

## Project Structure

- `src/train.py`: training entrypoint (optionally runs test after fit)
- `src/eval.py`: evaluation-only entrypoint
- `configs/`: Hydra configuration groups for data, model, logger, trainer, paths
- `src/models/`: Lightning modules and model components
- `src/testers/`: KITTI latent evaluation harness/runner
- `src/utils/plotting/`: plotting utilities for training-loss and trajectory plots

## Setup

From the `CAVIO` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/setup.sh
```

## Configuration

The main training config is `configs/train.yaml`, which composes:

- `data: latent_kitti_vio`
- `model: cavio`
- `logger: many_loggers`
- `trainer: default` (use `trainer=gpu` on the CLI for GPU; see `configs/trainer/gpu.yaml`)

Evaluation config is `configs/eval.yaml`.

## Train

Run training for a specific experiment:

```bash
python src/train.py experiment=cavio_baseline trainer=gpu
```

## Evaluate

Evaluate a checkpoint for a specific experiment:

```bash
python src/eval.py experiment=cross_attn_d512_ff1024 trainer=gpu ckpt_path=/path/to/checkpoint.ckpt
```

## Outputs

Hydra writes each run to a unique output directory (see `configs/hydra/default.yaml` and `configs/paths/default.yaml`).

Artifacts include:

- model checkpoints
- csv logs
- tensorboard logs
- `error_metrics.json`: final test metrics
- `plots/` directory containing:
  - `loss_plot.png`: train/val loss plot
  - `component_loss_plot.png`: train/val component loss plot (rotation and translation)
  - KITTI trajectory plots: top-down path and vertical trajectory plot

## Notes

- Training/validation component loss plots are generated automatically at fit end.
- KITTI test metrics and trajectory plots are generated during test epoch end.
- If you modify logger layout, keep CSV logging enabled so loss plotting can read `metrics.csv`.
