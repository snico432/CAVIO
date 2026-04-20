# CAVIO

Cross-Attention Visual-Inertial Odometry (CAVIO): a PyTorch Lightning + Hydra codebase for training and evaluating latent-space VIO models.

## Acknowledgement

This repository is heavily influenced by the VIFT repository design and workflow.  
Many project patterns (training/eval structure, configuration style, and parts of the VIO pipeline) follow that prior codebase and are adapted here for CAVIO experiments.

VIFT repository: [https://github.com/ybkurt/vift](https://github.com/ybkurt/vift)

## Where to review the project work (CAVIO-specific)

Aside from the shared VIFT-style training shell, data layout, and pretrained encoder, the highest-impact paths for grading or code review are:

- **`src/models/components/cavio.py`** — `CAVIOPoseTransformer` (cross-attention fusion + causal self-attention); the main architecture.
- **`src/models/components/`** — Ablations: `cavio_gated.py`, `cavio_visual_residual.py`, `imu_only.py`.
- **`src/losses/weighted_loss.py`** — RPMG pose losses and configurable `CustomWeightedPoseLoss` (used in α and axis-weighting experiments).
- **`configs/model/`** and **`configs/experiment/`** — Hydra defaults and experiment presets; optional batch reruns via **`scripts/schedule_report.sh`**.
- **`src/utils/plotting/`** — Training curves (`plot_train_losses.py`) and KITTI trajectory / speed figures (`kitti_traj_plotting.py`).
- **`src/testers/latent_kitti_eval_runner.py`** and **`latent_kitti_eval_harness.py`** — Latent KITTI eval (encoder wrapper, metrics, plots from the Lightning test step).

For pipeline glue (KITTI metric definitions, Hydra wiring, latent caching, checkpoint safe globals), see `src/metrics/kitti_metrics.py`, `src/utils/lit_hydra.py`, `src/utils/safe_globals.py`, and `src/data/latent_caching.py`.

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

- If you modify logger layout, keep CSV logging enabled so loss plotting can read `metrics.csv`.
