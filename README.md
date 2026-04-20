# CAVIO

Cross-Attention Visual-Inertial Odometry (CAVIO): a PyTorch Lightning + Hydra codebase for training and evaluating latent-space VIO models.

## Acknowledgement

This repository is heavily influenced by the VIFT repository design and workflow.  
Many project patterns (training/eval structure, configuration style, and parts of the VIO pipeline) follow that prior codebase and are adapted here for CAVIO experiments.

VIFT repository: [https://github.com/ybkurt/vift](https://github.com/ybkurt/vift)

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

Run training (and test afterwards if `test: True`):

```bash
python src/train.py trainer=gpu
```

Common overrides:

```bash
python src/train.py trainer=gpu seed=42
python src/train.py trainer=gpu train=False test=True ckpt_path=/path/to/checkpoint.ckpt
```

## Evaluate

Evaluate a checkpoint:

```bash
python src/eval.py trainer=gpu ckpt_path=/path/to/checkpoint.ckpt
```

## Outputs

Hydra writes each run to a unique output directory (see `configs/hydra/default.yaml` and `configs/paths/default.yaml`).

Typical artifacts include:

- model checkpoints
- `csv/.../metrics.csv`
- `error_metrics.json`
- `plots/` directory containing:
  - `loss_plot.png`
  - `component_loss_plot.png`
  - KITTI trajectory plots

## Notes

- Training/validation component loss plots are generated automatically at fit end.
- KITTI test metrics and trajectory plots are generated during test epoch end.
- If you modify logger layout, keep CSV logging enabled so loss plotting can read `metrics.csv`.
# Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

# Install the requirements

```bash
pip install -r requirements.txt
```

# Run the setup script

```bash
./scripts/setup.sh
```
