# CAVIO

Cross-Attention Visual-Inertial Odometry (CAVIO) is a PyTorch Lightning and Hydra codebase for training latent-space visual-inertial odometry models on KITTI. The project investigates whether replacing VIFT-style feature concatenation with structured cross-attention improves visual/IMU fusion for camera pose estimation.

## Overview

VIFT combines visual features from a pretrained encoder with IMU features, then passes the concatenated representation through a transformer pose model. CAVIO keeps the same general training and latent-data workflow, but changes the fusion mechanism: IMU latents query visual latents through cross-attention, followed by causal self-attention over the fused temporal sequence.

The project includes the main CAVIO transformer, several ablations, configurable pose losses, and a KITTI evaluation harness for trajectory and odometry metrics.

For more background and experimental details, see the CIS4910 [literature review](docs/literature_review.pdf) and [final report](docs/final_report.pdf).

## Implementation Highlights

- Built `CAVIOPoseTransformer`, a cross-attention VIO model where IMU features query visual features before causal temporal self-attention.
- Added ablation models for IMU-only, gated cross-attention, and visual-residual fusion variants.
- Refactored VIFT-derived training, loss, and evaluation code to reduce duplication and improve readability, maintainability, and reuse.
- Organized experiments with Hydra presets for baseline, architecture-size, dropout, loss-weighting, and ablation runs.

## Results Summary

The strongest CAVIO configuration used a 512-dimensional transformer embedding, 1024-dimensional feed-forward layers, 8 attention heads, and a rotation loss weight of 25. In the final report, this configuration improved selected sequence-level KITTI metrics compared with the reproduced VIFT baseline while remaining competitive overall.

The experiments also showed that vertical trajectory estimation remained difficult: top-down motion was captured more reliably than the y-axis component. The IMU-only ablation performed substantially worse, confirming that visual features contributed meaningful signal even when fusion quality was the main bottleneck.

## Project Structure

- `src/models/components/cavio.py`: main cross-attention transformer architecture
- `src/models/components/`: CAVIO ablations and VIFT-compatible components
- `src/models/weighted_vio_module.py`: Lightning module for training and evaluation
- `src/losses/weighted_loss.py`: weighted pose losses and RPMG-based objectives
- `src/metrics/kitti_metrics.py`: KITTI odometry metric utilities
- `src/testers/`: latent KITTI evaluation harness and runner
- `src/utils/plotting/`: training-loss and trajectory plotting utilities
- `configs/`: Hydra configuration groups and experiment presets
- `scripts/`: setup, debug, and batch experiment helpers

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
- `trainer: default`

Use `trainer=gpu` on the CLI for GPU training. Evaluation config is `configs/eval.yaml`.

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

Hydra writes each run to a unique output directory using `configs/hydra/default.yaml` and `configs/paths/default.yaml`.

Artifacts include:

- model checkpoints
- CSV logs
- TensorBoard logs
- `error_metrics.json`: final test metrics
- `plots/`:
  - `loss_plot.png`: train/validation loss plot
  - `component_loss_plot.png`: rotation and translation loss plot
  - `trajectories.png`: KITTI trajectory plots with top-down path and vertical trajectory comparison

## Acknowledgements

This repository is heavily influenced by the VIFT repository design and workflow. Many project patterns, including the training/evaluation structure, configuration style, latent-data flow, and parts of the VIO pipeline, follow that prior codebase and are adapted here for CAVIO experiments.

VIFT repository: <https://github.com/ybkurt/vift>

Development used [Cursor](https://cursor.com) for AI-assisted refactoring, documentation, and tooling. Model design, experiments, and analysis are the author's own.

## Notes

- Keep CSV logging enabled if you modify the logger layout; loss plotting expects `metrics.csv`.
