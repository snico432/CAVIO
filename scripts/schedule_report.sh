#!/usr/bin/env bash
#
# Train CAVIO runs that correspond to the CIS4910 final report experiments.
#
# Usage (from CAVIO repository root):
#   bash scripts/schedule_report.sh
#
# Expect long wall-clock time (200 epochs per run by default). Comment out
# sections, run subsets, or submit individual lines to a cluster. Only
# ``trainer=gpu`` is passed explicitly; ``logger=many_loggers`` comes from
# ``configs/train.yaml`` defaults.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# --- Cross-attention baseline: d_model=768, d_ff=128, alpha=40 (report §4.1, Table baseline_config) ---
python src/train.py experiment=cavio_baseline trainer=gpu

# --- Rotational loss weight alpha sweep: d_model=512, d_ff=1024 (report Table alpha_sweep) ---
for alpha in 10 20 23 25 27 30 35 40; do
  python src/train.py experiment=cross_attn_d512_ff1024 trainer=gpu \
    "model.criterion.angle_weight=${alpha}"
done

# --- FFN width sweep at alpha=25 (report Table ffn_sweep).
# Same alpha as ``configs/model/cavio.yaml`` (angle_weight: 25); no CLI override needed.
# d_ff=1024 at alpha=25 is already produced in the alpha loop above; train 1536 and 2048 only. ---
python src/train.py experiment=cross_attn_d512_ff1536 trainer=gpu
python src/train.py experiment=cross_attn_d512_ff2048 trainer=gpu

# --- FFN dropout sweep (report Table dropout_sweep); alpha=25 is still the model default. ---
python src/train.py experiment=cavio_d512_ff1024_do005 trainer=gpu
python src/train.py experiment=cavio_d512_ff1024_do010 trainer=gpu

# --- IMU-only ablation (report §4.1.5) ---
python src/train.py experiment=imu_only trainer=gpu

# --- Optional: other CAVIO presets mentioned briefly in report §5.1 (uncomment to run) ---
# python src/train.py experiment=cavio_dataweighting trainer=gpu
# python src/train.py experiment=cavio_visual_residual trainer=gpu
# python src/train.py experiment=gated_cavio trainer=gpu
