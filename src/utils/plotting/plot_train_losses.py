#!/usr/bin/env python3
"""
Plot training and validation loss curves from the Lightning CSVLogger metrics.csv.

Produces two PNGs:
  - loss_plot.png: combined train/val loss
  - component_loss_plot.png: translation and weighted rotation component curves (train/val)

CAVIO-specific; invoked from ``WeightedVIOLitModule.on_fit_end``. Not in upstream VIFT
(https://github.com/ybkurt/vift).
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

import yaml  # type: ignore


def load_angle_weight(run_root: Path) -> float:
    """Return α for L = L_translation + α·L_rotation. Parse from Hydra config."""
    cfg_path = run_root / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        return -1
    data = yaml.safe_load(cfg_path.read_text()) or {}
    aw = data.get("model", {}).get("criterion", {}).get("angle_weight")
    return float(aw) if aw is not None else -1


def infer_run_root_from_metrics(metrics_path: Path) -> Path:
    """Infer run root from .../<run>/csv/version_x/metrics.csv."""
    return metrics_path.resolve().parent.parent.parent


def plot_train_losses(metrics_path: Path, output_dir: Path | None = None) -> tuple[Path, Path | None]:
    """Generate train/val loss plots from one metrics.csv and return output paths."""
    metrics_path = metrics_path.resolve()
    if not metrics_path.is_file():
        raise FileNotFoundError(f"{metrics_path} not found.")

    run_root = infer_run_root_from_metrics(metrics_path)
    out_dir = output_dir.resolve() if output_dir else run_root
    out_dir.mkdir(parents=True, exist_ok=True)

    train_epochs, train_vals = [], []
    val_epochs, val_vals = [], []
    train_rotation_epochs, train_rotation_vals = [], []
    train_translation_epochs, train_translation_vals = [], []
    val_rotation_epochs, val_rotation_vals = [], []
    val_translation_epochs, val_translation_vals = [], []

    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row.get("epoch", 0))

            if row.get("train/loss_epoch"):
                train_epochs.append(epoch)
                train_vals.append(float(row["train/loss_epoch"]))
            if row.get("val/loss"):
                val_epochs.append(epoch)
                val_vals.append(float(row["val/loss"]))
            train_rotation = row.get("train/L_rot_epoch")
            if train_rotation:
                train_rotation_epochs.append(epoch)
                train_rotation_vals.append(float(train_rotation))
            train_translation = row.get("train/L_trans_epoch")
            if train_translation:
                train_translation_epochs.append(epoch)
                train_translation_vals.append(float(train_translation))
            val_rotation = row.get("val/L_rot")
            if val_rotation:
                val_rotation_epochs.append(epoch)
                val_rotation_vals.append(float(val_rotation))
            val_translation = row.get("val/L_trans")
            if val_translation:
                val_translation_epochs.append(epoch)
                val_translation_vals.append(float(val_translation))

    alpha = load_angle_weight(run_root)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(train_epochs, train_vals, label="Train Loss", linewidth=1.5)
    ax.plot(val_epochs, val_vals, label="Val Loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    loss_path = out_dir / "loss_plot.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    comp_path = None

    if train_rotation_epochs or train_translation_epochs:
        train_rotation_scaled = [alpha * v for v in train_rotation_vals]
        val_rotation_scaled = [alpha * v for v in val_rotation_vals] if val_rotation_vals else []

        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        ax.plot(
            train_translation_epochs,
            train_translation_vals,
            label="Train L_trans",
            linewidth=1.8,
            color="C0",
        )
        ax.plot(
            train_rotation_epochs,
            train_rotation_scaled,
            label=f"Train α·L_rot",
            linewidth=1.8,
            color="C1",
        )
        if val_translation_vals:
            ax.plot(
                val_translation_epochs,
                val_translation_vals,
                label="Val L_trans",
                linewidth=1.8,
                color="C0",
                linestyle="--",
            )
        if val_rotation_vals:
            ax.plot(
                val_rotation_epochs,
                val_rotation_scaled,
                label="Val α·L_rot",
                linewidth=1.8,
                color="C1",
                linestyle="--",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss Components: L_trans and α·L_rot (α={alpha:g})")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.margins(y=0.08)

        comp_path = out_dir / "component_loss_plot.png"
        plt.savefig(comp_path, dpi=160, bbox_inches="tight")
        plt.close()

    return loss_path, comp_path
