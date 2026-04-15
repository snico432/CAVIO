"""Matplotlib plots for KITTI eval: top-down X–Z path and speed heatmap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FRAME_RATE_HZ = 10.0  # KITTI odometry sequence rate


def extract_xyz(pose_list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-frame position from each 4×4 pose: one (x, y, z) sample per frame."""
    x = np.asarray([p[0, 3] for p in pose_list], dtype=np.float64)
    y = np.asarray([p[1, 3] for p in pose_list], dtype=np.float64)
    z = np.asarray([p[2, 3] for p in pose_list], dtype=np.float64)
    return x, y, z


def _poses_to_xz(poses_mat) -> tuple[np.ndarray, np.ndarray]:
    x, _, z = extract_xyz(poses_mat)
    return x, z


def plot_top_down_path_on_ax(
    ax,
    x_gt: np.ndarray,
    z_gt: np.ndarray,
    x_est: np.ndarray,
    z_est: np.ndarray,
    *,
    method_label: str = "Ours",
    title: str | None = None,
    fontsize: float = 10,
    legend_fontsize: float = 8,
) -> None:
    """Draw X–Z trajectory (ground truth vs estimate): equal aspect, grid."""
    style_gt = "r-"
    style_est = "b-"
    ax.plot(x_gt, z_gt, style_gt, label="Ground Truth", linewidth=1.2)
    ax.plot(x_est, z_est, style_est, label=method_label, linewidth=1.2)
    ax.plot(0, 0, "ko", label="Start Point", markersize=4)
    ax.set_xlabel("x (m)", fontsize=fontsize)
    ax.set_ylabel("z (m)", fontsize=fontsize)
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=legend_fontsize)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def plot_top_down_speed_on_ax(
    ax,
    x_est: np.ndarray,
    z_est: np.ndarray,
    speed,
    *,
    title: str | None = "speed heatmap",
    fontsize: float = 10,
) -> None:
    """Scatter estimated X–Z colored by speed; equal aspect; colorbar on the figure for this axes."""
    cout = np.asarray(speed)
    mappable = ax.scatter(x_est, z_est, marker="o", c=cout)
    ax.set_xlabel("x (m)", fontsize=fontsize)
    ax.set_ylabel("z (m)", fontsize=fontsize)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    max_speed = float(np.max(cout))
    min_speed = float(np.min(cout))
    ticks = np.floor(np.linspace(min_speed, max_speed, num=5))
    fig = ax.get_figure()
    cbar = fig.colorbar(mappable, ax=ax, ticks=ticks)
    cbar.ax.set_yticklabels([str(i) + "m/s" for i in ticks])


def plot_top_down_view(
    seq: str,
    poses_gt_mat,
    poses_est_mat,
    plot_path_dir: Path | str,
    *,
    method_label: str = "Ours",
    figure_title: str = "Top-down path",
) -> None:
    """Save a single-sequence top-down X–Z path figure ({seq}_path_2d.png)."""
    plot_dir = Path(plot_path_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    x_gt, z_gt = _poses_to_xz(poses_gt_mat)
    x_est, z_est = _poses_to_xz(poses_est_mat)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    plot_top_down_path_on_ax(
        ax,
        x_gt,
        z_gt,
        x_est,
        z_est,
        method_label=method_label,
        title=figure_title,
        fontsize=10,
        legend_fontsize=10,
    )
    plt.savefig(plot_dir / f"{seq}_path_2d.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()


def plot_top_down_speed(
    seq: str,
    poses_est_mat,
    speed,
    plot_path_dir: Path | str,
) -> None:
    """Save a single-sequence speed heatmap on the estimated path ({seq}_speed.png)."""
    plot_dir = Path(plot_path_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    x_est, z_est = _poses_to_xz(poses_est_mat)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    plot_top_down_speed_on_ax(ax, x_est, z_est, speed, title="speed heatmap", fontsize=10)
    plt.savefig(plot_dir / f"{seq}_speed.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()


def plotPath_2D(seq, poses_gt_mat, poses_est_mat, plot_path_dir, speed, *, method_label: str = "Ours"):
    """Save per-sequence top-down path and speed PNGs."""
    plot_top_down_view(seq, poses_gt_mat, poses_est_mat, plot_path_dir, method_label=method_label)
    plot_top_down_speed(seq, poses_est_mat, speed, plot_path_dir)


def plotAllPaths(
    val_seq: list[str],
    est: list[dict],
    save_dir: Path | str,
    *,
    method_label: str = "Ours",
    plot_speed: bool = False,
) -> None:
    """Multi-column grid from ``LatentKittiEvalRunner.run_all_sequences`` ``est`` output.

    Rows: top-down X–Z path; if ``plot_speed``, speed heatmap on estimated path; bottom Y vs time.
    """
    assert len(val_seq) == len(est), "val_seq and est must have the same length"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fontsize_ = 10
    style_gt = "r-"
    style_est = "b-"

    nrows = 3 if plot_speed else 2
    fig_h = 12 if plot_speed else 8
    fig, axes = plt.subplots(nrows, 3, figsize=(12, fig_h))

    for col, seq in enumerate(val_seq):
        entry = est[col]
        x_gt, y_gt, z_gt = extract_xyz(entry["pose_gt_global"])
        x_est, y_est, z_est = extract_xyz(entry["pose_est_global"])
        n = len(x_gt)
        t = np.arange(n) / FRAME_RATE_HZ

        plot_top_down_path_on_ax(
            axes[0, col],
            x_gt,
            z_gt,
            x_est,
            z_est,
            method_label=method_label,
            title=f"Sequence {seq}",
            fontsize=fontsize_,
            legend_fontsize=8,
        )

        if plot_speed:
            if "speed" not in entry:
                raise ValueError(f'est[{col}] must include "speed" when plot_speed=True')
            plot_top_down_speed_on_ax(
                axes[1, col],
                x_est,
                z_est,
                entry["speed"],
                title=f"Speed {seq}",
                fontsize=fontsize_,
            )
            y_row = 2
        else:
            y_row = 1

        ax_y = axes[y_row, col]
        ax_y.plot(t, y_gt, style_gt, label="Ground Truth", linewidth=1.2)
        ax_y.plot(t, y_est, style_est, label=method_label, linewidth=1.2)
        ax_y.set_xlabel("Time (s)", fontsize=fontsize_)
        ax_y.set_ylabel("y (m)", fontsize=fontsize_)
        ax_y.set_title(f"Sequence {seq}")
        ax_y.legend(loc="upper right", fontsize=8)
        ax_y.grid(True, alpha=0.3)

    out_path = save_dir / "all_paths.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Saved: {out_path.name}")
