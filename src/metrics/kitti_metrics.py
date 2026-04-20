"""KITTI odometry metrics: RMSE, segment relative translation/rotation error, speed.

The numerical pipeline matches VIFT ``kitti_eval`` and ``kitti_err_cal`` in
``src/utils/kitti_eval.py`` (https://github.com/ybkurt/vift): accumulate relative
poses with ``path_accu``, then evaluate 100--800\,m segments via
``computeOverallErr``, with the same
scaling of ``t_rel`` / ``r_rel`` / rotational RMSE as in VIFT.

CAVIO exposes this logic as small functions in ``src/metrics/`` for the latent
eval harness. VIFT additionally ships a class-based ``KITTIMetricsCalculator`` in
``src/metrics/kitti_metrics_calculator.py`` (dict of multi-sequence results); that
wrapper is not ported—callers here pass poses directly.
"""

import numpy as np

from src.utils.kitti_utils import (
    computeOverallErr,
    lastFrameFromSegmentLength,
    path_accu,
    rmse_err_cal,
    rotationError,
    trajectoryDistances,
    translationError,
)


def compute_kitti_odometry_metrics(pose_est, pose_gt):
    """Integrate relative poses, then RMSE plus KITTI-style segment relative errors and speed.

    Equivalent to VIFT ``kitti_eval`` (``src/utils/kitti_eval.py``).

    Args:
        pose_est: Estimated relative poses (same layout as ``pose_gt``).
        pose_gt: Ground-truth relative poses.

    Returns:
        Global pose matrices, scaled t/r errors and RMSE, and per-frame speed from GT trajectory.
    """
    t_rmse, r_rmse = rmse_err_cal(pose_est, pose_gt)

    pose_est_mat = path_accu(pose_est)
    pose_gt_mat = path_accu(pose_gt)

    _, t_rel, r_rel, speed = compute_kitti_segment_errors(pose_est_mat, pose_gt_mat)

    t_rel = t_rel * 100
    r_rel = r_rel / np.pi * 180 * 100
    r_rmse = r_rmse / np.pi * 180

    return pose_est_mat, pose_gt_mat, t_rel, r_rel, t_rmse, r_rmse, speed


def compute_kitti_segment_errors(pose_est_mat, pose_gt_mat):
    """KITTI odometry benchmark: relative rotation/translation error over fixed path lengths.

    Equivalent to VIFT ``kitti_err_cal`` (``src/utils/kitti_eval.py``).

    Returns per-segment error rows, aggregate t/r drift, and trajectory speed from GT.
    """
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    num_lengths = len(lengths)

    err = []
    dist, speed = trajectoryDistances(pose_gt_mat)
    step_size = 10  # 10Hz

    for first_frame in range(0, len(pose_gt_mat), step_size):
        for i in range(num_lengths):
            len_ = lengths[i]
            last_frame = lastFrameFromSegmentLength(dist, first_frame, len_)
            if last_frame == -1 or last_frame >= len(pose_est_mat) or first_frame >= len(pose_est_mat):
                continue

            pose_delta_gt = np.dot(np.linalg.inv(pose_gt_mat[first_frame]), pose_gt_mat[last_frame])
            pose_delta_result = np.dot(np.linalg.inv(pose_est_mat[first_frame]), pose_est_mat[last_frame])

            r_err = rotationError(pose_delta_result, pose_delta_gt)
            t_err = translationError(pose_delta_result, pose_delta_gt)

            err.append([first_frame, r_err / len_, t_err / len_, len_])

    t_rel, r_rel = computeOverallErr(err)
    return err, t_rel, r_rel, np.asarray(speed)
