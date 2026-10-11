from typing import Tuple
import numpy as np
from scipy.optimize import least_squares


def align_relative_depth_to_metric(
    rel_depth: np.ndarray,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    sparse_metric_depths: np.ndarray,
    huber_delta: float = 1.0
) -> Tuple[np.ndarray, float, float]:
    """Fits affine scale (s) and shift (b) to anchor relative AI depth
    to the drift-corrected BA sparse 3D points (§3.2 & §8 of Spec).
    d_metric ≈ s * d_rel + b.
    Returns: (dense_metric_depth_map, s, b)
    """
    assert len(u_coords) == len(v_coords) == len(sparse_metric_depths)
    h, w = rel_depth.shape

    if len(sparse_metric_depths) < 2:
        # Default nominal fallback: scale by 50m
        s_default = 50.0
        b_default = 5.0
        return (s_default * rel_depth + b_default), s_default, b_default

    # Sample relative depths at keypoint coordinates
    u_clamped = np.clip(u_coords, 0, w - 1)
    v_clamped = np.clip(v_coords, 0, h - 1)
    sample_rel = rel_depth[v_clamped, u_clamped]

    # Initial guess via standard linear least squares
    A = np.column_stack([sample_rel, np.ones_like(sample_rel)])
    res_lstsq, _, _, _ = np.linalg.lstsq(A, sparse_metric_depths, rcond=None)
    s0 = float(np.clip(res_lstsq[0], 0.2, 450.0))
    b0 = float(np.clip(res_lstsq[1], -90.0, 450.0))

    # Robust Huber fitting to reject triangulation outliers
    def residuals(params):
        s, b = params
        pred = s * sample_rel + b
        return pred - sparse_metric_depths

    opt = least_squares(
        residuals,
        [s0, b0],
        loss="huber",
        f_scale=huber_delta,
        bounds=([0.1, -100.0], [500.0, 500.0])
    )

    s_fit, b_fit = float(opt.x[0]), float(opt.x[1])
    dense_metric = np.clip(s_fit * rel_depth + b_fit, 0.5, 1000.0).astype(np.float32)

    return dense_metric, s_fit, b_fit
