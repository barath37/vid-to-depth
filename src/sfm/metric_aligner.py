from typing import List, Optional, Tuple
import numpy as np

from src.common.coordinates import umeyama_sim3, align_gravity_and_ground_plane
from src.common.types import CameraPose


class MetricTrajectoryAligner:
    """Scales and georeferences SfM camera poses to the EKF trajectory (§2.5 & §4 of Spec).
    Resolves the single-pass flight observability trap using IMU gravity and ground plane normals.
    """

    def __init__(self, ransac_threshold_m: float = 1.0):
        self.ransac_threshold_m = ransac_threshold_m

    def align_to_ekf(
        self,
        sfm_poses: List[CameraPose],
        ekf_positions: np.ndarray,
        gravity_vector: Optional[np.ndarray] = None,
        ground_normal: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray, np.ndarray, List[CameraPose]]:
        """Finds Sim(3) transform (scale s, rotation R, translation t) mapping SfM poses to EKF positions.
        Returns: (scale, R, t, aligned_poses)
        """
        assert len(sfm_poses) == len(ekf_positions)
        sfm_centers = np.array([p.center for p in sfm_poses])

        # Step 1: Base Sim(3) alignment via Umeyama
        scale, R_sim, t_sim = umeyama_sim3(sfm_centers, ekf_positions, with_scale=True)

        # Step 2: Observability Trap Resolution (§4.3 of Spec)
        # Check trajectory curvature / collinearity via SVD of centered EKF positions
        centered_ekf = ekf_positions - np.mean(ekf_positions, axis=0)
        _, s_vals, _ = np.linalg.svd(centered_ekf)
        collinearity_ratio = float(s_vals[1] / max(s_vals[0], 1e-6)) if len(s_vals) > 1 else 0.0
        is_collinear = (collinearity_ratio < 0.15)

        # Apply roll-lock only on collinear paths; bypass on orbital/curved flights
        if is_collinear and gravity_vector is not None and ground_normal is not None:
            R_roll_lock = align_gravity_and_ground_plane(gravity_vector, ground_normal)
            # Combine rotations: R_final = R_roll_lock @ R_sim
            R_sim = R_roll_lock @ R_sim
            # Recompute translation
            mu_sfm = np.mean(sfm_centers, axis=0)
            mu_ekf = np.mean(ekf_positions, axis=0)
            t_sim = mu_ekf - scale * (R_sim @ mu_sfm)

        # Step 3: Transform all camera poses into metrically-scaled ENU world coordinates
        aligned_poses: List[CameraPose] = []
        for p in sfm_poses:
            # New camera center: c_new = scale * R_sim @ c_old + t_sim
            c_new = scale * (R_sim @ p.center) + t_sim
            # New rotation: R_cam_new = p.R @ R_sim.T
            R_cam_new = p.R @ R_sim.T
            # New translation: t_new = -R_cam_new @ c_new
            t_new = -R_cam_new @ c_new

            aligned_poses.append(CameraPose(
                frame_idx=p.frame_idx,
                R=R_cam_new,
                t=t_new,
                timestamp=p.timestamp
            ))

        return scale, R_sim, t_sim, aligned_poses
