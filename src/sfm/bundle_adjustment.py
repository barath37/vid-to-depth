from typing import List, Tuple
import cv2
import numpy as np
from scipy.optimize import least_squares

from src.common.types import CameraPose


class GlobalBundleAdjuster:
    """Global Bundle Adjustment (GBA) pass (§2.4 of Spec).
    Optimizes all camera poses and 3D sparse points simultaneously with Huber robust loss.
    Executed strictly BEFORE any sparse points are injected into depth completion.
    """

    def __init__(self, intrinsic_matrix: np.ndarray, loss_delta: float = 1.5):
        self.K = intrinsic_matrix
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.cx = intrinsic_matrix[0, 2]
        self.cy = intrinsic_matrix[1, 2]
        self.loss_delta = loss_delta

    def _project(self, points3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Projects 3D points into camera view given Rodrigues rvec and tvec."""
        R, _ = cv2.Rodrigues(rvec)
        pts_c = (R @ points3d.T).T + tvec
        z = pts_c[:, 2]
        z[np.abs(z) < 1e-4] = 1e-4

        u = self.fx * (pts_c[:, 0] / z) + self.cx
        v = self.fy * (pts_c[:, 1] / z) + self.cy
        return np.column_stack([u, v])

    def optimize(
        self,
        poses: List[CameraPose],
        points3d: np.ndarray,
        observations: List[np.ndarray], # observations[i] is (N, 2) 2D points seen by pose i
        max_nfev: int = 50
    ) -> Tuple[List[CameraPose], np.ndarray]:
        """Refines camera poses and 3D point coordinates."""
        n_cameras = len(poses)
        n_points = len(points3d)

        if n_cameras < 2 or n_points == 0:
            return poses, points3d

        # Param vector: [rvec_1..rvec_C, tvec_1..tvec_C, pts_1..pts_P]
        # Camera 0 is fixed as gauge constraint to prevent datum drift
        cam_params = []
        for i, pose in enumerate(poses):
            if i == 0:
                continue
            rvec, _ = cv2.Rodrigues(pose.R)
            cam_params.extend(rvec.ravel())
            cam_params.extend(pose.t.ravel())

        x0 = np.hstack([np.array(cam_params, dtype=np.float64), points3d.ravel()])

        def residuals(x: np.ndarray) -> np.ndarray:
            # Unpack parameters
            n_moving_cams = n_cameras - 1
            cam_part = x[:n_moving_cams * 6]
            pts_part = x[n_moving_cams * 6:].reshape(-1, 3)

            res = []
            for cam_idx in range(n_cameras):
                if cam_idx == 0:
                    rvec, _ = cv2.Rodrigues(poses[0].R)
                    tvec = poses[0].t
                else:
                    offset = (cam_idx - 1) * 6
                    rvec = cam_part[offset:offset + 3]
                    tvec = cam_part[offset + 3:offset + 6]

                obs_2d = observations[cam_idx]
                if len(obs_2d) > 0:
                    proj = self._project(pts_part, rvec, tvec)
                    diff = (proj - obs_2d).ravel()
                    res.append(diff)

            if not res:
                return np.zeros(1)
            return np.concatenate(res)

        res_opt = least_squares(
            residuals,
            x0,
            loss="huber",
            f_scale=self.loss_delta,
            max_nfev=max_nfev,
            method="trf"
        )

        # Unpack optimized parameters
        opt_x = res_opt.x
        n_moving = n_cameras - 1
        cam_part = opt_x[:n_moving * 6]
        opt_points = opt_x[n_moving * 6:].reshape(-1, 3)

        optimized_poses = [poses[0]]
        for cam_idx in range(1, n_cameras):
            offset = (cam_idx - 1) * 6
            rvec = cam_part[offset:offset + 3]
            tvec = cam_part[offset + 3:offset + 6]
            R, _ = cv2.Rodrigues(rvec)
            optimized_poses.append(CameraPose(poses[cam_idx].frame_idx, R, tvec, poses[cam_idx].timestamp))

        return optimized_poses, opt_points
