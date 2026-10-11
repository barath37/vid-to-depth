from typing import Tuple
import cv2
import numpy as np


class RelativePoseEstimator:
    """Estimates relative camera rotation (R), translation direction (t),
    and triangulates 3D points via Epipolar Geometry (§2.2 & §2.3 of Spec).
    """

    def __init__(self, intrinsic_matrix: np.ndarray):
        self.K = intrinsic_matrix

    def estimate_pose(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recovers relative rotation R (3x3) and unit translation t (3,1)
        between two views using the Essential Matrix.
        pts1: (N, 2)
        pts2: (N, 2)
        Returns: (R, t, inlier_indices)
        """
        assert len(pts1) == len(pts2)
        if len(pts1) < 8:
            return np.eye(3), np.zeros((3, 1)), np.array([], dtype=int)

        # Essential Matrix via RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.5
        )

        if E is None or mask is None:
            return np.eye(3), np.zeros((3, 1)), np.array([], dtype=int)

        # Recover relative pose (cheirality check selects positive depth solution)
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        inliers = np.where(mask_pose.ravel() > 0)[0]
        return R, t, inliers

    def triangulate_points(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        R: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """Linear triangulation of matched 2D points into 3D Euclidean coordinates.
        P1 = K [I | 0]
        P2 = K [R | t]
        Returns: (N, 3) float64 3D points in Camera 1 frame.
        """
        assert len(pts1) == len(pts2)
        if len(pts1) == 0:
            return np.zeros((0, 3), dtype=np.float64)

        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t.reshape(3, 1)])

        pts1_h = pts1.T
        pts2_h = pts2.T

        pts4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
        # Convert from homogeneous coordinates (X, Y, Z, W) to (X/W, Y/W, Z/W)
        w = pts4d[3, :]
        w[np.abs(w) < 1e-6] = 1e-6 # Avoid division by zero
        pts3d = (pts4d[:3, :] / w).T

        return pts3d
