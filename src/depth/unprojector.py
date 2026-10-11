from typing import Optional
import cv2
import numpy as np

from src.common.types import CameraPose, ConfidenceTag, PointCloudModel


class PseudoLiDARUnprojector:
    """Unprojects per-pixel metric depth maps into a unified 3D Euclidean point cloud (§3.5 of Spec).
    Filters sky pixels and dynamic objects, and computes view-oriented normal vectors.
    """

    def __init__(self, intrinsic_matrix: np.ndarray, max_depth_m: float = 300.0):
        self.K = intrinsic_matrix
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.cx = intrinsic_matrix[0, 2]
        self.cy = intrinsic_matrix[1, 2]
        self.max_depth_m = max_depth_m

    def unproject_frame(
        self,
        depth_metric: np.ndarray,
        color_bgr: np.ndarray,
        confidence_map: np.ndarray,
        pose: CameraPose,
        mask: Optional[np.ndarray] = None
    ) -> PointCloudModel:
        """Unprojects a single frame's depth map to world Euclidean coordinates.
        depth_metric: (H, W) float32 in meters.
        color_bgr: (H, W, 3) uint8.
        confidence_map: (H, W) float32 in [0, 1].
        pose: CameraPose with R, t (world to camera).
        Returns: PointCloudModel.
        """
        H, W = depth_metric.shape

        # Valid pixel mask: finite, within depth bounds, and not masked by dynamic object filter
        valid = (depth_metric > 0.5) & (depth_metric < self.max_depth_m) & np.isfinite(depth_metric)
        if mask is not None:
            valid = valid & (mask > 0)

        v_indices, u_indices = np.where(valid)
        if len(u_indices) == 0:
            return PointCloudModel(
                points=np.zeros((0, 3), dtype=np.float32),
                colors=np.zeros((0, 3), dtype=np.uint8),
                confidence=np.zeros(0, dtype=np.float32),
                tags=[]
            )

        z_vals = depth_metric[v_indices, u_indices]
        x_cam = (u_indices - self.cx) * z_vals / self.fx
        y_cam = (v_indices - self.cy) * z_vals / self.fy
        pts_cam = np.column_stack([x_cam, y_cam, z_vals])

        # World coordinates: X_world = pose.R.T @ (X_cam - pose.t)
        # c = -pose.R.T @ pose.t
        pts_world = (pose.R.T @ (pts_cam.T - pose.t.reshape(3, 1))).T.astype(np.float32)

        # RGB Colors
        rgb_img = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        colors = rgb_img[v_indices, u_indices].astype(np.uint8)

        # Confidence & Tags
        conf_vals = confidence_map[v_indices, u_indices]
        tags = []
        for c in conf_vals:
            if c >= 0.70:
                tags.append(ConfidenceTag.GREEN)
            elif c >= 0.40:
                tags.append(ConfidenceTag.YELLOW)
            else:
                tags.append(ConfidenceTag.RED)

        # View-oriented normals pointing toward camera optical center
        cam_center = pose.center
        ray_vecs = cam_center - pts_world
        norms = np.linalg.norm(ray_vecs, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1e-6
        normals = (ray_vecs / norms).astype(np.float32)

        return PointCloudModel(
            points=pts_world,
            colors=colors,
            confidence=conf_vals,
            tags=tags,
            normals=normals
        )
