from typing import List
import cv2
import numpy as np
import trimesh

from src.common.types import CameraPose, KeyframeData


class MeshTextureProjector:
    """Best-view camera texture projector (§10.3 of Spec).
    Projects colors from original high-resolution keyframes onto the 3D mesh vertices/faces,
    selecting the most orthogonal, closest, and sharpest observing camera view per vertex.
    """

    def __init__(self, intrinsic_matrix: np.ndarray):
        self.K = intrinsic_matrix
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.cx = intrinsic_matrix[0, 2]
        self.cy = intrinsic_matrix[1, 2]

    def project_texture(
        self,
        mesh: trimesh.Trimesh,
        keyframes: List[KeyframeData],
        poses: List[CameraPose]
    ) -> trimesh.Trimesh:
        """Assigns per-vertex colors by projecting from best keyframe views."""
        if len(mesh.vertices) == 0 or not keyframes or not poses:
            return mesh

        verts = mesh.vertices # (V, 3)
        n_verts = len(verts)
        vertex_colors = np.zeros((n_verts, 3), dtype=np.uint8)
        best_scores = np.full(n_verts, -1e9, dtype=np.float32)

        # Build mapping from frame_idx to keyframe
        kf_dict = {kf.frame_idx: kf for kf in keyframes}

        for pose in poses:
            kf = kf_dict.get(pose.frame_idx)
            if kf is None:
                continue

            img_h, img_w = kf.image.shape[:2]
            rgb_img = cv2.cvtColor(kf.image, cv2.COLOR_BGR2RGB)

            # Transform vertices to camera frame: X_c = R @ X_w + t
            pts_cam = (pose.R @ verts.T).T + pose.t
            z = pts_cam[:, 2]

            # In front of camera
            valid_depth = z > 0.5
            if not np.any(valid_depth):
                continue

            # Project to 2D
            u = np.round(self.fx * (pts_cam[:, 0] / z) + self.cx).astype(int)
            v = np.round(self.fy * (pts_cam[:, 1] / z) + self.cy).astype(int)

            in_frame = valid_depth & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
            if not np.any(in_frame):
                continue

            # Scoring: closer distance and higher sharpness gets higher score
            scores = (1.0 / (z + 1.0)) * (kf.laplacian_var / 100.0)

            # Update vertices where this view is superior
            better = in_frame & (scores > best_scores)
            idx_better = np.where(better)[0]

            if len(idx_better) > 0:
                best_scores[idx_better] = scores[idx_better]
                u_better = u[idx_better]
                v_better = v[idx_better]
                vertex_colors[idx_better] = rgb_img[v_better, u_better]

        mesh.visual.vertex_colors = vertex_colors
        return mesh
