from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.common.types import PointCloudModel


@dataclass
class GaussianSplatModel:
    """Tier 2 Depth-Initialized 3D Gaussian Splatting Visual Asset (§10.5 of Spec).
    Initializes 3D Gaussians (positions, covariance scales, rotation quaternions, opacities, and SH colors)
    directly onto the metric pseudo-LiDAR point cloud coordinates for rapid photoreal rendering.
    """

    positions: np.ndarray        # (N, 3) float32
    scales: np.ndarray           # (N, 3) float32 log-scales
    rotations: np.ndarray        # (N, 4) float32 normalized quaternions [w, x, y, z]
    opacities: np.ndarray        # (N, 1) float32 inverse sigmoid opacities
    colors_sh: np.ndarray        # (N, 3) float32 base RGB or Spherical Harmonics

    @property
    def num_gaussians(self) -> int:
        return len(self.positions)

    @classmethod
    def from_point_cloud(
        cls,
        point_cloud: PointCloudModel,
        default_scale: float = 0.15,
        default_opacity: float = 0.80
    ) -> "GaussianSplatModel":
        """Initializes 3D Gaussians directly from the metrically scaled point cloud."""
        pts = point_cloud.points
        N = len(pts)

        if N == 0:
            return cls(
                positions=np.zeros((0, 3), dtype=np.float32),
                scales=np.zeros((0, 3), dtype=np.float32),
                rotations=np.zeros((0, 4), dtype=np.float32),
                opacities=np.zeros((0, 1), dtype=np.float32),
                colors_sh=np.zeros((0, 3), dtype=np.float32)
            )

        positions = pts.copy().astype(np.float32)

        # Initialize scales around point spacing (log scale)
        log_scale = np.log(max(default_scale, 1e-3))
        scales = np.full((N, 3), log_scale, dtype=np.float32)

        # Unit quaternions [w=1, x=0, y=0, z=0]
        rotations = np.zeros((N, 4), dtype=np.float32)
        rotations[:, 0] = 1.0

        # Opacities
        opacities = np.full((N, 1), default_opacity, dtype=np.float32)

        # Base colors normalized to [-0.5, 0.5] or standard SH 0-th order
        if point_cloud.colors is not None and len(point_cloud.colors) == N:
            colors_sh = (point_cloud.colors / 255.0 - 0.5).astype(np.float32)
        else:
            colors_sh = np.zeros((N, 3), dtype=np.float32)

        return cls(
            positions=positions,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            colors_sh=colors_sh
        )

    def save_ply(self, output_path: str):
        """Exports 3D Gaussians to standard 3DGS PLY format for web/desktop viewers."""
        from plyfile import PlyData, PlyElement

        # Create structured array for plyfile
        dtype_list = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
            ("opacity", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")
        ]
        elements = np.empty(self.num_gaussians, dtype=dtype_list)
        elements["x"] = self.positions[:, 0]
        elements["y"] = self.positions[:, 1]
        elements["z"] = self.positions[:, 2]
        elements["scale_0"] = self.scales[:, 0]
        elements["scale_1"] = self.scales[:, 1]
        elements["scale_2"] = self.scales[:, 2]
        elements["rot_0"] = self.rotations[:, 0]
        elements["rot_1"] = self.rotations[:, 1]
        elements["rot_2"] = self.rotations[:, 2]
        elements["rot_3"] = self.rotations[:, 3]
        elements["opacity"] = self.opacities[:, 0]
        elements["f_dc_0"] = self.colors_sh[:, 0]
        elements["f_dc_1"] = self.colors_sh[:, 1]
        elements["f_dc_2"] = self.colors_sh[:, 2]

        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(output_path)
