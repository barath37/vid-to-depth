from typing import Dict, Tuple
import cv2
import numpy as np

from src.common.types import PointCloudModel


class FastOrthoDSMGenerator:
    """Tier 1 Fast-Path Mapping (§4.1 & §10.4 of Spec).
    Generates 2D Orthomosaic (RGB uint8) and Digital Surface Model (DSM float32 height raster)
    directly from the point cloud in seconds, delivering immediate situational awareness.
    """

    def __init__(self, resolution_m: float = 0.20):
        self.resolution_m = resolution_m

    def generate(
        self,
        point_cloud: PointCloudModel
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Projects points into nadir grid cells.
        Returns: (orthomosaic (H, W, 3) uint8, dsm (H, W) float32, bounds_dict)
        """
        pts = point_cloud.points
        colors = point_cloud.colors

        if len(pts) == 0:
            return np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10), dtype=np.float32), {}

        min_x, max_x = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        min_y, max_y = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))

        # Add small buffer
        min_x -= self.resolution_m
        max_x += self.resolution_m
        min_y -= self.resolution_m
        max_y += self.resolution_m

        grid_w = max(1, int(np.ceil((max_x - min_x) / self.resolution_m)))
        grid_h = max(1, int(np.ceil((max_y - min_y) / self.resolution_m)))

        # Coordinate to grid cell mapping (Y is inverted for standard image coordinates)
        col_indices = np.clip(((pts[:, 0] - min_x) / self.resolution_m).astype(int), 0, grid_w - 1)
        row_indices = np.clip(((max_y - pts[:, 1]) / self.resolution_m).astype(int), 0, grid_h - 1)

        dsm = np.full((grid_h, grid_w), -9999.0, dtype=np.float32)
        ortho = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        # Sort points by Z ascending so higher points overwrite lower ones
        sort_order = np.argsort(pts[:, 2])
        for idx in sort_order:
            r = row_indices[idx]
            c = col_indices[idx]
            z = pts[idx, 2]
            if z > dsm[r, c]:
                dsm[r, c] = z
                ortho[r, c] = colors[idx]

        # Infill small no-data gaps in DSM via nearest valid neighbor interpolation
        nodata_mask = (dsm < -9000.0).astype(np.uint8)
        if np.any(nodata_mask) and not np.all(nodata_mask):
            min_valid_z = np.min(dsm[dsm > -9000.0])
            max_valid_z = float(np.max(pts[:, 2]))
            dsm_clean = np.where(dsm > -9000.0, dsm, min_valid_z)
            dsm_inpaint = cv2.inpaint(dsm_clean, nodata_mask, 3, cv2.INPAINT_TELEA)
            dsm = np.clip(dsm_inpaint, min_valid_z, max_valid_z)

        bounds = {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "resolution_m": self.resolution_m,
            "grid_w": grid_w,
            "grid_h": grid_h
        }

        return ortho, dsm, bounds
