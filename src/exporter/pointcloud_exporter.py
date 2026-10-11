import os
from typing import Optional
import laspy
import numpy as np
from pyproj import CRS

from src.common.types import PointCloudModel


def export_pointcloud_las(
    cloud: PointCloudModel,
    output_path: str,
    epsg_code: int = 32643,
    scale: float = 0.001
) -> str:
    """Exports point cloud to LAS 1.4 format with RGB colors and epistemic confidence extra dimension (§11 of Spec)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pts = cloud.points
    N = len(pts)

    if N == 0:
        pts = np.zeros((1, 3), dtype=np.float32)
        N = 1

    # LAS 1.4 Header with Point Format 3 (includes RGB) or Point Format 7
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([scale, scale, scale])
    header.offsets = np.min(pts, axis=0)

    # Embed CRS
    try:
        crs = CRS.from_epsg(epsg_code)
        header.add_crs(crs)
    except Exception:
        pass

    # Add extra bytes dimension for epistemic confidence
    extra_bytes = laspy.ExtraBytesParams(name="confidence", type=np.float32)
    header.add_extra_dim(extra_bytes)

    las = laspy.LasData(header)
    las.x = pts[:, 0]
    las.y = pts[:, 1]
    las.z = pts[:, 2]

    if cloud.colors is not None and len(cloud.colors) == N:
        # LAS expects 16-bit colors (0-65535), upscale from uint8
        las.red = (cloud.colors[:, 0].astype(np.uint16) * 256)
        las.green = (cloud.colors[:, 1].astype(np.uint16) * 256)
        las.blue = (cloud.colors[:, 2].astype(np.uint16) * 256)

    if cloud.confidence is not None and len(cloud.confidence) == N:
        las.confidence = cloud.confidence.astype(np.float32)
    else:
        las.confidence = np.ones(N, dtype=np.float32)

    las.write(output_path)
    return output_path
