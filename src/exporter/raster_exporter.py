import os
from typing import Dict, Optional
import numpy as np
import rasterio
from rasterio.transform import from_origin


def export_geotiff_raster(
    raster_data: np.ndarray,
    bounds: Dict[str, float],
    output_path: str,
    epsg_code: int = 32643,
    nodata: Optional[float] = None
) -> str:
    """Exports Orthomosaic or DSM array to georeferenced GeoTIFF (§11 of Spec)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    min_x = bounds["min_x"]
    max_y = bounds["max_y"]
    res = bounds.get("resolution_m", 0.20)

    # Affine geotransform: top-left corner (min_x, max_y)
    transform = from_origin(min_x, max_y, res, res)

    if raster_data.ndim == 2:
        # Single band (e.g. DSM float32)
        count = 1
        dtype = raster_data.dtype
        data_to_write = raster_data[np.newaxis, :, :]
    else:
        # Multi-band (e.g. RGB orthomosaic HxWx3 -> 3xHxW)
        count = raster_data.shape[2]
        dtype = raster_data.dtype
        data_to_write = np.transpose(raster_data, (2, 0, 1))

    height, width = data_to_write.shape[1], data_to_write.shape[2]

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=f"EPSG:{epsg_code}",
        transform=transform,
        nodata=nodata,
        compress="lzw"
    ) as dst:
        dst.write(data_to_write)

    return output_path
