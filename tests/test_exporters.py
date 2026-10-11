import os
import tempfile
import numpy as np
import pytest
import trimesh

from src.common.types import PointCloudModel
from src.exporter.mesh_exporter import export_mesh_multi_format
from src.exporter.pointcloud_exporter import export_pointcloud_las
from src.exporter.raster_exporter import export_geotiff_raster
from src.exporter.fbx_converter import convert_to_fbx


def test_export_mesh_multi_format():
    # Simple cube mesh
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    local_origin = np.array([769558.0, 1101680.0, 400.0]) # Large UTM-like coordinates

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "test_model")
        outputs = export_mesh_multi_format(
            mesh,
            base_path=base_path,
            local_origin=local_origin,
            epsg_code=32643
        )

        assert os.path.exists(outputs["obj"])
        assert os.path.exists(outputs["ply"])
        assert os.path.exists(outputs["glb"])
        assert os.path.exists(outputs["georef_json"])

        # Check that GLB is readable and valid
        loaded_glb = trimesh.load(outputs["glb"])
        assert loaded_glb is not None


def test_export_pointcloud_las():
    N = 100
    points = np.random.uniform(0, 50, (N, 3)).astype(np.float32)
    colors = np.random.randint(0, 255, (N, 3), dtype=np.uint8)
    conf = np.random.uniform(0.3, 1.0, N).astype(np.float32)
    cloud = PointCloudModel(points=points, colors=colors, confidence=conf)

    with tempfile.TemporaryDirectory() as tmpdir:
        las_path = os.path.join(tmpdir, "pointcloud.las")
        export_pointcloud_las(
            cloud,
            las_path,
            epsg_code=32643
        )

        assert os.path.exists(las_path)
        import laspy
        las = laspy.read(las_path)
        assert len(las.points) == N
        assert hasattr(las, "confidence")


def test_export_geotiff_raster():
    H, W = 40, 50
    raster_data = np.linspace(10.0, 50.0, H * W, dtype=np.float32).reshape(H, W)
    bounds = {
        "min_x": 100.0,
        "max_x": 150.0,
        "min_y": 200.0,
        "max_y": 240.0,
        "resolution_m": 1.0
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tif_path = os.path.join(tmpdir, "dsm.tif")
        export_geotiff_raster(
            raster_data,
            bounds,
            tif_path,
            epsg_code=32643,
            nodata=-9999.0
        )

        assert os.path.exists(tif_path)
        import rasterio
        with rasterio.open(tif_path) as src:
            assert src.width == W
            assert src.height == H
            assert src.crs.to_epsg() == 32643


def test_convert_to_fbx():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    with tempfile.TemporaryDirectory() as tmpdir:
        glb_path = os.path.join(tmpdir, "test.glb")
        mesh.export(glb_path)
        fbx_path = os.path.join(tmpdir, "test.fbx")

        # Fallback exporter writes a valid FBX or placeholder sidecar
        res_path = convert_to_fbx(glb_path, fbx_path)
        assert os.path.exists(res_path)
