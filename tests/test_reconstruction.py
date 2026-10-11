import numpy as np
import pytest

from src.common.types import CameraPose, PointCloudModel, KeyframeData, ConfidenceTag
from src.reconstruction.fast_ortho_dsm import FastOrthoDSMGenerator
from src.reconstruction.surface_mesher import PoissonSurfaceMesher
from src.reconstruction.texture_projector import MeshTextureProjector
from src.reconstruction.gaussian_splatting import GaussianSplatModel


def test_fast_ortho_dsm_generation():
    # 3D points representing a 20m x 20m terrain patch at Z=0 with a 5m raised mound at center
    N = 1000
    np.random.seed(42)
    x = np.random.uniform(-10, 10, N)
    y = np.random.uniform(-10, 10, N)
    z = np.zeros(N)
    # Mound in center
    mound = (x**2 + y**2 < 9.0)
    z[mound] = 5.0

    points = np.column_stack([x, y, z]).astype(np.float32)
    colors = np.full((N, 3), 100, dtype=np.uint8)
    colors[mound] = [50, 200, 50]
    conf = np.ones(N, dtype=np.float32) * 0.9

    cloud = PointCloudModel(points=points, colors=colors, confidence=conf)

    generator = FastOrthoDSMGenerator(resolution_m=0.5)
    ortho, dsm, bounds = generator.generate(cloud)

    assert ortho.shape[:2] == dsm.shape
    assert bounds["min_x"] <= -10.0
    assert bounds["max_x"] >= 10.0
    # The center of the DSM should have elevation close to 5.0m
    assert np.max(dsm) == pytest.approx(5.0, abs=0.5)


def test_poisson_surface_mesher():
    # Sample points on a sphere of radius 2.0 with outward normals
    N = 500
    np.random.seed(42)
    u = np.random.uniform(0, 2 * np.pi, N)
    v = np.random.uniform(0, np.pi, N)
    r = 2.0
    x = r * np.sin(v) * np.cos(u)
    y = r * np.sin(v) * np.sin(u)
    z = r * np.cos(v)

    points = np.column_stack([x, y, z]).astype(np.float32)
    normals = (points / r).astype(np.float32)
    colors = np.full((N, 3), 180, dtype=np.uint8)
    conf = np.ones(N, dtype=np.float32) * 0.95

    cloud = PointCloudModel(points=points, colors=colors, confidence=conf, normals=normals)

    mesher = PoissonSurfaceMesher(depth=7, density_trim_quantile=0.02)
    mesh = mesher.reconstruct(cloud)

    assert mesh is not None
    assert len(mesh.vertices) > 20
    assert len(mesh.faces) > 20


def test_texture_projector():
    # Simple single-triangle mesh
    vertices = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Keyframe facing the triangle from z=5m
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 5.0])
    pose = CameraPose(0, R, t)

    kf = KeyframeData(0, 0.0, img, 100.0)

    projector = MeshTextureProjector(K)
    colored_mesh = projector.project_texture(mesh, [kf], [pose])

    assert colored_mesh.visual is not None


def test_gaussian_splat_initialization():
    points = np.random.uniform(-5, 5, (50, 3)).astype(np.float32)
    colors = np.random.randint(0, 255, (50, 3), dtype=np.uint8)
    conf = np.ones(50, dtype=np.float32) * 0.8
    cloud = PointCloudModel(points=points, colors=colors, confidence=conf)

    gs_model = GaussianSplatModel.from_point_cloud(cloud)

    assert gs_model.num_gaussians == 50
    assert gs_model.positions.shape == (50, 3)
    assert gs_model.scales.shape == (50, 3)
    assert gs_model.opacities.shape == (50, 1)
