import numpy as np
import pytest

from src.common.types import CameraPose, ConfidenceTag
from src.depth.monocular_estimator import MonocularDepthEstimator
from src.depth.metric_fusion import align_relative_depth_to_metric
from src.depth.confidence_tagger import EpistemicConfidenceTagger
from src.depth.unprojector import PseudoLiDARUnprojector


def test_monocular_depth_estimator_synthetic_fallback():
    estimator = MonocularDepthEstimator(use_mock=True)
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    depth_rel = estimator.infer_depth(img)

    assert depth_rel.shape == (240, 320)
    assert np.all(depth_rel >= 0.0)
    assert np.all(depth_rel <= 1.0)


def test_align_relative_depth_to_metric_huber():
    # Synthetic relative depth map (affine: rel = 0.05 * metric - 0.2)
    H, W = 100, 100
    true_metric = np.linspace(10.0, 50.0, H * W).reshape(H, W)
    true_s = 20.0 # metric scale
    true_b = 4.0  # metric shift
    # rel = (metric - true_b) / true_s
    rel_depth = (true_metric - true_b) / true_s

    # 15 sparse anchor measurements
    np.random.seed(42)
    u_coords = np.random.randint(10, 90, 15)
    v_coords = np.random.randint(10, 90, 15)
    sparse_metric_depths = true_metric[v_coords, u_coords]

    fitted_metric_depth, s_fit, b_fit = align_relative_depth_to_metric(
        rel_depth,
        u_coords,
        v_coords,
        sparse_metric_depths
    )

    assert pytest.approx(true_s, abs=0.5) == s_fit
    assert pytest.approx(true_b, abs=0.5) == b_fit
    assert np.allclose(fitted_metric_depth, true_metric, atol=0.5)


def test_epistemic_confidence_tagger():
    H, W = 60, 80
    tagger = EpistemicConfidenceTagger()

    # Anchor at center (40, 30)
    sparse_uvs = np.array([[40, 30]])
    triangulation_angles_deg = np.array([8.0]) # Good parallax

    conf_map, tag_map = tagger.compute_confidence(
        image_shape=(H, W),
        sparse_uvs=sparse_uvs,
        triangulation_angles=triangulation_angles_deg
    )

    assert conf_map.shape == (H, W)
    # Center pixel should have high confidence (Green)
    assert conf_map[30, 40] >= 0.70
    assert tag_map[30, 40] == ConfidenceTag.GREEN

    # Far corner (0, 0) is distant from anchors -> lower confidence (Yellow or Red)
    assert conf_map[0, 0] < conf_map[30, 40]


def test_pseudolidar_unprojector():
    K = np.array([[500.0, 0.0, 50.0], [0.0, 500.0, 50.0], [0.0, 0.0, 1.0]])
    H, W = 100, 100

    depth_metric = np.full((H, W), 20.0, dtype=np.float32) # Uniform depth = 20m
    confidence_map = np.full((H, W), 0.85, dtype=np.float32)
    color_img = np.full((H, W, 3), 120, dtype=np.uint8)

    pose = CameraPose(0, np.eye(3), np.zeros(3))
    unprojector = PseudoLiDARUnprojector(K, max_depth_m=100.0)

    cloud = unprojector.unproject_frame(depth_metric, color_img, confidence_map, pose)

    assert len(cloud.points) == H * W
    assert pytest.approx(20.0, abs=0.1) == cloud.points[H//2 * W + W//2, 2] # Z coordinate
    assert cloud.tags[0] == ConfidenceTag.GREEN
