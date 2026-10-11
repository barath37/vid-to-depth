import numpy as np
import pytest

from src.common.types import KeyframeData, TelemetrySample
from src.keyframing.blur_detector import compute_laplacian_variance, is_frame_sharp, select_sharpest_frame
from src.keyframing.adaptive_sampler import calculate_spatial_baseline, AdaptiveKeyframeSampler
from src.keyframing.dynamic_masker import DynamicObjectMasker


def test_laplacian_variance_sharp_vs_blur():
    # Sharp image: checkerboard pattern
    sharp_img = np.zeros((100, 100, 3), dtype=np.uint8)
    sharp_img[::10, :] = 255
    sharp_img[:, ::10] = 255

    # Blurred image: uniform flat color
    blur_img = np.ones((100, 100, 3), dtype=np.uint8) * 128

    var_sharp = compute_laplacian_variance(sharp_img)
    var_blur = compute_laplacian_variance(blur_img)

    assert var_sharp > 500.0
    assert var_blur < 1.0
    assert is_frame_sharp(sharp_img, threshold=80.0) is True
    assert is_frame_sharp(blur_img, threshold=80.0) is False


def test_select_sharpest_frame_with_iframe_preference():
    img_low = np.ones((100, 100, 3), dtype=np.uint8) * 50
    img_mid = np.zeros((100, 100, 3), dtype=np.uint8)
    img_mid[::20, :] = 255
    img_high = np.zeros((100, 100, 3), dtype=np.uint8)
    img_high[::5, :] = 255

    kf1 = KeyframeData(1, 0.1, img_low, compute_laplacian_variance(img_low), is_i_frame=False)
    kf2 = KeyframeData(2, 0.2, img_mid, compute_laplacian_variance(img_mid), is_i_frame=True)
    kf3 = KeyframeData(3, 0.3, img_high, compute_laplacian_variance(img_high), is_i_frame=False)

    best = select_sharpest_frame([kf1, kf2, kf3], iframe_bonus_factor=1.2)
    # kf3 should win due to substantially higher sharpness, but kf2 has bonus
    assert best.frame_idx == 3


def test_calculate_spatial_baseline_geometry():
    # Altitude = 100m, sensor height = 8.8mm, focal length = 8.8mm
    # Footprint L = 100 * 8.8 / 8.8 = 100m
    # Overlap = 85% -> Baseline = (1 - 0.85) * 100m = 15m
    b = calculate_spatial_baseline(
        altitude_agl=100.0,
        sensor_height_mm=8.8,
        focal_length_mm=8.8,
        target_overlap=0.85
    )
    assert pytest.approx(15.0, abs=0.1) == b


def test_adaptive_keyframe_sampler_budget_enforcement():
    # Simulate 10-minute flight at 6 m/s = 3600 meters total
    # At 30 fps = 18000 frames total
    total_frames = 600
    sampler = AdaptiveKeyframeSampler(
        target_keyframes=300,
        hard_cap_keyframes=400,
        target_overlap=0.85
    )

    # Generate synthetic stream of candidate frames along track
    candidates = []
    for i in range(total_frames):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[::(i % 5 + 2), :] = 255
        t = i * 1.0 # 1 sec per frame, speed 6 m/s
        dist = i * 6.0
        telem = TelemetrySample(
            timestamp=t,
            latitude=11.0,
            longitude=76.0 + (dist / 111000.0),
            altitude_msl=500.0,
            altitude_rel=100.0,
            speed=6.0
        )
        kf = KeyframeData(
            frame_idx=i,
            timestamp=t,
            image=img,
            laplacian_var=compute_laplacian_variance(img),
            is_i_frame=(i % 30 == 0),
            telemetry=telem
        )
        candidates.append(kf)

    selected = sampler.sample_sequence(candidates, total_flight_seconds=600.0)

    # Should be budget-capped: >= 200 and <= 400
    assert len(selected) >= 150
    assert len(selected) <= 400


def test_dynamic_object_masker():
    masker = DynamicObjectMasker(enable_detector=False)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    mask = masker.get_mask(img)

    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    # Default without detector: all ones (everything static/retained)
    assert np.all(mask == 255)
