import numpy as np
import pytest

from src.common.coordinates import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    geodetic_to_enu,
    enu_to_geodetic,
    umeyama_sim3,
    align_gravity_and_ground_plane
)


def test_wgs84_ecef_roundtrip():
    # Test point: New Delhi area (approx lat 28.6139, lon 77.2090, alt 216.0)
    lat, lon, alt = 28.6139, 77.2090, 216.0
    x, y, z = geodetic_to_ecef(lat, lon, alt)
    lat_r, lon_r, alt_r = ecef_to_geodetic(x, y, z)

    assert pytest.approx(lat, abs=1e-7) == lat_r
    assert pytest.approx(lon, abs=1e-7) == lon_r
    assert pytest.approx(alt, abs=1e-3) == alt_r


def test_geodetic_to_enu_origin_is_zero():
    ref_lat, ref_lon, ref_alt = 11.0168, 76.9558, 411.0 # Coimbatore
    e, n, u = geodetic_to_enu(ref_lat, ref_lon, ref_alt, ref_lat, ref_lon, ref_alt)
    assert pytest.approx(0.0, abs=1e-6) == e
    assert pytest.approx(0.0, abs=1e-6) == n
    assert pytest.approx(0.0, abs=1e-6) == u


def test_enu_roundtrip():
    ref_lat, ref_lon, ref_alt = 11.0168, 76.9558, 411.0
    # Point displaced 100m East, 200m North, 50m Up
    orig_e, orig_n, orig_u = 100.0, 200.0, 50.0
    lat, lon, alt = enu_to_geodetic(orig_e, orig_n, orig_u, ref_lat, ref_lon, ref_alt)
    e, n, u = geodetic_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)

    assert pytest.approx(orig_e, abs=1e-3) == e
    assert pytest.approx(orig_n, abs=1e-3) == n
    assert pytest.approx(orig_u, abs=1e-3) == u


def test_umeyama_sim3_exact_recovery():
    # Synthetic random 3D points
    np.random.seed(42)
    src = np.random.uniform(-50, 50, (30, 3))

    true_scale = 1.75
    # Rotation by 30 deg around z-axis
    theta = np.deg2rad(30.0)
    true_R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0]
    ])
    true_t = np.array([12.5, -45.2, 5.8])

    dst = true_scale * (src @ true_R.T) + true_t

    s_est, R_est, t_est = umeyama_sim3(src, dst)

    assert pytest.approx(true_scale, abs=1e-4) == s_est
    assert np.allclose(true_R, R_est, atol=1e-4)
    assert np.allclose(true_t, t_est, atol=1e-4)


def test_align_gravity_and_ground_plane():
    # Points on horizontal ground plane with normal [0, 0, 1] tilted by 10 deg around x-axis
    alpha = np.deg2rad(10.0)
    R_tilt = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(alpha), -np.sin(alpha)],
        [0.0, np.sin(alpha), np.cos(alpha)]
    ])
    ground_normal_measured = R_tilt @ np.array([0.0, 0.0, 1.0])
    # Measured gravity should point down opposite to up
    gravity_vector = -ground_normal_measured

    R_correct = align_gravity_and_ground_plane(gravity_vector, ground_normal_measured)
    corrected_normal = R_correct @ ground_normal_measured

    # Corrected normal should align with ENU-Up [0, 0, 1]
    assert np.allclose(corrected_normal, np.array([0.0, 0.0, 1.0]), atol=1e-4)
