import math
from typing import Tuple
import numpy as np


# WGS84 Ellipsoid constants
WGS84_A = 6378137.0          # Semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563 # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F) # Semi-minor axis (meters)
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2 # First eccentricity squared
WGS84_E_PRIME2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2 # Second eccentricity squared


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
    """Convert WGS84 geodetic coordinates (lat, lon in degrees, alt in meters) to ECEF (X, Y, Z)."""
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    sin_lam = math.sin(lam)
    cos_lam = math.cos(lam)

    # Radius of curvature in prime vertical
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_phi**2)

    x = (N + alt_m) * cos_phi * cos_lam
    y = (N + alt_m) * cos_phi * sin_lam
    z = (N * (1.0 - WGS84_E2) + alt_m) * sin_phi
    return float(x), float(y), float(z)


def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert ECEF coordinates (X, Y, Z in meters) to WGS84 geodetic coordinates (lat, lon, alt)
    using Bowring's closed-form algorithm (accurate to sub-millimeter).
    """
    p = math.sqrt(x**2 + y**2)
    if p < 1e-6:
        # Polar singularity
        lat = 90.0 if z >= 0 else -90.0
        lon = 0.0
        alt = abs(z) - WGS84_B
        return lat, lon, float(alt)

    theta = math.atan2(z * WGS84_A, p * WGS84_B)
    phi = math.atan2(
        z + WGS84_E_PRIME2 * WGS84_B * math.sin(theta)**3,
        p - WGS84_E2 * WGS84_A * math.cos(theta)**3
    )
    lam = math.atan2(y, x)

    sin_phi = math.sin(phi)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_phi**2)
    alt = p / math.cos(phi) - N

    return math.degrees(phi), math.degrees(lam), float(alt)


def rot_ecef_to_enu(ref_lat_deg: float, ref_lon_deg: float) -> np.ndarray:
    """Compute 3x3 rotation matrix from ECEF to Local ENU frame."""
    phi = math.radians(ref_lat_deg)
    lam = math.radians(ref_lon_deg)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)

    R = np.array([
        [-sin_lam, cos_lam, 0.0],
        [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi],
        [cos_phi * cos_lam, cos_phi * sin_lam, sin_phi]
    ], dtype=np.float64)
    return R


def geodetic_to_enu(
    lat_deg: float, lon_deg: float, alt_m: float,
    ref_lat: float, ref_lon: float, ref_alt: float
) -> Tuple[float, float, float]:
    """Convert WGS84 geodetic coordinates to local East-North-Up (ENU) coordinates (meters)
    relative to a reference tangent origin.
    """
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    x0, y0, z0 = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)

    R = rot_ecef_to_enu(ref_lat, ref_lon)
    diff = np.array([x - x0, y - y0, z - z0], dtype=np.float64)
    enu = R @ diff
    return float(enu[0]), float(enu[1]), float(enu[2])


def enu_to_geodetic(
    e: float, n: float, u: float,
    ref_lat: float, ref_lon: float, ref_alt: float
) -> Tuple[float, float, float]:
    """Convert local ENU coordinates (meters) back to WGS84 geodetic coordinates."""
    x0, y0, z0 = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)
    R = rot_ecef_to_enu(ref_lat, ref_lon)
    # Inverse rotation is transpose
    enu = np.array([e, n, u], dtype=np.float64)
    ecef_diff = R.T @ enu

    x = x0 + ecef_diff[0]
    y = y0 + ecef_diff[1]
    z = z0 + ecef_diff[2]

    return ecef_to_geodetic(x, y, z)


def umeyama_sim3(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    with_scale: bool = True
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Umeyama closed-form algorithm for Sim(3) alignment.
    Finds s, R, t such that: dst ≈ s * src @ R.T + t.
    src_points: (N, 3)
    dst_points: (N, 3)
    Returns: (scale, R (3x3), t (3,))
    """
    assert src_points.shape == dst_points.shape
    N, m = src_points.shape
    assert m == 3 and N >= 3

    # Means
    mu_src = np.mean(src_points, axis=0)
    mu_dst = np.mean(dst_points, axis=0)

    # Demeaned
    src_c = src_points - mu_src
    dst_c = dst_points - mu_dst

    # Covariance
    cov = (dst_c.T @ src_c) / N

    # SVD
    U, D, Vt = np.linalg.svd(cov)

    # Reflection check
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    if with_scale:
        var_src = np.sum(src_c**2) / N
        scale = float((1.0 / var_src) * np.trace(np.diag(D) @ S))
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def align_gravity_and_ground_plane(
    gravity_vector: np.ndarray,
    ground_plane_normal: np.ndarray
) -> np.ndarray:
    """Compute 3x3 rotation matrix to align the estimated upward vector with ENU-Up [0, 0, 1].
    Observability trap fix (§4.3 of Spec): fuses IMU gravity and point cloud ground normal.
    """
    # Estimated upward vector: opposite to gravity, and pointing along positive ground normal
    up_grav = -gravity_vector / np.linalg.norm(gravity_vector)
    up_norm = ground_plane_normal / np.linalg.norm(ground_plane_normal)
    if np.dot(up_grav, up_norm) < 0:
        up_norm = -up_norm

    # Blended up estimate
    up_est = 0.5 * up_grav + 0.5 * up_norm
    up_est /= np.linalg.norm(up_est)

    target_up = np.array([0.0, 0.0, 1.0])

    # Rotation vector from up_est to target_up
    v = np.cross(up_est, target_up)
    c = np.dot(up_est, target_up)
    s = np.linalg.norm(v)

    if s < 1e-6:
        if c > 0:
            return np.eye(3)
        else:
            # 180 deg flip around x
            return np.diag([1.0, -1.0, -1.0])

    # Rodrigues' rotation formula
    v_skew = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])
    R = np.eye(3) + v_skew + (v_skew @ v_skew) * ((1.0 - c) / (s**2))
    return R
