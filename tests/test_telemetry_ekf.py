import tempfile
import numpy as np
import pytest

from src.sensor_fusion.telemetry_parser import (
    parse_dji_srt,
    parse_csv_telemetry,
    TelemetryLog
)
from src.sensor_fusion.time_sync import estimate_time_offset
from src.sensor_fusion.ekf_smoother import TrajectoryEKFSmoother
from src.sensor_fusion.camera_calibration import CameraIntrinsics


def test_parse_dji_srt_file():
    srt_content = """1
00:00:01,000 --> 00:00:01,033
HOME(76.9558,11.0168) 2026.09.15 10:00:00
GPS(76.9558, 11.0168, 15) BAROMETER: 100.5M
ISO: 100 Shutter: 1/1000 EV: 0 Fnum: F2.8
[latitude: 11.016850] [longitude: 76.955850] [rel_altitude: 100.500] [abs_altitude: 512.500]
[gimbal_pitch: -45.0] [gimbal_roll: 0.0] [gimbal_yaw: 90.0]

2
00:00:02,000 --> 00:00:02,033
HOME(76.9558,11.0168) 2026.09.15 10:00:01
GPS(76.9559, 11.0169, 16) BAROMETER: 100.8M
ISO: 100 Shutter: 1/1000 EV: 0 Fnum: F2.8
[latitude: 11.016920] [longitude: 76.955910] [rel_altitude: 100.800] [abs_altitude: 512.800]
[gimbal_pitch: -45.0] [gimbal_roll: 0.0] [gimbal_yaw: 90.0]
"""
    with tempfile.NamedTemporaryFile("w+", suffix=".srt", delete=False) as f:
        f.write(srt_content)
        f.flush()
        samples = parse_dji_srt(f.name)

    assert len(samples) == 2
    assert samples[0].timestamp == pytest.approx(1.0)
    assert samples[0].latitude == pytest.approx(11.016850)
    assert samples[0].longitude == pytest.approx(76.955850)
    assert samples[0].altitude_rel == pytest.approx(100.5)
    assert samples[0].pitch == pytest.approx(-45.0)


def test_parse_csv_telemetry():
    csv_content = """timestamp,latitude,longitude,altitude_msl,altitude_rel,speed,pitch,roll,yaw
0.0,11.0168,76.9558,500.0,100.0,5.5,0.0,0.0,45.0
1.0,11.0169,76.9559,500.5,100.5,5.6,0.0,0.0,45.0
"""
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        samples = parse_csv_telemetry(f.name)

    assert len(samples) == 2
    assert samples[1].timestamp == 1.0
    assert samples[1].speed == 5.6
    assert samples[1].yaw == 45.0


def test_time_sync_cross_correlation():
    # Synthetic ground speed with true time shift of 0.5 seconds
    t = np.linspace(0, 30, 300)
    true_offset = 0.5
    # Signal with a distinctive acceleration peak
    v_vo = 5.0 + 2.0 * np.sin(0.5 * t) + 3.0 * np.exp(-((t - 15.0) ** 2) / 4.0)

    # GNSS speeds sampled at time t + true_offset
    v_gnss = 5.0 + 2.0 * np.sin(0.5 * (t + true_offset)) + 3.0 * np.exp(-((t + true_offset - 15.0) ** 2) / 4.0)
    # Add small measurement noise
    v_gnss += np.random.normal(0, 0.05, len(v_gnss))

    est_offset = estimate_time_offset(t, v_vo, t, v_gnss, max_window_sec=2.0)
    assert pytest.approx(true_offset, abs=0.15) == est_offset


def test_ekf_rts_smoother_trajectory():
    # Simulate a 10-second flight path with constant velocity in local ENU
    dt = 0.1
    times = np.arange(0, 10.0, dt)
    true_x = 5.0 * times # 5 m/s East
    true_y = 2.0 * times # 2 m/s North
    true_z = 100.0 + 0.5 * np.sin(times) # 100m Up with slight oscillation

    # Generate noisy GPS measurements (std = 1.5m)
    np.random.seed(42)
    measured_gps = np.column_stack([
        true_x + np.random.normal(0, 1.5, len(times)),
        true_y + np.random.normal(0, 1.5, len(times)),
        true_z + np.random.normal(0, 2.0, len(times))
    ])

    smoother = TrajectoryEKFSmoother(
        gps_std=1.5,
        baro_std=0.5,
        accel_std=0.2
    )

    smoothed_traj = smoother.filter_and_smooth(times, measured_gps)
    assert smoothed_traj.shape == (len(times), 3)

    # The smoothed trajectory should have lower error than raw noisy measurements
    raw_rmse = np.sqrt(np.mean((measured_gps - np.column_stack([true_x, true_y, true_z])) ** 2))
    smooth_rmse = np.sqrt(np.mean((smoothed_traj - np.column_stack([true_x, true_y, true_z])) ** 2))

    assert smooth_rmse < raw_rmse
    assert smooth_rmse < 1.0 # Meets <= 1m smoothing requirement


def test_camera_calibration_intrinsics():
    calib = CameraIntrinsics.from_sensor_specs(
        image_width=3840,
        image_height=2160,
        sensor_width_mm=13.2,
        sensor_height_mm=8.8,
        focal_length_mm=8.8
    )
    K = calib.intrinsic_matrix
    assert K[0, 0] == pytest.approx(3840 * 8.8 / 13.2)
    assert K[1, 1] == pytest.approx(2160 * 8.8 / 8.8)
    assert K[0, 2] == pytest.approx(3840 / 2.0)
    assert K[1, 2] == pytest.approx(2160 / 2.0)
