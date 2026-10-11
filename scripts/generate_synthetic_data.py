import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from src.common.coordinates import enu_to_geodetic
from src.sensor_fusion.camera_calibration import CameraIntrinsics


def generate_synthetic_dataset(
    output_dir: str = "data/samples",
    num_frames: int = 120,
    fps: float = 10.0,
    width: int = 640,
    height: int = 360,
    flight_speed_m_s: float = 6.0,
    flight_altitude_m: float = 60.0,
    ref_lat: float = 11.0168,
    ref_lon: float = 76.9558,
    ref_alt: float = 400.0
):
    """Generates synthetic single-pass drone flight video and synchronized SRT/CSV telemetry.
    Includes known-dimension ground validation targets (§13.1 of Spec):
    - Tennis Court: 23.77m x 10.97m
    - Shipping Container: 12.19m x 2.44m
    - Two rectangular buildings with flat roofs.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "synthetic_drone_flight.mp4")
    srt_path = os.path.join(output_dir, "synthetic_drone_flight.srt")
    csv_path = os.path.join(output_dir, "synthetic_drone_flight.csv")
    gt_path = os.path.join(output_dir, "ground_truth_targets.json")

    # Camera calibration (focal length ~600px for 640x360)
    fx = fy = 500.0
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Oblique downward gimbal: pitch -60 deg (pointing forward-down)
    pitch_rad = np.deg2rad(-60.0)
    # Camera orientation in ENU:
    # Camera Z forward, X right, Y down
    # Heading along +Y (North), pitch tilts downwards around X
    R_gimbal = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0.0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    # ENU to camera transform: ENU X(East) -> Cam X(Right), ENU Y(North) -> Cam -Z(Up), ENU Z(Up) -> Cam -Y
    R_enu_to_cam0 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0]
    ])
    R_cam = R_gimbal @ R_enu_to_cam0

    # 3D Scene Geometry: Ground features and buildings in Local ENU (meters)
    # Ground terrain plane Z = 0
    ground_features = []
    # Grid of textured points/dots on terrain
    for x in np.linspace(-40, 40, 30):
        for y in np.linspace(-20, 120, 50):
            # Grass / soil colors
            c = (int(np.random.randint(40, 70)), int(np.random.randint(110, 160)), int(np.random.randint(30, 60)))
            ground_features.append((np.array([x, y, 0.0]), c, 2))

    # Road strip along Y axis (North-South): X from -5 to +5, Y from -20 to 120
    for y in np.linspace(-20, 120, 80):
        ground_features.append((np.array([-4.5, y, 0.0]), (80, 80, 80), 3))
        ground_features.append((np.array([4.5, y, 0.0]), (80, 80, 80), 3))
        # Yellow dashed center line
        if int(y) % 6 < 3:
            ground_features.append((np.array([0.0, y, 0.0]), (0, 220, 220), 2))

    # Building 1: Center at (15, 30), size 20m (X) x 15m (Y) x 12m (Z height)
    b1_corners = [
        np.array([5.0, 22.5, 12.0]), np.array([25.0, 22.5, 12.0]),
        np.array([25.0, 37.5, 12.0]), np.array([5.0, 37.5, 12.0])
    ]
    # Building 2: Center at (-18, 60), size 16m x 14m x 15m height
    b2_corners = [
        np.array([-26.0, 53.0, 15.0]), np.array([-10.0, 53.0, 15.0]),
        np.array([-10.0, 67.0, 15.0]), np.array([-26.0, 67.0, 15.0])
    ]
    # Tennis court: Center at (12, 70), 10.97m (X) x 23.77m (Y) on ground
    tc_corners = [
        np.array([6.5, 58.1, 0.0]), np.array([17.5, 58.1, 0.0]),
        np.array([17.5, 81.9, 0.0]), np.array([6.5, 81.9, 0.0])
    ]
    # Container: Center at (-10, 20), 2.44m (X) x 12.19m (Y) x 2.6m (Z)
    cont_corners = [
        np.array([-11.2, 13.9, 2.6]), np.array([-8.8, 13.9, 2.6]),
        np.array([-8.8, 26.1, 2.6]), np.array([-11.2, 26.1, 2.6])
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    srt_entries = []
    csv_rows = ["timestamp,latitude,longitude,altitude_msl,altitude_rel,speed,pitch,roll,yaw\n"]

    for i in range(num_frames):
        t = i / fps
        # Drone flies straight along +Y (North) at flight_speed_m_s
        drone_enu = np.array([0.0, t * flight_speed_m_s, flight_altitude_m])

        # Project 3D points into camera view: x_cam = R_cam @ (X_enu - drone_enu)
        # Background: sky / horizon
        frame = np.full((height, width, 3), (180, 210, 235), dtype=np.uint8) # Sky blue
        # Ground half
        horizon_y = int(height * 0.35)
        frame[horizon_y:, :] = (50, 110, 50) # Green terrain

        # Helper to project 3D ENU point
        def project(pt):
            p_cam = R_cam @ (pt - drone_enu)
            if p_cam[2] <= 0.5:
                return None
            u = int(fx * p_cam[0] / p_cam[2] + cx)
            v = int(fy * p_cam[1] / p_cam[2] + cy)
            if 0 <= u < width and 0 <= v < height:
                return (u, v)
            return None

        # Draw ground features
        for pt, col, rad in ground_features:
            px = project(pt)
            if px:
                cv2.circle(frame, px, rad, col, -1)

        # Draw Tennis Court (blue/green court)
        pts_tc = [project(p) for p in tc_corners]
        if all(p is not None for p in pts_tc):
            cv2.fillPoly(frame, [np.array(pts_tc, dtype=np.int32)], (140, 100, 40))
            cv2.polylines(frame, [np.array(pts_tc, dtype=np.int32)], True, (255, 255, 255), 2)

        # Draw Container (red shipping container)
        pts_cont = [project(p) for p in cont_corners]
        if all(p is not None for p in pts_cont):
            cv2.fillPoly(frame, [np.array(pts_cont, dtype=np.int32)], (30, 40, 180))
            cv2.polylines(frame, [np.array(pts_cont, dtype=np.int32)], True, (255, 255, 255), 1)

        # Draw Building 1 (terracotta rooftop)
        pts_b1 = [project(p) for p in b1_corners]
        if all(p is not None for p in pts_b1):
            cv2.fillPoly(frame, [np.array(pts_b1, dtype=np.int32)], (70, 90, 190))
            cv2.polylines(frame, [np.array(pts_b1, dtype=np.int32)], True, (255, 255, 255), 2)

        # Draw Building 2 (grey rooftop)
        pts_b2 = [project(p) for p in b2_corners]
        if all(p is not None for p in pts_b2):
            cv2.fillPoly(frame, [np.array(pts_b2, dtype=np.int32)], (150, 150, 150))
            cv2.polylines(frame, [np.array(pts_b2, dtype=np.int32)], True, (255, 255, 255), 2)

        out.write(frame)

        # Compute synchronized GPS telemetry
        lat, lon, alt_msl = enu_to_geodetic(
            drone_enu[0], drone_enu[1], drone_enu[2], ref_lat, ref_lon, ref_alt
        )

        # Format SRT entry
        m, s = divmod(t, 60.0)
        h, m = divmod(m, 60.0)
        ms = int((t - int(t)) * 1000)
        t_str1 = f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"
        t2 = t + (1.0 / fps)
        m2, s2 = divmod(t2, 60.0)
        h2, m2 = divmod(m2, 60.0)
        ms2 = int((t2 - int(t2)) * 1000)
        t_str2 = f"{int(h2):02d}:{int(m2):02d}:{int(s2):02d},{ms2:03d}"

        srt_block = f"""{i+1}
{t_str1} --> {t_str2}
HOME({ref_lon:.4f},{ref_lat:.4f})
GPS({lon:.6f}, {lat:.6f}, 16) BAROMETER: {flight_altitude_m:.1f}M
[latitude: {lat:.6f}] [longitude: {lon:.6f}] [rel_altitude: {flight_altitude_m:.3f}] [abs_altitude: {alt_msl:.3f}]
[gimbal_pitch: -60.0] [gimbal_roll: 0.0] [gimbal_yaw: 0.0]
"""
        srt_entries.append(srt_block)
        csv_rows.append(f"{t:.2f},{lat:.7f},{lon:.7f},{alt_msl:.2f},{flight_altitude_m:.2f},{flight_speed_m_s:.1f},-60.0,0.0,0.0\n")

    out.release()

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_entries))

    with open(csv_path, "w", encoding="utf-8") as f:
        f.writelines(csv_rows)

    import json
    gt_data = {
        "tennis_court": {"length_m": 23.77, "width_m": 10.97, "center_enu": [12.0, 70.0, 0.0]},
        "container": {"length_m": 12.19, "width_m": 2.44, "height_m": 2.6, "center_enu": [-10.0, 20.0, 1.3]},
        "building_1": {"width_x_m": 20.0, "length_y_m": 15.0, "height_z_m": 12.0, "center_enu": [15.0, 30.0, 6.0]},
        "building_2": {"width_x_m": 16.0, "length_y_m": 14.0, "height_z_m": 15.0, "center_enu": [-18.0, 60.0, 7.5]}
    }
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, indent=2)

    return video_path, srt_path, csv_path, gt_path


if __name__ == "__main__":
    v, s, c, g = generate_synthetic_dataset()
    print(f"Generated synthetic drone pass: {v}, {s}, {c}, {g}")
