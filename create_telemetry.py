import cv2
from datetime import datetime, timedelta

video_path = "data/samples/Family.mp4"
srt_path = "data/samples/Family.srt"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

duration_sec = total_frames / fps
start_time = datetime(2026, 10, 11, 10, 0, 0)

# Base coordinates (example drone flight)
base_lat = 11.0168
base_lon = 76.9558
base_alt = 50.0  # 50m survey altitude

with open(srt_path, "w", encoding="utf-8") as f:
    for i in range(int(duration_sec)):
        t_start = timedelta(seconds=i)
        t_end = timedelta(seconds=i + 1)

        # Format timestamps as HH:MM:SS,mmm
        s_str = f"{str(t_start).split('.')[0]},000".zfill(12)
        e_str = f"{str(t_end).split('.')[0]},000".zfill(12)

        # Simulate smooth forward movement along the flight line
        lat = base_lat + (i * 0.00003)
        lon = base_lon + (i * 0.00003)
        alt = base_alt + (0.05 * (i % 5))
        yaw = 45.0 + (0.1 * i)

        f.write(f"{i + 1}\n")
        f.write(f"{s_str} --> {e_str}\n")
        f.write(
            f"HOME({base_lon:.4f},{base_lat:.4f}) 2026.10.11 10:00:{i % 60:02d}\n"
            f"GPS ({lat:.6f}, {lon:.6f}, {alt:.1f}m) BAROMETER: {alt:.1f}m ISO: 100 Shutter: 1/500 F: 2.8\n"
            f"[latitude: {lat:.6f}] [longitude: {lon:.6f}] [rel_altitude: {alt:.3f}] [abs_altitude: {alt + 500.0:.3f}] "
            f"[gimbal_pitch: -30.0] [gimbal_roll: 0.0] [gimbal_yaw: {yaw:.1f}]\n\n"
        )

print(f"Generated matching telemetry for {total_frames} frames ({duration_sec:.1f}s) at {srt_path}")