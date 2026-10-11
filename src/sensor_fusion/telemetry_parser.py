import csv
import re
from typing import List, Optional
import numpy as np

from src.common.types import TelemetrySample


class TelemetryLog:
    """Encapsulates a time-series of drone telemetry samples with query & interpolation methods."""

    def __init__(self, samples: List[TelemetrySample]):
        self.samples = sorted(samples, key=lambda s: s.timestamp)
        self.timestamps = np.array([s.timestamp for s in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def sample_at_time(self, t: float) -> TelemetrySample:
        """Interpolate telemetry at time t."""
        if not self.samples:
            raise ValueError("Telemetry log is empty")
        if t <= self.timestamps[0]:
            return self.samples[0]
        if t >= self.timestamps[-1]:
            return self.samples[-1]

        idx = np.searchsorted(self.timestamps, t)
        t0, t1 = self.timestamps[idx - 1], self.timestamps[idx]
        s0, s1 = self.samples[idx - 1], self.samples[idx]
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0

        lat = (1.0 - alpha) * s0.latitude + alpha * s1.latitude
        lon = (1.0 - alpha) * s0.longitude + alpha * s1.longitude
        alt_msl = (1.0 - alpha) * s0.altitude_msl + alpha * s1.altitude_msl
        alt_rel = (1.0 - alpha) * s0.altitude_rel + alpha * s1.altitude_rel
        speed = (1.0 - alpha) * s0.speed + alpha * s1.speed
        pitch = (1.0 - alpha) * s0.pitch + alpha * s1.pitch
        roll = (1.0 - alpha) * s0.roll + alpha * s1.roll
        yaw = (1.0 - alpha) * s0.yaw + alpha * s1.yaw

        return TelemetrySample(
            timestamp=t,
            latitude=lat,
            longitude=lon,
            altitude_msl=alt_msl,
            altitude_rel=alt_rel,
            pitch=pitch,
            roll=roll,
            yaw=yaw,
            speed=speed
        )


def _parse_srt_timestamp(time_str: str) -> float:
    """Parse HH:MM:SS,mmm to seconds."""
    parts = time_str.strip().split("-->")[0].strip()
    match = re.match(r"(\d+):(\d+):(\d+)[,\.](\d+)", parts)
    if match:
        h, m, s, ms = match.groups()
        return int(h) * 3600.0 + int(m) * 60.0 + int(s) + int(ms) / 1000.0
    return 0.0


def parse_dji_srt(srt_path: str) -> List[TelemetrySample]:
    """Parse standard DJI SRT subtitle log files."""
    samples: List[TelemetrySample] = []
    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) < 2:
            continue

        # Line with timestamp
        time_line = [l for l in lines if "-->" in l]
        if not time_line:
            continue
        ts = _parse_srt_timestamp(time_line[0])

        block_text = " ".join(lines)

        # Regex extractors
        lat_m = re.search(r"\[latitude:\s*([-+]?\d*\.?\d+)\]", block_text, re.IGNORECASE)
        lon_m = re.search(r"\[longitude:\s*([-+]?\d*\.?\d+)\]", block_text, re.IGNORECASE)
        rel_alt_m = re.search(r"\[rel_altitude:\s*([-+]?\d*\.?\d+)\]", block_text, re.IGNORECASE)
        abs_alt_m = re.search(r"\[abs_altitude:\s*([-+]?\d*\.?\d+)\]", block_text, re.IGNORECASE)
        pitch_m = re.search(r"\[gimbal_pitch:\s*([-+]?\d*\.?\d+)\]", block_text, re.IGNORECASE)
        roll_m = re.search(r"\[gimbal_roll:\s*([-+]?\d*\.?\d+)\]", block_text, re.IGNORECASE)
        yaw_m = re.search(r"\[gimbal_yaw:\s*([-+]?\d*\.?\d+)\]", block_text, re.IGNORECASE)

        # Fallback to GPS(lat, lon) format if brackets not found
        if not lat_m or not lon_m:
            gps_m = re.search(r"GPS\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)", block_text)
            if gps_m:
                lon_val = float(gps_m.group(1))
                lat_val = float(gps_m.group(2))
            else:
                continue
        else:
            lat_val = float(lat_m.group(1))
            lon_val = float(lon_m.group(1))

        rel_alt = float(rel_alt_m.group(1)) if rel_alt_m else 0.0
        abs_alt = float(abs_alt_m.group(1)) if abs_alt_m else rel_alt
        pitch = float(pitch_m.group(1)) if pitch_m else 0.0
        roll = float(roll_m.group(1)) if roll_m else 0.0
        yaw = float(yaw_m.group(1)) if yaw_m else 0.0

        samples.append(TelemetrySample(
            timestamp=ts,
            latitude=lat_val,
            longitude=lon_val,
            altitude_msl=abs_alt,
            altitude_rel=rel_alt,
            pitch=pitch,
            roll=roll,
            yaw=yaw
        ))

    return samples


def parse_csv_telemetry(csv_path: str) -> List[TelemetrySample]:
    """Parse CSV telemetry log file."""
    samples: List[TelemetrySample] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row.get("timestamp", row.get("time", 0.0)))
            lat = float(row.get("latitude", row.get("lat", 0.0)))
            lon = float(row.get("longitude", row.get("lon", 0.0)))
            alt_msl = float(row.get("altitude_msl", row.get("alt", 0.0)))
            alt_rel = float(row.get("altitude_rel", row.get("rel_alt", alt_msl)))
            speed = float(row.get("speed", 0.0))
            pitch = float(row.get("pitch", 0.0))
            roll = float(row.get("roll", 0.0))
            yaw = float(row.get("yaw", row.get("heading", 0.0)))

            samples.append(TelemetrySample(
                timestamp=ts,
                latitude=lat,
                longitude=lon,
                altitude_msl=alt_msl,
                altitude_rel=alt_rel,
                pitch=pitch,
                roll=roll,
                yaw=yaw,
                speed=speed
            ))

    return samples
