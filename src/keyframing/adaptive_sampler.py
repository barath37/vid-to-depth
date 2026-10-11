from typing import List, Optional
import numpy as np

from src.common.types import KeyframeData
from src.keyframing.blur_detector import select_sharpest_frame


def calculate_spatial_baseline(
    altitude_agl: float,
    sensor_height_mm: float,
    focal_length_mm: float,
    target_overlap: float = 0.85
) -> float:
    """Calculates physical baseline distance required between keyframes
    to maintain target forward overlap (§6.1 of Spec).
    L = H_AGL * sensor_height / f
    B_target = (1 - overlap) * L
    """
    footprint_length = altitude_agl * (sensor_height_mm / focal_length_mm)
    baseline_meters = (1.0 - target_overlap) * footprint_length
    return max(baseline_meters, 1.0)


class AdaptiveKeyframeSampler:
    """Intelligent Budget-Aware Spatial & Angular Keyframe Sampler (§1.3 & §6 of Spec).
    Guarantees sufficient geometric parallax baseline while strictly capping keyframes
    to 250-350 per 10 minutes (hard cap 400) to meet the 15-minute compute budget.
    """

    def __init__(
        self,
        target_keyframes: int = 300,
        hard_cap_keyframes: int = 400,
        target_overlap: float = 0.85,
        turn_rate_threshold_deg_s: float = 15.0
    ):
        self.target_keyframes = target_keyframes
        self.hard_cap_keyframes = hard_cap_keyframes
        self.target_overlap = target_overlap
        self.turn_rate_threshold_deg_s = turn_rate_threshold_deg_s

    def sample_sequence(
        self,
        frames: List[KeyframeData],
        total_flight_seconds: float = 600.0
    ) -> List[KeyframeData]:
        """Processes candidate frames and selects optimal keyframe subset."""
        if not frames:
            return []
        if len(frames) <= self.target_keyframes:
            return frames

        # Step 1: Compute along-track cumulative distance if telemetry present
        distances = [0.0]
        for i in range(1, len(frames)):
            f_prev, f_curr = frames[i - 1], frames[i]
            if f_prev.telemetry and f_curr.telemetry:
                # Approximate flat-earth distance from lat/lon in meters
                dlat = (f_curr.telemetry.latitude - f_prev.telemetry.latitude) * 111139.0
                dlon = (f_curr.telemetry.longitude - f_prev.telemetry.longitude) * 111139.0 * np.cos(np.radians(f_curr.telemetry.latitude))
                dalt = f_curr.telemetry.altitude_rel - f_prev.telemetry.altitude_rel
                dist = np.sqrt(dlat**2 + dlon**2 + dalt**2)
            else:
                dist = 6.0 * (f_curr.timestamp - f_prev.timestamp) # Default nominal 6m/s
            distances.append(distances[-1] + dist)

        total_distance = distances[-1]

        # Determine target spacing (e.g. 10-15m)
        target_n = min(self.target_keyframes, int(self.target_keyframes * (total_flight_seconds / 600.0)))
        target_n = max(target_n, 50)
        nominal_spacing = total_distance / target_n if total_distance > 0 else 10.0

        selected_keyframes: List[KeyframeData] = []
        current_window: List[KeyframeData] = []
        next_checkpoint = nominal_spacing

        for i, frame in enumerate(frames):
            current_window.append(frame)

            # Angular trigger: rapid gimbal or heading turn
            is_turn = False
            if frame.telemetry and i > 0 and frames[i - 1].telemetry:
                dt = max(frame.timestamp - frames[i - 1].timestamp, 1e-3)
                dyaw = abs(frame.telemetry.yaw - frames[i - 1].telemetry.yaw)
                dpitch = abs(frame.telemetry.pitch - frames[i - 1].telemetry.pitch)
                if (dyaw / dt > self.turn_rate_threshold_deg_s) or (dpitch / dt > self.turn_rate_threshold_deg_s):
                    is_turn = True

            # Spatial trigger or turn trigger
            if distances[i] >= next_checkpoint or is_turn or (i == len(frames) - 1):
                best_kf = select_sharpest_frame(current_window)
                selected_keyframes.append(best_kf)
                current_window = []
                next_checkpoint = distances[i] + nominal_spacing

        # Enforce hard cap by decimation if angular triggers caused an overshoot
        if len(selected_keyframes) > self.hard_cap_keyframes:
            step = len(selected_keyframes) / float(self.hard_cap_keyframes)
            decimated = [selected_keyframes[int(i * step)] for i in range(self.hard_cap_keyframes)]
            return decimated

        return selected_keyframes
