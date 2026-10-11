from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import numpy as np


class ConfidenceTag(str, Enum):
    GREEN = "green"    # Directly measured / BA-verified (C >= 0.7)
    YELLOW = "yellow"  # Partially supported (0.4 <= C < 0.7)
    RED = "red"        # AI-inferred / occluded (C < 0.4)


@dataclass
class TelemetrySample:
    timestamp: float          # In seconds
    latitude: float           # WGS84 decimal degrees
    longitude: float          # WGS84 decimal degrees
    altitude_msl: float       # Ellipsoidal / MSL altitude in meters
    altitude_rel: float       # Height above takeoff in meters
    pitch: float = 0.0        # Gimbal/drone pitch in degrees
    roll: float = 0.0         # Gimbal/drone roll in degrees
    yaw: float = 0.0          # Gimbal/drone heading in degrees
    speed: float = 0.0        # Ground speed in m/s
    baro_altitude: Optional[float] = None
    imu_accel: Optional[np.ndarray] = None  # [ax, ay, az] in m/s^2
    imu_gyro: Optional[np.ndarray] = None   # [gx, gy, gz] in rad/s


@dataclass
class KeyframeData:
    frame_idx: int
    timestamp: float
    image: np.ndarray         # HxWx3 uint8 BGR
    laplacian_var: float
    is_i_frame: bool = False
    mask: Optional[np.ndarray] = None       # HxW uint8 mask (0 = dynamic object to ignore)
    telemetry: Optional[TelemetrySample] = None


@dataclass
class CameraPose:
    frame_idx: int
    R: np.ndarray             # 3x3 rotation matrix (world to camera)
    t: np.ndarray             # 3x1 translation vector (world to camera)
    timestamp: float = 0.0

    @property
    def center(self) -> np.ndarray:
        """Camera optical center in world Euclidean coordinates: c = -R^T @ t."""
        return -self.R.T @ self.t


@dataclass
class PointCloudModel:
    points: np.ndarray                       # Nx3 float32
    colors: np.ndarray                       # Nx3 uint8 RGB
    confidence: np.ndarray                   # N float32 in [0, 1]
    tags: List[ConfidenceTag] = field(default_factory=list)
    normals: Optional[np.ndarray] = None     # Nx3 float32 view-oriented normals
