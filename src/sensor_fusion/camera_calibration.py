from dataclasses import dataclass, field
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    """Manages drone camera pinhole intrinsics matrix K and lens distortion coefficients."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @classmethod
    def from_sensor_specs(
        cls,
        image_width: int,
        image_height: int,
        sensor_width_mm: float,
        sensor_height_mm: float,
        focal_length_mm: float,
        dist_coeffs: Optional[np.ndarray] = None
    ) -> "CameraIntrinsics":
        """Compute fx, fy, cx, cy from manufacturer drone spec sheet."""
        fx = (focal_length_mm / sensor_width_mm) * image_width
        fy = (focal_length_mm / sensor_height_mm) * image_height
        cx = image_width / 2.0
        cy = image_height / 2.0

        if dist_coeffs is None:
            dist_coeffs = np.zeros(5, dtype=np.float64)

        return cls(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=image_width,
            height=image_height,
            dist_coeffs=dist_coeffs
        )

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Undistorts an input frame using OpenCV undistort."""
        if np.all(self.dist_coeffs == 0):
            return image
        return cv2.undistort(image, self.intrinsic_matrix, self.dist_coeffs)

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Undistorts 2D image coordinates (N, 2)."""
        if len(points) == 0:
            return points
        pts_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv2.undistortPoints(
            pts_reshaped, self.intrinsic_matrix, self.dist_coeffs, P=self.intrinsic_matrix
        )
        return undistorted.reshape(-1, 2)
