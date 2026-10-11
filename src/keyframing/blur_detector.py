from typing import List
import cv2
import numpy as np

from src.common.types import KeyframeData


def compute_laplacian_variance(image: np.ndarray) -> float:
    """Calculates Variance of Laplacian for motion blur & compression artifact rejection (§1.2 of Spec).
    Higher value indicates sharper edge contrast.
    """
    if image is None or image.size == 0:
        return 0.0

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return variance


def is_frame_sharp(image: np.ndarray, threshold: float = 80.0) -> bool:
    """Checks if frame exceeds sharpness threshold."""
    return compute_laplacian_variance(image) >= threshold


def select_sharpest_frame(
    candidates: List[KeyframeData],
    iframe_bonus_factor: float = 1.15
) -> KeyframeData:
    """Within a spatial distance window, select the sharpest frame,
    providing a bonus multiplier for video encoder I-frames (§6.2 of Spec).
    """
    if not candidates:
        raise ValueError("Candidates list cannot be empty")
    if len(candidates) == 1:
        return candidates[0]

    def score(kf: KeyframeData) -> float:
        s = kf.laplacian_var
        if kf.is_i_frame:
            s *= iframe_bonus_factor
        return s

    return max(candidates, key=score)
