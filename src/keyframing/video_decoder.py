from typing import Generator, Tuple
import cv2
import numpy as np

from src.keyframing.blur_detector import compute_laplacian_variance


class StreamVideoDecoder:
    """Stream-based dynamic video decoding. Does not load the full video into memory (§1.1 of Spec)."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray, float], None, None]:
        """Yields (frame_idx, timestamp_seconds, image_bgr, laplacian_variance)."""
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            ts = idx / self.fps
            var_l = compute_laplacian_variance(frame)
            yield idx, ts, frame, var_l
            idx += 1

    def release(self):
        if self.cap is not None:
            self.cap.release()
