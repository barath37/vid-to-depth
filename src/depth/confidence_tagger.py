from typing import Optional, Tuple
import cv2
import numpy as np

from src.common.types import ConfidenceTag


class EpistemicConfidenceTagger:
    """Computes per-pixel epistemic confidence scores C ∈ [0, 1] (§3.4 & §9 of Spec).
    Combines:
    - Anchor Proximity (A)
    - Parallax Triangulation Angle (P)
    - Multi-View Agreement (M)
    - Temporal Consistency (T)
    Outputs:
    - Continuous confidence map in [0, 1]
    - Categorical tag map: Green (>=0.7), Yellow (0.4-0.7), Red (<0.4).
    """

    def __init__(
        self,
        anchor_decay_px: float = 30.0,
        green_threshold: float = 0.70,
        yellow_threshold: float = 0.40
    ):
        self.anchor_decay_px = anchor_decay_px
        self.green_threshold = green_threshold
        self.yellow_threshold = yellow_threshold

    def compute_confidence(
        self,
        image_shape: Tuple[int, int],
        sparse_uvs: np.ndarray,
        triangulation_angles: Optional[np.ndarray] = None,
        multiview_agreement: Optional[np.ndarray] = None,
        temporal_stability: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates per-pixel confidence map and categorical tag array.
        image_shape: (H, W)
        sparse_uvs: (N, 2) in integer image coordinates.
        Returns: (conf_map (H, W) float32, tag_map (H, W) object with ConfidenceTag)
        """
        H, W = image_shape

        # Signal 1: Anchor Proximity via Euclidean Distance Transform
        anchor_mask = np.ones((H, W), dtype=np.uint8)
        if len(sparse_uvs) > 0:
            u_c = np.clip(sparse_uvs[:, 0].astype(int), 0, W - 1)
            v_c = np.clip(sparse_uvs[:, 1].astype(int), 0, H - 1)
            anchor_mask[v_c, u_c] = 0 # 0 at anchor locations for cv2.distanceTransform

        dist_transform = cv2.distanceTransform(anchor_mask, cv2.DIST_L2, 5)
        # Closer to anchor -> score closer to 1.0
        A = np.exp(-dist_transform / self.anchor_decay_px).astype(np.float32)

        # Signal 2: Parallax Triangulation Angle (saturates at 10 deg)
        if triangulation_angles is not None and len(triangulation_angles) > 0:
            avg_angle = float(np.mean(triangulation_angles))
            P_val = min(avg_angle / 10.0, 1.0)
            P = np.full((H, W), P_val, dtype=np.float32)
        else:
            P = np.full((H, W), 0.5, dtype=np.float32)

        # Signal 3: Multi-view Agreement
        if multiview_agreement is not None:
            M = multiview_agreement
        else:
            M = np.full((H, W), 0.8, dtype=np.float32)

        # Signal 4: Temporal Stability
        if temporal_stability is not None:
            T = temporal_stability
        else:
            T = np.full((H, W), 0.8, dtype=np.float32)

        # Weighted combination C = 0.25*A + 0.25*P + 0.25*M + 0.25*T
        C = np.clip(0.40 * A + 0.20 * P + 0.20 * M + 0.20 * T, 0.0, 1.0).astype(np.float32)

        # Categorical Tag Map
        tag_map = np.empty((H, W), dtype=object)
        tag_map[C >= self.green_threshold] = ConfidenceTag.GREEN
        tag_map[(C >= self.yellow_threshold) & (C < self.green_threshold)] = ConfidenceTag.YELLOW
        tag_map[C < self.yellow_threshold] = ConfidenceTag.RED

        return C, tag_map
