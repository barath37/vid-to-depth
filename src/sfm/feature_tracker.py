from typing import List, Optional, Tuple
import cv2
import numpy as np


class SequentialFeatureTracker:
    """Detects and tracks features across keyframe sequences (§2.1 & §7 of Spec).
    Uses sequential neighbor matching to avoid O(N^2) complexity.
    """

    def __init__(
        self,
        detector_type: str = "ORB",
        n_features: int = 2000,
        matching_window: int = 4,
        ratio_threshold: float = 0.80
    ):
        self.detector_type = detector_type.upper()
        self.n_features = n_features
        self.matching_window = matching_window
        self.ratio_threshold = ratio_threshold

        if self.detector_type == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=n_features)
            self.norm_type = cv2.NORM_L2
        else:
            self.detector = cv2.ORB_create(nfeatures=n_features)
            self.norm_type = cv2.NORM_HAMMING

        self.matcher = cv2.BFMatcher(self.norm_type, crossCheck=False)

    def detect_and_compute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Extracts keypoints and descriptors from image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        kps, descs = self.detector.detectAndCompute(gray, mask=mask)
        return kps, descs

    def match_pairs(
        self,
        feats1: Tuple[List[cv2.KeyPoint], Optional[np.ndarray]],
        feats2: Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]
    ) -> List[Tuple[int, int]]:
        """Matches features between two frames using Lowe's ratio test and epipolar RANSAC.
        Returns list of (kpt1_idx, kpt2_idx) correspondences.
        """
        kps1, descs1 = feats1
        kps2, descs2 = feats2

        if descs1 is None or descs2 is None or len(descs1) < 8 or len(descs2) < 8:
            return []

        knn_matches = self.matcher.knnMatch(descs1, descs2, k=2)
        good_matches = []
        for m_pair in knn_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append((m.queryIdx, m.trainIdx))

        if len(good_matches) < 8:
            return good_matches

        # Epipolar RANSAC check via Fundamental Matrix
        pts1 = np.float32([kps1[i].pt for i, _ in good_matches])
        pts2 = np.float32([kps2[j].pt for _, j in good_matches])

        F, inlier_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
        if inlier_mask is not None:
            inliers = [good_matches[k] for k, inl in enumerate(inlier_mask.ravel()) if inl == 1]
            return inliers

        return good_matches
