from typing import Optional
import numpy as np


class DynamicObjectMasker:
    """Masks dynamic objects (vehicles, humans, animals) in keyframes
    before feature tracking and pseudo-LiDAR unprojection (§1.5 of Spec).
    Returns binary mask: 255 = static geometry (keep), 0 = dynamic entity (ignore).
    """

    def __init__(self, enable_detector: bool = False, model_name: str = "yolov8n.pt"):
        self.enable_detector = enable_detector
        self.model_name = model_name
        self.model = None

        if self.enable_detector:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_name)
            except Exception:
                # Graceful fallback if ultralytics is not installed
                self.model = None

    def get_mask(self, image: np.ndarray) -> np.ndarray:
        """Returns 2D uint8 mask (H, W) where 255 is static background and 0 is moving object."""
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        if not self.enable_detector or self.model is None:
            return mask

        try:
            results = self.model(image, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    # COCO classes for dynamic objects: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck, 15-23=animals
                    if cls_id in [0, 2, 3, 5, 7, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(w, xyxy[2]), min(h, xyxy[3])
                        mask[y1:y2, x1:x2] = 0
        except Exception:
            pass

        return mask
