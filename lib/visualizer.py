"""Shared overlay / panoptic / instance mask visualisation."""

from __future__ import annotations
import cv2
import numpy as np
from .detection import Detection


class Visualizer:
    """Configurable segmentation visualiser used across all pipelines."""

    def __init__(self, colors: list, alpha: float = 0.6,
                 show_labels: bool = True, show_confidence: bool = False,
                 show_boxes: bool = False, show_track_id: bool = False):
        self.colors = [tuple(c) for c in colors]
        self.alpha = alpha
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self.show_boxes = show_boxes
        self.show_track_id = show_track_id

    @classmethod
    def from_config(cls, config: dict, colors: list) -> "Visualizer":
        out = config.get("output", {})
        return cls(
            colors=colors,
            alpha=out.get("overlay_alpha", 0.6),
            show_labels=out.get("show_labels", True),
            show_confidence=out.get("show_confidence", False),
            show_boxes=out.get("show_boxes", False),
            show_track_id=out.get("show_track_id", False),
        )

    def _color_for(self, det: Detection) -> tuple[int, int, int]:
        return self.colors[det.class_id % len(self.colors)]

    def _instance_color(self, det: Detection) -> tuple[int, int, int]:
        base = self._color_for(det)
        offset = (det.instance_id * 37) % 60 - 30
        return tuple(max(0, min(255, c + offset)) for c in base)

    # -- overlays --------------------------------------------------------

    def create_overlay(self, image: np.ndarray,
                       detections: list[Detection]) -> np.ndarray:
        mask_layer = image.copy()
        for det in detections:
            mask_layer[det.mask.astype(bool)] = self._instance_color(det)

        result = cv2.addWeighted(mask_layer, self.alpha, image, 1 - self.alpha, 0)

        for det in detections:
            color = self._instance_color(det)

            if self.show_boxes:
                x1, y1, x2, y2 = det.bbox.astype(int)
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            label_parts: list[str] = []
            if self.show_track_id and det.track_id != -1:
                label_parts.append(f"#{det.track_id}")
            if self.show_labels and det.class_name:
                label_parts.append(det.class_name)
            if self.show_confidence:
                label_parts.append(f"{det.confidence:.2f}")

            if label_parts:
                label = " ".join(label_parts)
                x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(result, (x1, y1 - th - 6),
                              (x1 + tw + 4, y1), color, -1)
                cv2.putText(result, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

        return result

    # -- masks -----------------------------------------------------------

    def create_panoptic_mask(self, shape: tuple,
                             detections: list[Detection]) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        for det in sorted(detections, key=lambda d: d.mask.sum(), reverse=True):
            mask[det.mask.astype(bool)] = self._instance_color(det)
        return mask

    def create_instance_mask(self, shape: tuple,
                             detections: list[Detection]) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        n = len(detections)
        for i, det in enumerate(
                sorted(detections, key=lambda d: d.mask.sum(), reverse=True)):
            val = int((i + 1) * 255 / (n + 1)) if n > 0 else i + 1
            mask[det.mask > 0] = val
        return mask
