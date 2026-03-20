"""Shared Detection / Track dataclasses and IoU computation."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Detection:
    bbox: np.ndarray
    confidence: float
    mask: np.ndarray
    class_id: int = 0
    class_name: str = ""
    instance_id: int = 0
    track_id: int = -1


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    class_id: int
    mask: np.ndarray
    confidence: float
    class_name: str = ""
    age: int = 0
    hits: int = 1
    misses: int = 0


def compute_iou(box_a, box_b) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (float(box_a[2]) - float(box_a[0])) * (float(box_a[3]) - float(box_a[1]))
    area_b = (float(box_b[2]) - float(box_b[0])) * (float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
