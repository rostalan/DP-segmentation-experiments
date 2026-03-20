"""IoU-based greedy tracker for temporal consistency across video frames."""

from __future__ import annotations
import numpy as np
from .detection import Detection, Track, compute_iou


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5,
                 min_hits: int = 1, match_class: bool = True):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.match_class = match_class
        self.tracks: list[Track] = []
        self.next_track_id = 1

    @classmethod
    def from_config(cls, config: dict) -> "SimpleTracker":
        tcfg = config.get("tracking", {})
        return cls(
            iou_threshold=tcfg.get("iou_threshold", 0.3),
            max_age=tcfg.get("max_age", 5),
            min_hits=tcfg.get("min_hits", 1),
        )

    def _iou_matrix(self, detections: list[Detection]) -> np.ndarray:
        n_tracks, n_dets = len(self.tracks), len(detections)
        if n_tracks == 0 or n_dets == 0:
            return np.zeros((n_tracks, n_dets))
        mat = np.zeros((n_tracks, n_dets))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                if self.match_class and track.class_id != det.class_id:
                    continue
                mat[i, j] = compute_iou(track.bbox, det.bbox)
        return mat

    def update(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            for t in self.tracks:
                t.misses += 1
                t.age += 1
            self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
            return []

        mat = self._iou_matrix(detections)
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        while mat.size > 0:
            if mat.max() < self.iou_threshold:
                break
            ti, di = np.unravel_index(mat.argmax(), mat.shape)
            t = self.tracks[ti]
            d = detections[di]
            t.bbox, t.mask, t.confidence = d.bbox, d.mask, d.confidence
            t.hits += 1
            t.misses = 0
            t.age += 1
            d.track_id = t.track_id
            matched_tracks.add(ti)
            matched_dets.add(di)
            mat[ti, :] = 0
            mat[:, di] = 0

        for ti in range(len(self.tracks)):
            if ti not in matched_tracks:
                self.tracks[ti].misses += 1
                self.tracks[ti].age += 1

        for di, det in enumerate(detections):
            if di not in matched_dets:
                new = Track(
                    track_id=self.next_track_id,
                    bbox=det.bbox, class_id=det.class_id,
                    mask=det.mask, confidence=det.confidence,
                    class_name=det.class_name,
                )
                self.tracks.append(new)
                det.track_id = self.next_track_id
                self.next_track_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]

        out: list[Detection] = []
        for t in self.tracks:
            if t.hits >= self.min_hits:
                adj_conf = t.confidence * (0.9 ** t.misses)
                out.append(Detection(
                    bbox=t.bbox, confidence=adj_conf, mask=t.mask,
                    class_id=t.class_id, class_name=t.class_name,
                    instance_id=t.track_id, track_id=t.track_id,
                ))
        return out

    def reset(self):
        self.tracks.clear()
        self.next_track_id = 1
