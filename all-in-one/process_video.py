#!/usr/bin/env python3
"""
Assembly Recording Pipeline
============================
Records assembly steps by detecting scene stillness (frame differencing) and
hand withdrawal (MediaPipe), then running YOLO detection at confirmed step
transitions.  Outputs keyframes, bounding boxes, segmentation masks, and
detection metadata per step.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import Counter
import time
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.hand import HAND_CONNECTIONS
from lib.detection import compute_iou
from lib.config import load_config, merge_global_config, get_colors


# ---------------------------------------------------------------------------
# HandPresenceDetector
# ---------------------------------------------------------------------------

class HandPresenceDetector:
    """Lightweight hand presence detection using MediaPipe HandLandmarker.

    Runs detection on a downscaled frame every ``detect_interval`` frames.
    Reports hands as absent only after ``absent_checks_required`` consecutive
    detection cycles with no hands found.
    """

    def __init__(self, detect_interval: int = 3, absent_checks_required: int = 5,
                 min_confidence: float = 0.5,
                 model_path: str = "../models/hand_landmarker.task",
                 resolution: Tuple[int, int] = (640, 480)):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_tracking_confidence=0.3,
            running_mode=vision.RunningMode.VIDEO)
        self._detector = vision.HandLandmarker.create_from_options(options)
        self.detect_interval = detect_interval
        self.absent_checks_required = absent_checks_required
        self.resolution = resolution
        self._frame_count: int = 0
        self._consecutive_absent: int = 0
        self._hands_visible: bool = False
        self._timestamp_ms: int = 0
        self._landmarks: list = []

    def update(self, frame: np.ndarray) -> None:
        self._frame_count += 1
        self._timestamp_ms += 33
        if self._frame_count % self.detect_interval != 0:
            return
        small = cv2.resize(frame, self.resolution)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect_for_video(mp_image, self._timestamp_ms)
        if result.hand_landmarks:
            self._consecutive_absent = 0
            self._hands_visible = True
            self._landmarks = result.hand_landmarks
        else:
            self._consecutive_absent += 1
            self._landmarks = []
            if self._consecutive_absent >= self.absent_checks_required:
                self._hands_visible = False

    def draw(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        for hand_lm in self._landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(image, pts[a], pts[b], (0, 255, 0), 2)
            for px, py in pts:
                cv2.circle(image, (px, py), 4, (0, 0, 255), -1)
        return image

    @property
    def hands_present(self) -> bool:
        return self._hands_visible

    def close(self):
        self._detector.close()


# ---------------------------------------------------------------------------
# YOLODetector
# ---------------------------------------------------------------------------

class YOLODetector:
    """Object detection and segmentation using a fine-tuned YOLO model."""

    def __init__(self, model_path: str, confidence: float = 0.5,
                 iou_threshold: float = 0.5):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self._warmed_up = False

    def detect(self, frame: np.ndarray
               ) -> Tuple[List[np.ndarray], List[str], List[float], Optional[List[np.ndarray]]]:
        """Run detection on a single frame.

        Returns (bboxes, labels, confidences, masks):
            bboxes       – list of [x1, y1, x2, y2] float32 arrays
            labels       – list of class-name strings
            confidences  – list of detection confidence floats
            masks        – list of HxW uint8 binary masks (or None)
        """
        if not self._warmed_up:
            self.model(frame, conf=self.confidence, iou=self.iou_threshold,
                       verbose=False)
            self._warmed_up = True

        results = self.model(frame, conf=self.confidence,
                             iou=self.iou_threshold, verbose=False)
        r = results[0]

        bboxes: List[np.ndarray] = []
        labels: List[str] = []
        confidences: List[float] = []
        masks: List[np.ndarray] = []

        if len(r.boxes) > 0:
            for i in range(len(r.boxes)):
                bboxes.append(r.boxes.xyxy[i].cpu().numpy().astype(np.float32))
                labels.append(r.names[int(r.boxes.cls[i])])
                confidences.append(float(r.boxes.conf[i]))
                if r.masks is not None:
                    mask = r.masks.data[i].cpu().numpy()
                    masks.append((mask > 0.5).astype(np.uint8))

        return bboxes, labels, confidences, masks if masks else None


# ---------------------------------------------------------------------------
# Detection grouping (overlapping bbox analysis)
# ---------------------------------------------------------------------------

def group_overlapping_detections(
    bboxes: List[np.ndarray],
    labels: List[str],
    confidences: List[float],
    iou_threshold: float = 0.3,
) -> List[Dict]:
    """Group spatially overlapping detections using Union-Find on IoU.

    Returns a list of groups.  Each group is a dict:
        detections – [(bbox, label, confidence), …] sorted by confidence desc
        ambiguous  – True when the group contains multiple distinct class labels
    """
    n = len(bboxes)
    if n == 0:
        return []

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if compute_iou(bboxes[i], bboxes[j]) >= iou_threshold:
                unite(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    groups: List[Dict] = []
    for indices in clusters.values():
        dets = [(bboxes[k], labels[k], confidences[k]) for k in indices]
        dets.sort(key=lambda d: d[2], reverse=True)
        unique_labels = {d[1] for d in dets}
        groups.append({
            "detections": dets,
            "ambiguous": len(unique_labels) > 1,
        })
    return groups


# ---------------------------------------------------------------------------
# StillnessDetector
# ---------------------------------------------------------------------------

class StillnessDetector:
    """Detects scene stillness by comparing consecutive frames.

    Two thresholds:
        still_frames_required  – minimum still frames to flag the scene as
                                 settled (used together with hand-absence gate).
        still_timeout          – if still for this many frames, force a
                                 transition even when hands are present.
    """

    def __init__(self, threshold: float = 3.0, still_frames_required: int = 15,
                 still_timeout: int = 90, scale: float = 0.25):
        self.threshold = threshold
        self.still_frames_required = still_frames_required
        self.still_timeout = still_timeout
        self.scale = scale
        self.prev_gray: Optional[np.ndarray] = None
        self.consecutive_still: int = 0
        self.scene_active: bool = False
        self._last_diff: float = 0.0

    def _to_gray_small(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.scale != 1.0:
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_AREA)
        return gray

    def reset(self, frame: np.ndarray):
        """Set a new reference after a step transition."""
        self.prev_gray = self._to_gray_small(frame)
        self.consecutive_still = 0
        self.scene_active = False

    def update(self, frame: np.ndarray) -> bool:
        """Process one frame.  Returns True when scene settles after activity."""
        gray = self._to_gray_small(frame)
        if self.prev_gray is None:
            self.prev_gray = gray
            return False

        self._last_diff = float(np.mean(cv2.absdiff(gray, self.prev_gray)))
        self.prev_gray = gray

        if self._last_diff > self.threshold:
            self.consecutive_still = 0
            self.scene_active = True
            return False

        self.consecutive_still += 1
        if self.consecutive_still >= self.still_frames_required and self.scene_active:
            self.scene_active = False
            return True
        return False

    @property
    def timed_out(self) -> bool:
        """True when the scene has been still long enough to force a transition."""
        return self.consecutive_still >= self.still_timeout and not self.scene_active

    @property
    def diff_value(self) -> float:
        return self._last_diff


# ---------------------------------------------------------------------------
# StepRecorder
# ---------------------------------------------------------------------------

class StepRecorder:
    """Saves per-step keyframes, detections, diffs, and masks."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_step: int = 0
        self.steps: List[Dict] = []
        self._prev_labels: List[str] = []

    def record_step(self, frame: np.ndarray, frame_idx: int,
                    bboxes: List[np.ndarray], labels: List[str],
                    confidences: List[float],
                    masks: Optional[List[np.ndarray]] = None
                    ) -> Tuple[int, List[str], List[str]]:
        """Save one step's data. Returns (step_number, added, removed)."""
        prev_counts = Counter(self._prev_labels)
        curr_counts = Counter(labels)
        added = sorted((curr_counts - prev_counts).elements())
        removed = sorted((prev_counts - curr_counts).elements())

        self.current_step += 1
        step_dir = self.output_dir / f"step_{self.current_step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(step_dir / "keyframe.jpg"), frame)

        detections = []
        for i, (bbox, label, conf) in enumerate(zip(bboxes, labels, confidences)):
            detections.append({
                "id": i,
                "class": label,
                "bbox": [round(float(v), 1) for v in bbox[:4]],
                "confidence": round(conf, 4),
            })
            if masks is not None and i < len(masks):
                mask_dir = step_dir / "masks"
                mask_dir.mkdir(exist_ok=True)
                mask_name = f"{i}_{label.replace(' ', '_')}.png"
                cv2.imwrite(str(mask_dir / mask_name), masks[i] * 255)

        step_data = {
            "step": self.current_step,
            "detections": detections,
            "changes": {"added": added, "removed": removed},
        }
        with open(step_dir / "detections.json", "w") as f:
            json.dump(step_data, f, indent=2)

        self.steps.append({
            "step": self.current_step,
            "frame_idx": frame_idx,
            "num_objects": len(bboxes),
            "classes": labels,
            "added": added,
            "removed": removed,
        })
        self._prev_labels = list(labels)
        return self.current_step, added, removed

    def save_summary(self):
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(self.steps, f, indent=2)

        all_classes = sorted({c for s in self.steps for c in s["classes"]})
        guide = {
            "total_steps": self.current_step,
            "classes": all_classes,
            "steps": [
                {
                    "step": s["step"],
                    "frame_idx": s["frame_idx"],
                    "objects": s["classes"],
                    "added": s["added"],
                    "removed": s["removed"],
                    "keyframe": f"step_{s['step']:03d}/keyframe.jpg",
                    "detections_file": f"step_{s['step']:03d}/detections.json",
                }
                for s in self.steps
            ],
        }
        with open(self.output_dir / "guide.json", "w") as f:
            json.dump(guide, f, indent=2)


# ---------------------------------------------------------------------------
# VideoProcessor
# ---------------------------------------------------------------------------

class VideoProcessor:
    """Simplified recording pipeline: YOLO at transitions, frame diff for stillness."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        merge_global_config(self.config, Path(__file__).resolve().parent)

        yolo_cfg = self.config.get('yolo_detection', {})
        self.detector = YOLODetector(
            model_path=yolo_cfg.get('model_path', '../models/ovoe-v4-weights.pt'),
            confidence=yolo_cfg.get('confidence', 0.5),
            iou_threshold=yolo_cfg.get('iou_threshold', 0.5))

        still_cfg = self.config.get('stillness', {})
        self.stillness = StillnessDetector(
            threshold=still_cfg.get('threshold', 3.0),
            still_frames_required=still_cfg.get('still_frames_required', 15),
            still_timeout=still_cfg.get('still_timeout', 90),
            scale=still_cfg.get('scale', 0.25))

        hand_cfg = self.config.get('hands', {})
        self.hand_detector = HandPresenceDetector(
            detect_interval=hand_cfg.get('detect_interval', 3),
            absent_checks_required=hand_cfg.get('absent_checks_required', 5),
            min_confidence=hand_cfg.get('min_confidence', 0.5),
            model_path=hand_cfg.get('model_path', '../models/hand_landmarker.task'),
            resolution=tuple(hand_cfg.get('resolution', [640, 480])))

        self.recorder = StepRecorder(
            output_dir=Path(self.config['paths']['output_dir']))

        display_cfg = self.config.get('display', {})
        self.display_enabled = display_cfg.get('enabled', False)
        self.display_scale = display_cfg.get('scale', 0.8)

        self.current_bboxes: List[np.ndarray] = []
        self.current_labels: List[str] = []
        self.current_confidences: List[float] = []
        self._pending_transition: bool = False
        self.overlap_iou: float = yolo_cfg.get('overlap_iou_threshold', 0.3)

        self.colors = [tuple(c) for c in get_colors(self.config)]

    def _detect_and_record(self, frame: np.ndarray, frame_idx: int) -> bool:
        """Run YOLO, compare to previous state, record if changed. Returns True if step recorded."""
        bboxes, labels, confidences, masks = self.detector.detect(frame)

        prev_counts = Counter(self.current_labels)
        new_counts = Counter(labels)

        if prev_counts == new_counts and self.recorder.current_step > 0:
            print(f"  Frame {frame_idx}: scene settled, no change in objects — skipping")
            return False

        step_num, added, removed = self.recorder.record_step(
            frame, frame_idx, bboxes, labels, confidences, masks)
        self.current_bboxes = bboxes
        self.current_labels = labels
        self.current_confidences = confidences

        print(f"\n>>> Step {step_num} recorded at frame {frame_idx}")
        print(f"    Objects ({len(bboxes)}):")
        for b, lbl, c in zip(bboxes, labels, confidences):
            print(f"      {lbl} ({c:.2f}): "
                  f"[{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}]")
        if added:
            print(f"    + Added: {', '.join(added)}")
        if removed:
            print(f"    - Removed: {', '.join(removed)}")
        return True

    def _create_display_frame(self, frame: np.ndarray, frame_idx: int,
                               fps_actual: float) -> np.ndarray:
        display = frame.copy()
        self.hand_detector.draw(display)

        CYAN = (255, 200, 0)
        groups = group_overlapping_detections(
            self.current_bboxes, self.current_labels,
            self.current_confidences, self.overlap_iou)

        color_idx = 0
        for group in groups:
            dets = group["detections"]
            if group["ambiguous"]:
                parts = [f"{lbl} {int(conf * 100)}%"
                         for _, lbl, conf in dets]
                combined = " / ".join(parts)
                for bbox, _, _ in dets:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    cv2.rectangle(display, (x1, y1), (x2, y2), CYAN, 2)
                px1, py1 = int(dets[0][0][0]), int(dets[0][0][1])
                (tw, th), _ = cv2.getTextSize(
                    combined, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(display, (px1, py1 - th - 6),
                              (px1 + tw + 4, py1), CYAN, -1)
                cv2.putText(display, combined, (px1 + 2, py1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                color_idx += len(dets)
            else:
                for bbox, label, conf in dets:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    color = self.colors[color_idx % len(self.colors)]
                    conf_label = f"{label} {int(conf * 100)}%"
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(
                        conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(display, (x1, y1 - th - 6),
                                  (x1 + tw + 4, y1), color, -1)
                    cv2.putText(display, conf_label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)
                    color_idx += 1

        active = "ACTIVE" if self.stillness.scene_active else "still"
        hand_s = "H" if self.hand_detector.hands_present else "h"
        pend = " PEND" if self._pending_transition else ""
        status = (f"F:{frame_idx} | Obj:{len(self.current_bboxes)} | "
                  f"Step:{self.recorder.current_step} | "
                  f"Diff:{self.stillness.diff_value:.1f} [{active}] | "
                  f"{hand_s}{pend} | {fps_actual:.1f}fps")
        cv2.rectangle(display, (0, 0), (len(status) * 9 + 20, 28), (0, 0, 0), -1)
        cv2.putText(display, status, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if self.display_scale != 1.0:
            h, w = display.shape[:2]
            display = cv2.resize(display, (int(w * self.display_scale),
                                           int(h * self.display_scale)))
        return display

    def process(self):
        """Main loop: decode frames, detect stillness, run YOLO at transitions."""
        input_path = Path(self.config['paths']['input_video'])
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        sampling_cfg = self.config.get('sampling', {})
        start_frame = sampling_cfg.get('start_frame', 0)
        max_frames = sampling_cfg.get('max_frames')

        print(f"\nInput: {input_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, "
              f"Total frames: {total_frames}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_frame = cap.read()
        if ret:
            self._detect_and_record(first_frame, start_frame)
            self.stillness.reset(first_frame)

        start_time = time.time()
        processed_count = 0
        end_frame = total_frames
        if max_frames:
            end_frame = min(start_frame + max_frames, total_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        fps_actual = 0.0

        try:
            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                scene_settled = self.stillness.update(frame)
                self.hand_detector.update(frame)

                if scene_settled:
                    self._pending_transition = True

                if self.stillness.scene_active:
                    self._pending_transition = False

                should_record = (
                    (self._pending_transition and not self.hand_detector.hands_present)
                    or self.stillness.timed_out
                )

                if should_record:
                    self._detect_and_record(frame, frame_idx)
                    self.stillness.reset(frame)
                    self._pending_transition = False

                if self.display_enabled:
                    display = self._create_display_frame(frame, frame_idx, fps_actual)
                    cv2.imshow("Pipeline", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nUser requested quit.")
                        break

                processed_count += 1
                elapsed = time.time() - start_time
                fps_actual = processed_count / elapsed if elapsed > 0 else 0

                if processed_count % 60 == 0 or processed_count <= 3:
                    active = "ACTIVE" if self.stillness.scene_active else "still"
                    hand_s = "H" if self.hand_detector.hands_present else "h"
                    pend = " PEND" if self._pending_transition else ""
                    print(f"Frame {frame_idx:6d} | Obj:{len(self.current_bboxes)} | "
                          f"Step:{self.recorder.current_step} | "
                          f"Diff:{self.stillness.diff_value:.1f} [{active}] | "
                          f"{hand_s}{pend} | {fps_actual:.1f}fps")

                frame_idx += 1

        finally:
            cap.release()
            self.hand_detector.close()
            self.recorder.save_summary()
            if self.display_enabled:
                cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print(f"\nProcessing complete!")
        print(f"Processed {processed_count} frames in {elapsed:.1f}s "
              f"({fps_actual:.1f} FPS)")
        print(f"Total steps: {self.recorder.current_step}")
        print(f"Output: {self.config['paths']['output_dir']}")


def main():
    parser = argparse.ArgumentParser(description="Assembly Recording Pipeline")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    processor = VideoProcessor(args.config)
    processor.process()


if __name__ == "__main__":
    main()
