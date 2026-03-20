#!/usr/bin/env python3
"""
Assembly Playback Prototype
============================
Loads a recorded guide (guide.json) and runs YOLO detection on a live
webcam or video feed, matching the scene to recorded steps and showing
AR-style overlays to guide the user through the assembly procedure.

Controls:
    n / Right  -- next step
    p / Left   -- previous step
    r          -- reset to step 1
    q / Esc    -- quit
"""

import cv2
import numpy as np
import yaml
import json
import argparse
import time
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Optional

from process_video import YOLODetector, group_overlapping_detections
from gesture_control import GestureController, GESTURE_ACTION_MAP


# ---------------------------------------------------------------------------
# GuidePlayer
# ---------------------------------------------------------------------------

class GuidePlayer:
    """Loads a guide.json and tracks playback state."""

    def __init__(self, guide_path: str):
        guide_dir = Path(guide_path).parent
        with open(guide_path) as f:
            data = json.load(f)

        self.total_steps: int = data["total_steps"]
        self.num_actions: int = max(0, self.total_steps - 1)
        self.classes: List[str] = data["classes"]
        self.steps: List[Dict] = data["steps"]
        self._step_idx: int = 1 if self.total_steps > 1 else 0

        for s in self.steps:
            kf_path = guide_dir / s["keyframe"]
            if kf_path.exists():
                img = cv2.imread(str(kf_path))
                h, w = img.shape[:2]
                thumb_w = 240
                thumb_h = int(h * thumb_w / w)
                s["_keyframe_thumb"] = cv2.resize(img, (thumb_w, thumb_h))
            else:
                s["_keyframe_thumb"] = None

    @property
    def step_idx(self) -> int:
        return self._step_idx

    @property
    def current(self) -> Dict:
        return self.steps[self._step_idx]

    @property
    def finished(self) -> bool:
        return self._step_idx >= self.total_steps

    def advance(self) -> bool:
        if self._step_idx < self.total_steps - 1:
            self._step_idx += 1
            return True
        return False

    def go_back(self) -> bool:
        if self._step_idx > 0:
            self._step_idx -= 1
            return True
        return False

    @property
    def action_num(self) -> int:
        """User-facing step number (1-based, baseline step excluded)."""
        return self._step_idx

    def reset(self):
        self._step_idx = 1 if self.total_steps > 1 else 0

    def match(self, detected_labels: List[str]
              ) -> Tuple[List[str], List[str], List[str], bool]:
        """Compare detected labels against current step's expected objects.

        Returns (matched, missing, extra, complete).
        """
        expected = Counter(self.current["objects"])
        detected = Counter(detected_labels)
        matched = sorted((expected & detected).elements())
        missing = sorted((expected - detected).elements())
        extra = sorted((detected - expected).elements())
        return matched, missing, extra, len(missing) == 0


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

COLOR_GREEN = (0, 200, 0)
COLOR_YELLOW = (0, 220, 255)
COLOR_ORANGE = (0, 140, 255)
COLOR_RED = (0, 0, 220)
COLOR_GRAY = (120, 120, 120)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_CYAN = (255, 200, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlays(frame: np.ndarray, guide: GuidePlayer,
                  bboxes: List[np.ndarray], labels: List[str],
                  confidences: List[float],
                  step_complete: bool, auto_progress: float,
                  fps: float, overlap_iou: float = 0.3) -> np.ndarray:
    """Draw all playback overlays onto the frame."""
    h, w = frame.shape[:2]
    step = guide.current
    expected_set = Counter(step["objects"])

    groups = group_overlapping_detections(bboxes, labels, confidences,
                                          overlap_iou)
    remaining_expected = Counter(expected_set)

    for group in groups:
        dets = group["detections"]

        if group["ambiguous"]:
            parts = [f"{lbl} {int(conf * 100)}%"
                     for _, lbl, conf in dets]
            combined = " / ".join(parts)

            for bbox, _, _ in dets:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_CYAN, 2)

            px1, py1 = int(dets[0][0][0]), int(dets[0][0][1])
            (tw, th), _ = cv2.getTextSize(combined, FONT, 0.45, 1)
            cv2.rectangle(frame, (px1, py1 - th - 8),
                          (px1 + tw + 6, py1), COLOR_CYAN, -1)
            cv2.putText(frame, combined, (px1 + 3, py1 - 4),
                        FONT, 0.45, COLOR_BLACK, 1)
        else:
            for bbox, label, conf in dets:
                if remaining_expected[label] > 0:
                    color = COLOR_GREEN
                    remaining_expected[label] -= 1
                else:
                    color = COLOR_ORANGE
                x1, y1, x2, y2 = map(int, bbox[:4])
                conf_label = f"{label} {int(conf * 100)}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(conf_label, FONT, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8),
                              (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, conf_label, (x1 + 3, y1 - 4),
                            FONT, 0.5, COLOR_WHITE, 1)

    # --- Step instruction panel (top-left) ---
    added_str = ", ".join(step["added"]) if step["added"] else "verify setup"
    instruction = f"Step {guide.action_num}/{guide.num_actions}: Add {added_str}"

    panel_h = 70
    cv2.rectangle(frame, (0, 0), (w, panel_h), COLOR_BLACK, -1)
    cv2.putText(frame, instruction, (12, 28), FONT, 0.7, COLOR_WHITE, 2)

    _, missing, _, _ = guide.match(labels)
    if missing:
        miss_str = f"Missing: {', '.join(missing)}"
        cv2.putText(frame, miss_str, (12, 55), FONT, 0.55, COLOR_YELLOW, 1)
    elif step_complete:
        status_text = "Step complete"
        if auto_progress > 0:
            status_text += f" (advancing {auto_progress:.0%})"
        cv2.putText(frame, status_text, (12, 55), FONT, 0.55, COLOR_GREEN, 1)

    # --- FPS ---
    cv2.putText(frame, f"{fps:.0f} fps", (w - 90, 28), FONT, 0.55, COLOR_GRAY, 1)

    # --- Reference keyframe thumbnail (top-right, below panel) ---
    thumb = step.get("_keyframe_thumb")
    if thumb is not None:
        th_h, th_w = thumb.shape[:2]
        y_off = panel_h + 5
        x_off = w - th_w - 5
        if y_off + th_h < h and x_off > 0:
            cv2.rectangle(frame, (x_off - 2, y_off - 2),
                          (x_off + th_w + 2, y_off + th_h + 2), COLOR_WHITE, 1)
            frame[y_off:y_off + th_h, x_off:x_off + th_w] = thumb

    # --- Progress bar (bottom, action steps only) ---
    bar_h = 18
    bar_y = h - bar_h
    cv2.rectangle(frame, (0, bar_y), (w, h), COLOR_BLACK, -1)
    n = guide.num_actions
    if n > 0:
        seg_w = w / n
        for i in range(n):
            action_idx = i + 1
            x_start = int(i * seg_w)
            x_end = int((i + 1) * seg_w) - 1
            if action_idx < guide.step_idx:
                color = COLOR_GREEN
            elif action_idx == guide.step_idx:
                color = COLOR_YELLOW if not step_complete else COLOR_GREEN
            else:
                color = COLOR_GRAY
            cv2.rectangle(frame, (x_start + 1, bar_y + 2),
                          (x_end, h - 2), color, -1)
            cv2.putText(frame, str(action_idx),
                        (x_start + int(seg_w / 2) - 4, h - 4),
                        FONT, 0.4, COLOR_BLACK, 1)

    # --- Border glow for step status ---
    border = 4
    if step_complete:
        border_color = COLOR_GREEN
    else:
        border_color = COLOR_YELLOW
    cv2.rectangle(frame, (0, panel_h), (w - 1, bar_y), border_color, border)

    return frame


# ---------------------------------------------------------------------------
# Main playback loop
# ---------------------------------------------------------------------------

def run_playback(config_path: str = "config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    pb_cfg = config.get("playback", {})
    guide_path = pb_cfg.get("guide_path", "output/steps/guide.json")
    input_source = pb_cfg.get("input", 0)
    auto_advance_frames = pb_cfg.get("auto_advance_frames", 15)
    detect_interval = pb_cfg.get("detection_interval", 3)
    display_scale = config.get("display", {}).get("scale", 0.8)

    yolo_cfg = config.get("yolo_detection", {})
    detector = YOLODetector(
        model_path=yolo_cfg.get("model_path", "../models/oboe-train-v1-weights.pt"),
        confidence=yolo_cfg.get("confidence", 0.5),
        iou_threshold=yolo_cfg.get("iou_threshold", 0.5))
    overlap_iou = yolo_cfg.get("overlap_iou_threshold", 0.3)

    gesture_cfg = config.get("gestures", {})
    gesture_enabled = gesture_cfg.get("enabled", True)
    gesture = None
    if gesture_enabled:
        gesture = GestureController(
            model_path=gesture_cfg.get("model_path", "../models/gesture_recognizer.task"),
            detect_interval=gesture_cfg.get("detect_interval", 2),
            hold_detections=gesture_cfg.get("hold_detections", 5),
            cooldown_detections=gesture_cfg.get("cooldown_detections", 8),
            min_confidence=gesture_cfg.get("min_confidence", 0.55),
            resolution=tuple(gesture_cfg.get("resolution", [640, 480])),
        )

    guide = GuidePlayer(guide_path)
    print(f"Loaded guide: {guide.total_steps} steps, "
          f"{len(guide.classes)} classes")
    for s in guide.steps:
        print(f"  Step {s['step']}: {s['objects']}  "
              f"(+{s['added']}, -{s['removed']})")

    if isinstance(input_source, int) or str(input_source).isdigit():
        cap = cv2.VideoCapture(int(input_source))
    else:
        cap = cv2.VideoCapture(str(input_source))
    if not cap.isOpened():
        raise ValueError(f"Could not open input: {input_source}")

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Input: {input_source} ({vid_w}x{vid_h} @ {vid_fps:.0f}fps)")

    frame_count = 0
    consecutive_complete = 0
    cached_bboxes: List[np.ndarray] = []
    cached_labels: List[str] = []
    cached_confs: List[float] = []
    fps_actual = 0.0
    t_start = time.time()

    print("\nPlayback started. Press 'q' to quit, 'n'/'p' to navigate steps.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(input_source, int):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if frame_count % detect_interval == 0:
            bboxes, labels, confs, _ = detector.detect(frame)
            cached_bboxes = bboxes
            cached_labels = labels
            cached_confs = confs
        else:
            bboxes = cached_bboxes
            labels = cached_labels
            confs = cached_confs

        matched, missing, extra, complete = guide.match(labels)

        if complete:
            consecutive_complete += 1
        else:
            consecutive_complete = 0

        auto_progress = min(consecutive_complete / auto_advance_frames, 1.0)
        advanced = False
        if consecutive_complete >= auto_advance_frames:
            if guide.advance():
                consecutive_complete = 0
                advanced = True
                print(f">>> Auto-advanced to step {guide.action_num}/{guide.num_actions}")

        gesture_action = gesture.update(frame) if gesture else None
        if gesture:
            raw = gesture.raw_gesture
            if raw and raw != "None":
                mapped_str = f" -> {GESTURE_ACTION_MAP[raw]}" if raw in GESTURE_ACTION_MAP else ""
                hold_str = f"  hold {int(gesture.progress * 100)}%" if gesture.active_gesture else ""
                print(f"  {raw} ({gesture.raw_confidence:.0%}){mapped_str}{hold_str}")
        if gesture_action == "advance":
            if guide.advance():
                consecutive_complete = 0
                advanced = True
                print(f"[gesture] Skipped to step {guide.action_num}/{guide.num_actions}")
        elif gesture_action == "go_back":
            if guide.go_back():
                consecutive_complete = 0
                print(f"[gesture] Back to step {guide.action_num}/{guide.num_actions}")
        elif gesture_action == "finish":
            print("[gesture] Finish requested")
            break

        display = draw_overlays(
            frame.copy(), guide, bboxes, labels, confs,
            complete, auto_progress, fps_actual, overlap_iou)

        if gesture:
            gesture.draw_feedback(display)

        if display_scale != 1.0:
            dh, dw = display.shape[:2]
            display = cv2.resize(display,
                                 (int(dw * display_scale),
                                  int(dh * display_scale)))

        cv2.imshow("Playback", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key in (ord("n"), 83):
            if guide.advance():
                consecutive_complete = 0
                print(f"-> Step {guide.action_num}/{guide.num_actions}")
        elif key in (ord("p"), 81):
            if guide.go_back():
                consecutive_complete = 0
                print(f"<- Step {guide.action_num}/{guide.num_actions}")
        elif key == ord("r"):
            guide.reset()
            consecutive_complete = 0
            print("Reset to step 1")

        frame_count += 1
        elapsed = time.time() - t_start
        fps_actual = frame_count / elapsed if elapsed > 0 else 0

        if frame_count % 60 == 0 or advanced:
            step_s = guide.current
            print(f"Frame {frame_count:5d} | Step {guide.action_num}/{guide.num_actions} | "
                  f"Det:{len(labels)} Miss:{len(missing)} | "
                  f"{'COMPLETE' if complete else 'in progress'} | "
                  f"{fps_actual:.1f}fps")

    cap.release()
    if gesture:
        gesture.close()
    cv2.destroyAllWindows()
    print(f"\nPlayback ended. {frame_count} frames processed.")


def main():
    parser = argparse.ArgumentParser(description="Assembly Playback Prototype")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    run_playback(args.config)


if __name__ == "__main__":
    main()
