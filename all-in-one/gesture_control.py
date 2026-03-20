#!/usr/bin/env python3
"""
Gesture-based playback control using MediaPipe GestureRecognizer.

Recognises three gestures and maps them to playback actions:
    Thumb_Up   → advance  (force-skip current step)
    Thumb_Down → go_back  (undo false advance)
    Victory    → finish   (end playback)

A gesture must be held for ``hold_detections`` consecutive recognition
cycles before triggering.  After triggering, a cooldown period prevents
repeated accidental fires.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.hand import HAND_CONNECTIONS


GESTURE_ACTION_MAP = {
    "Thumb_Up": "advance",
    "Thumb_Down": "go_back",
    "Victory": "finish",
}

ACTION_DISPLAY = {
    "advance": ("SKIP >>", (0, 220, 0)),
    "go_back": ("<< BACK", (0, 140, 255)),
    "finish": ("FINISH", (0, 200, 255)),
}

GESTURE_DISPLAY = {
    "Thumb_Up": "Skip step",
    "Thumb_Down": "Go back",
    "Victory": "Finish",
}


class GestureController:
    """Wraps MediaPipe GestureRecognizer with debounce and cooldown logic."""

    def __init__(
        self,
        model_path: str = "../models/gesture_recognizer.task",
        detect_interval: int = 2,
        hold_detections: int = 5,
        cooldown_detections: int = 8,
        min_confidence: float = 0.55,
        resolution: Tuple[int, int] = (640, 480),
    ):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions

        self._mp = mp
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_tracking_confidence=0.3,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

        self.detect_interval = detect_interval
        self.hold_detections = hold_detections
        self.cooldown_detections = cooldown_detections
        self.min_confidence = min_confidence
        self.resolution = resolution

        self._frame_count: int = 0
        self._timestamp_ms: int = 0
        self._current_gesture: Optional[str] = None
        self._current_confidence: float = 0.0
        self._consecutive: int = 0
        self._cooldown_remaining: int = 0
        self._last_action: Optional[str] = None
        self._flash_frames: int = 0
        self._landmarks: list = []
        self._raw_gesture: Optional[str] = None
        self._raw_confidence: float = 0.0

    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> Optional[str]:
        """Process one frame.  Returns an action string when triggered."""
        self._frame_count += 1
        self._timestamp_ms += 33

        if self._flash_frames > 0:
            self._flash_frames -= 1

        if self._frame_count % self.detect_interval != 0:
            return None

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._consecutive = 0
            self._current_gesture = None
            return None

        small = cv2.resize(frame, self.resolution)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        result = self._recognizer.recognize_for_video(
            mp_image, self._timestamp_ms
        )

        self._landmarks = result.hand_landmarks if result.hand_landmarks else []

        gesture_name = None
        confidence = 0.0
        self._raw_gesture = None
        self._raw_confidence = 0.0
        if result.gestures and len(result.gestures) > 0:
            top = result.gestures[0][0]
            self._raw_gesture = top.category_name
            self._raw_confidence = top.score
            if (
                top.category_name in GESTURE_ACTION_MAP
                and top.score >= self.min_confidence
            ):
                gesture_name = top.category_name
                confidence = top.score

        if gesture_name and gesture_name == self._current_gesture:
            self._consecutive += 1
            self._current_confidence = confidence
        elif gesture_name:
            self._current_gesture = gesture_name
            self._current_confidence = confidence
            self._consecutive = 1
        else:
            self._current_gesture = None
            self._current_confidence = 0.0
            self._consecutive = 0

        if self._consecutive >= self.hold_detections:
            action = GESTURE_ACTION_MAP[self._current_gesture]
            self._last_action = action
            self._flash_frames = 20
            self._cooldown_remaining = self.cooldown_detections
            self._consecutive = 0
            self._current_gesture = None
            return action

        return None

    # ------------------------------------------------------------------
    # Properties for overlay drawing
    # ------------------------------------------------------------------

    @property
    def active_gesture(self) -> Optional[str]:
        return self._current_gesture

    @property
    def progress(self) -> float:
        if self.hold_detections <= 0:
            return 0.0
        return min(self._consecutive / self.hold_detections, 1.0)

    @property
    def in_cooldown(self) -> bool:
        return self._cooldown_remaining > 0

    @property
    def raw_gesture(self) -> Optional[str]:
        """Last detected gesture name (any, not just mapped ones)."""
        return self._raw_gesture

    @property
    def raw_confidence(self) -> float:
        return self._raw_confidence

    # ------------------------------------------------------------------
    # Visual feedback
    # ------------------------------------------------------------------

    def draw_hand(self, frame: np.ndarray,
                  color_line: Tuple[int, int, int] = (0, 255, 0),
                  color_point: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """Draw hand skeleton landmarks onto *frame* (mutates).

        Landmarks are in normalised [0, 1] coordinates relative to the
        detection resolution, so they scale to any output size.
        """
        h, w = frame.shape[:2]
        for hand_lm in self._landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], color_line, 2)
            for px, py in pts:
                cv2.circle(frame, (px, py), 4, color_point, -1)
        return frame

    def draw_feedback(self, frame: np.ndarray) -> np.ndarray:
        """Draw gesture recognition feedback onto *frame* (mutates)."""
        h, w = frame.shape[:2]
        self.draw_hand(frame)

        if self._flash_frames > 0 and self._last_action:
            label, color = ACTION_DISPLAY.get(
                self._last_action, (self._last_action, (255, 255, 255))
            )
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
            )
            cx = w // 2 - tw // 2
            cy = h // 2
            cv2.rectangle(
                frame,
                (cx - 14, cy - th - 14),
                (cx + tw + 14, cy + 14),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame, label, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3,
            )

        if self._current_gesture and self._consecutive > 0:
            label = GESTURE_DISPLAY.get(
                self._current_gesture, self._current_gesture
            )
            progress = self.progress

            bar_w = 200
            bar_h = 22
            bx = w // 2 - bar_w // 2
            by = h - 90

            cv2.rectangle(
                frame, (bx, by), (bx + bar_w, by + bar_h), (60, 60, 60), -1
            )
            fill_w = int(bar_w * progress)
            bar_color = (0, 200, 255) if progress < 1.0 else (0, 255, 0)
            cv2.rectangle(
                frame, (bx, by), (bx + fill_w, by + bar_h), bar_color, -1
            )
            cv2.rectangle(
                frame, (bx, by), (bx + bar_w, by + bar_h), (200, 200, 200), 1
            )

            conf_str = f"{label}  {self._current_confidence:.0%}"
            (tw, _), _ = cv2.getTextSize(
                conf_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.putText(
                frame, conf_str, (w // 2 - tw // 2, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

        return frame

    # ------------------------------------------------------------------

    def close(self):
        self._recognizer.close()
