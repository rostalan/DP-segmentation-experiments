#!/usr/bin/env python3
"""
Gesture recognition test — webcam viewer that shows hand landmarks and
the detected gesture in real time.  No YOLO or guide needed.

Controls:
    q / Esc  -- quit
"""

import cv2
import time
from gesture_control import GestureController, GESTURE_ACTION_MAP

FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    gesture = GestureController(
        model_path="../models/gesture_recognizer.task",
        detect_interval=1,
        hold_detections=5,
        cooldown_detections=8,
        min_confidence=0.50,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam: {w}x{h}")
    print(f"Mapped gestures: {list(GESTURE_ACTION_MAP.keys())}")
    print("Hold a gesture to trigger an action. Press 'q' to quit.\n")

    frame_count = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        action = gesture.update(frame)

        raw = gesture.raw_gesture
        raw_conf = gesture.raw_confidence
        active = gesture.active_gesture
        progress = gesture.progress

        if raw and raw != "None":
            mapped_str = f" -> {GESTURE_ACTION_MAP[raw]}" if raw in GESTURE_ACTION_MAP else ""
            hold_str = f"  hold {int(progress * 100)}%" if active else ""
            print(f"  {raw} ({raw_conf:.0%}){mapped_str}{hold_str}")
        if action:
            print(f"  >>> ACTION: {action}")

        display = frame.copy()
        gesture.draw_hand(display)

        h_d, w_d = display.shape[:2]

        y = 30
        cv2.rectangle(display, (0, 0), (w_d, 100), (0, 0, 0), -1)

        if raw:
            mapped = "YES" if raw in GESTURE_ACTION_MAP else "no"
            text = f"Detected: {raw}  ({raw_conf:.0%})  mapped={mapped}"
            cv2.putText(display, text, (10, y), FONT, 0.65, (200, 200, 200), 1)
        else:
            cv2.putText(display, "No gesture detected", (10, y),
                        FONT, 0.65, (100, 100, 100), 1)

        y += 30
        if active:
            bar_text = f"Holding: {active}  [{int(progress * 100)}%]"
            color = (0, 200, 255) if progress < 1.0 else (0, 255, 0)
            cv2.putText(display, bar_text, (10, y), FONT, 0.65, color, 2)

            bar_x, bar_y, bar_w, bar_h = 10, y + 8, 300, 16
            cv2.rectangle(display, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            cv2.rectangle(display, (bar_x, bar_y),
                          (bar_x + int(bar_w * progress), bar_y + bar_h),
                          color, -1)
            cv2.rectangle(display, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
        elif gesture.in_cooldown:
            cv2.putText(display, "Cooldown...", (10, y),
                        FONT, 0.65, (0, 100, 200), 1)

        gesture.draw_feedback(display)

        frame_count += 1
        elapsed = time.time() - t_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(display, f"{fps:.0f} fps", (w_d - 100, 30),
                    FONT, 0.6, (120, 120, 120), 1)

        cv2.imshow("Gesture Test", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    gesture.close()
    cv2.destroyAllWindows()
    print(f"\nDone. {frame_count} frames, {fps:.1f} fps avg.")


if __name__ == "__main__":
    main()
