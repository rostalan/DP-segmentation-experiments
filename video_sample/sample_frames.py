#!/usr/bin/env python3
"""
Video Frame Sampler

Plays a video slowed down and allows saving individual frames via keystroke.

Usage:
    python sample_frames.py <video_path> [--speed SPEED] [--output OUTPUT_DIR]

Controls:
    SPACE - Save current frame
    S     - Save current frame  
    Q/ESC - Quit
    P     - Pause/Resume
    LEFT  - Go back 30 frames
    RIGHT - Skip forward 30 frames
"""

import argparse
import cv2
import os
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Sample frames from a video")
    parser.add_argument("video_path", type=str, help="Path to the input video")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="Playback speed multiplier (default: 0.5 = half speed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sampled_frames",
        help="Output directory for saved frames (default: sampled_frames)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate video path
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return 1

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path.name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Playback speed: {args.speed}x")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    print("Controls:")
    print("  SPACE/S - Save current frame")
    print("  P       - Pause/Resume")
    print("  LEFT    - Go back 30 frames")
    print("  RIGHT   - Skip forward 30 frames")
    print("  Q/ESC   - Quit")
    print()

    # Calculate delay between frames (in ms)
    # Lower speed = higher delay
    base_delay = int(1000 / fps) if fps > 0 else 33
    delay = int(base_delay / args.speed)

    window_name = f"Video Sampler - {video_path.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Set a larger initial window size (scale up to at least 1280x720, maintaining aspect ratio)
    target_width = max(1280, width)
    scale = target_width / width
    target_height = int(height * scale)
    cv2.resizeWindow(window_name, target_width, target_height)

    frame_count = 0
    saved_count = 0
    paused = False
    current_frame = None

    video_basename = video_path.stem

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video reached.")
                break
            current_frame = frame.copy()
            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if current_frame is not None:
            # Create display frame with info overlay
            display_frame = current_frame.copy()

            # Add frame info overlay
            info_text = f"Frame: {frame_count}/{total_frames}"
            if paused:
                info_text += " [PAUSED]"
            cv2.putText(
                display_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                f"Saved: {saved_count} | Speed: {args.speed}x",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow(window_name, display_frame)

        # Wait for key press
        key = cv2.waitKey(delay if not paused else 50) & 0xFF

        # Handle key presses
        if key == ord("q") or key == 27:  # Q or ESC
            print("\nQuitting...")
            break

        elif key == ord(" ") or key == ord("s"):  # SPACE or S - save frame
            if current_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"{video_basename}_frame{frame_count:06d}_{timestamp}.png"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), current_frame)
                saved_count += 1
                print(f"Saved: {filename}")

        elif key == ord("p"):  # P - pause/resume
            paused = not paused
            status = "Paused" if paused else "Playing"
            print(f"\n{status}")

        elif key == 81 or key == 2:  # LEFT arrow - go back
            new_pos = max(0, frame_count - 30)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, frame = cap.read()
            if ret:
                current_frame = frame.copy()
                frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"\nJumped to frame {frame_count}")

        elif key == 83 or key == 3:  # RIGHT arrow - skip forward
            new_pos = min(total_frames - 1, frame_count + 30)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, frame = cap.read()
            if ret:
                current_frame = frame.copy()
                frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"\nJumped to frame {frame_count}")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nTotal frames saved: {saved_count}")
    if saved_count > 0:
        print(f"Saved to: {output_dir.absolute()}")

    return 0


if __name__ == "__main__":
    exit(main())
