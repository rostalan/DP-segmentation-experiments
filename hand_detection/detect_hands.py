#!/usr/bin/env python3
"""
MediaPipe Hand Detection & Gesture Recognition Pipeline
=====================================================
Detects hands, 3D landmarks, and recognized gestures in video frames using MediaPipe.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
import time
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.hand import HAND_CONNECTIONS
from lib.config import load_config, merge_global_config
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        merge_global_config(self.config, Path(__file__).resolve().parent)
        
        mp_cfg = self.config.get('mediapipe', {})
        self.mode = mp_cfg.get('mode', 'landmarks')  # 'landmarks' or 'gestures'
        
        num_hands = mp_cfg.get('max_num_hands', 2)
        min_det_conf = mp_cfg.get('min_detection_confidence', 0.5)
        min_track_conf = mp_cfg.get('min_tracking_confidence', 0.5)
        
        if self.mode == 'gestures':
            model_path = mp_cfg.get('model_path_gestures', '../models/gesture_recognizer.task')
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                num_hands=num_hands,
                min_hand_detection_confidence=min_det_conf,
                min_tracking_confidence=min_track_conf,
                running_mode=vision.RunningMode.VIDEO
            )
            self.detector = vision.GestureRecognizer.create_from_options(options)
            print(f"Initialized Gesture Recognizer with model: {model_path}")
        else:
            model_path = mp_cfg.get('model_path_landmarks', '../models/hand_landmarker.task')
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=num_hands,
                min_hand_detection_confidence=min_det_conf,
                min_tracking_confidence=min_track_conf,
                running_mode=vision.RunningMode.VIDEO
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            print(f"Initialized Hand Landmarker with model: {model_path}")
        
    def draw_landmarks(self, image, hand_landmarks_list):
        h, w = image.shape[:2]
        for landmarks in hand_landmarks_list:
            # Draw points
            points = []
            for lm in landmarks:
                px = int(lm.x * w)
                py = int(lm.y * h)
                points.append((px, py))
                cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
            
            # Draw connections
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                if p1_idx < len(points) and p2_idx < len(points):
                    cv2.line(image, points[p1_idx], points[p2_idx], (0, 255, 0), 2)

    def draw_gestures(self, image, gestures_list, hand_landmarks_list):
        h, w = image.shape[:2]
        
        # First draw landmarks
        if hand_landmarks_list:
            self.draw_landmarks(image, hand_landmarks_list)
            
        # Then draw gesture labels
        if gestures_list and hand_landmarks_list:
            for hand_idx, (gestures, landmarks) in enumerate(zip(gestures_list, hand_landmarks_list)):
                if not gestures: continue
                
                # Get top gesture
                top_gesture = gestures[0]
                text = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
                
                # Position text above the wrist (landmark 0)
                wrist = landmarks[0]
                px = int(wrist.x * w)
                py = int(wrist.y * h) - 20
                
                cv2.putText(image, text, (max(10, px - 50), max(30, py)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    def process_video(self):
        input_path = Path(self.config['paths']['input_video'])
        output_dir = Path(self.config['paths']['output_dir'])
        
        dirs = {}
        if self.config['output'].get('save_overlay', True):
            dirs['overlay'] = output_dir / "overlay"
            dirs['overlay'].mkdir(parents=True, exist_ok=True)
        if self.config['output'].get('save_crop', False):
            dirs['crops'] = output_dir / "crops"
            dirs['crops'].mkdir(parents=True, exist_ok=True)
            
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Error opening video: {input_path}")
            return
            
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Input: {input_path}")
        print(f"Resolution: {w}x{h}, FPS: {fps:.2f}, Frames: {total}")
        
        sampling = self.config['sampling']
        frame_interval = sampling.get('frame_interval', 1)
        start_frame = sampling.get('start_frame', 0)
        max_frames = sampling.get('max_frames')
        
        frame_indices = list(range(start_frame, total, frame_interval))
        if max_frames:
            frame_indices = frame_indices[:max_frames]
            
        print(f"Processing {len(frame_indices)} frames...")
        
        img_fmt = self.config['output'].get('image_format', 'png')
        start_time = time.time()
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            # MediaPipe Tasks requires RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Current timestamp in ms
            timestamp_ms = int(frame_idx * 1000 / fps) if fps > 0 else int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            if self.mode == 'gestures':
                result = self.detector.recognize_for_video(mp_image, timestamp_ms)
                has_results = bool(result.gestures)
            else:
                result = self.detector.detect_for_video(mp_image, timestamp_ms)
                has_results = bool(result.hand_landmarks)
            
            frame_name = f"frame_{frame_idx:06d}"
            
            # Draw overlay
            if 'overlay' in dirs:
                overlay = frame.copy()
                if self.mode == 'gestures' and result.gestures:
                    self.draw_gestures(overlay, result.gestures, result.hand_landmarks)
                elif self.mode == 'landmarks' and result.hand_landmarks:
                    self.draw_landmarks(overlay, result.hand_landmarks)
                cv2.imwrite(str(dirs['overlay'] / f"{frame_name}.{img_fmt}"), overlay)
                
            # Save crops
            if 'crops' in dirs and has_results and result.hand_landmarks:
                for hand_idx, landmarks in enumerate(result.hand_landmarks):
                    # Get bounding box
                    x_list = [lm.x for lm in landmarks]
                    y_list = [lm.y for lm in landmarks]
                    
                    x1, x2 = int(min(x_list) * w), int(max(x_list) * w)
                    y1, y2 = int(min(y_list) * h), int(max(y_list) * h)
                    
                    # Add padding
                    pad = 20
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(w, x2 + pad)
                    y2 = min(h, y2 + pad)
                    
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        crop_name = f"{frame_name}_hand_{hand_idx}.{img_fmt}"
                        cv2.imwrite(str(dirs['crops'] / crop_name), crop)

            elapsed = time.time() - start_time
            fps_actual = (i + 1) / elapsed if elapsed > 0 else 0
            n_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
            
            if i % 10 == 0:
                print(f"Frame {frame_idx:6d} | Hands: {n_hands} | FPS: {fps_actual:.2f}")
                
        cap.release()
        print(f"\nDone. Output in {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    
    detector = HandDetector(args.config)
    detector.process_video()

if __name__ == "__main__":
    main()
