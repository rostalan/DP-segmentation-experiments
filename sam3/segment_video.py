#!/usr/bin/env python3
"""
SAM3 Interactive Video Segmentation
"""

import cv2
import numpy as np
from pathlib import Path
import time
import argparse
import os
import tempfile

# Disable torch.compile/Inductor to avoid Python.h dependency issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
try:
    torch.compile = lambda model, *args, **kwargs: model
except Exception:
    pass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.config import load_config, merge_global_config
from lib.sam_utils import InteractiveSelector, make_custom_sam3_predictor


class SAM3VideoProcessor:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        merge_global_config(self.config, Path(__file__).resolve().parent)
        self.colors = self.config.get('_global_colors', [])

        self.model_path = self.config['paths'].get('sam3_model', self.config['paths'].get('sam3', '../models/sam3.pt'))
        if 'yolo' in str(self.model_path).lower():
             self.model_path = "../models/sam3.pt"
             
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.input_video = self.config['paths']['input_video']
        
    def run(self):
        start_frame = self.config['sampling'].get('start_frame', 0)
        frame_interval = self.config['sampling'].get('frame_interval', 1)
        max_frames = self.config['sampling'].get('max_frames')

        cap = cv2.VideoCapture(self.input_video)
        if not cap.isOpened():
            print(f"Error opening video: {self.input_video}")
            return
            
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
        ret, first_frame = cap.read()
        if not ret:
            print("Error reading first frame.")
            return
            
        selector = InteractiveSelector(first_frame)
        bboxes = selector.select()
        
        if not bboxes:
            print("No objects selected. Exiting.")
            return
            
        print(f"Selected {len(bboxes)} objects. Initializing SAM3...")
        
        conf_threshold = self.config['inference'].get('conf_threshold', 0.25)
        device = self.config['inference'].get('device', 'cpu')
        
        overrides = dict(
            conf=conf_threshold,
            task="segment",
            mode="predict",
            model=self.model_path,
            half=False,
            save=False,
            device=device
        )
        
        try:
            CustomSAM3VideoPredictor = make_custom_sam3_predictor()
            predictor = CustomSAM3VideoPredictor(overrides=overrides)
        except Exception as e:
            print(f"Failed to initialize CustomSAM3VideoPredictor: {e}")
            return

        def read_frames():
            frames = []
            frames.append(first_frame)
            count = 1
            
            print(f"Reading frames (Start: {start_frame}, Interval: {frame_interval})...")
            
            while cap.isOpened():
                if max_frames is not None and count >= max_frames:
                    break
                
                skipped = 0
                while skipped < (frame_interval - 1):
                    if not cap.grab(): 
                        return frames
                    skipped += 1
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame)
                count += 1
                
                if count % 100 == 0:
                    print(f"Read {count} frames...")
            
            return frames

        frames = read_frames()
        cap.release()
        
        print(f"Loaded {len(frames)} frames.")

        temp_video_path = None
        source_input = self.input_video

        if frames:
            height, width = frames[0].shape[:2]
            temp_video_fd, temp_video_path = tempfile.mkstemp(suffix='.mp4')
            os.close(temp_video_fd)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (width, height))
            
            print(f"Creating temp video: {temp_video_path}")
            for frame in frames:
                out.write(frame)
            out.release()
            source_input = temp_video_path

        try:
            results = predictor(
                source=source_input,
                bboxes=bboxes,
                stream=True
            )
            
            overlay_dir = self.output_dir / "overlay"
            overlay_dir.mkdir(parents=True, exist_ok=True)
            
            start_time = time.time()
            
            show_boxes = self.config['output'].get('show_boxes', True)
            show_conf = self.config['output'].get('show_conf', False)

            for i, result in enumerate(results):
                annotated_frame = result.plot(boxes=show_boxes, conf=show_conf)
                frame_name = f"frame_{i:06d}.jpg"
                save_path = overlay_dir / frame_name
                cv2.imwrite(str(save_path), annotated_frame)
                
                elapsed = time.time() - start_time
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"Processed frame {i} | FPS: {fps:.2f} | Saved to {save_path}")

        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                print(f"Removing temp video: {temp_video_path}")
                os.unlink(temp_video_path)
            
        print(f"\nProcessing complete. Output: {overlay_dir}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 Interactive Video Segmentation")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    processor = SAM3VideoProcessor(args.config)
    processor.run()

if __name__ == "__main__":
    main()
