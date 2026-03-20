#!/usr/bin/env python3
"""
SAM2-Only Video Segmentation
============================
Uses SAM2 for automatic segmentation without a detection model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from typing import List, Tuple
import time
import warnings

warnings.filterwarnings('ignore')

from lib.detection import Detection
from lib.tracker import SimpleTracker
from lib.visualizer import Visualizer
from lib.config import load_config, merge_global_config, get_colors


class SAM2Segmenter:
    """SAM2 automatic segmentation."""
    
    def __init__(self, config: dict):
        self.config = config
        
        sam2_cfg = config.get('sam2', {})
        self.min_mask_region_area = sam2_cfg.get('min_mask_region_area', 100)
        self.imgsz = config['inference'].get('imgsz', 640)
        
        device = config['inference']['device']
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        from ultralytics import SAM
        
        sam_path = config['paths']['sam2']
        print(f"Loading SAM2: {sam_path}")
        self.sam = SAM(sam_path)
        print(f"SAM2 loaded on: {self.device}")
    
    def __call__(self, image: np.ndarray) -> List[Detection]:
        h, w = image.shape[:2]
        target_size = self.imgsz
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        results = self.sam.predict(image_small, verbose=False)
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') and result.boxes.conf is not None else np.ones(len(masks))
                
                for i, (mask, conf) in enumerate(zip(masks, confs)):
                    if scale != 1.0:
                        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    if mask_binary.sum() < self.min_mask_region_area:
                        continue
                    
                    ys, xs = np.where(mask_binary > 0)
                    if len(xs) == 0: continue
                    
                    bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
                    
                    detections.append(Detection(
                        bbox=bbox,
                        confidence=float(conf),
                        mask=mask_binary,
                        instance_id=i + 1
                    ))
        
        return detections


class VideoProcessor:
    def __init__(self, config_path: str = "config.yaml"):
        script_dir = Path(__file__).resolve().parent
        self.config = load_config(config_path)
        merge_global_config(self.config, script_dir)
        colors = get_colors(self.config)

        self.model = SAM2Segmenter(self.config)
        self.visualizer = Visualizer.from_config(self.config, colors)
        self.tracker = SimpleTracker.from_config(self.config) if self.config['tracking']['enabled'] else None
        
        self.frame_interval = self.config['sampling']['frame_interval']
        self.start_frame = self.config['sampling']['start_frame']
        self.max_frames = self.config['sampling'].get('max_frames')
    
    def process(self):
        input_path = Path(self.config['paths']['input_video'])
        output_dir = Path(self.config['paths']['output_dir'])
        
        dirs = {}
        if self.config['output']['save_overlay']:
            dirs['overlay'] = output_dir / "overlay"
            dirs['overlay'].mkdir(parents=True, exist_ok=True)
        if self.config['output']['save_panoptic_mask']:
            dirs['panoptic'] = output_dir / "panoptic"
            dirs['panoptic'].mkdir(parents=True, exist_ok=True)
        if self.config['output']['save_instance_mask']:
            dirs['instance'] = output_dir / "instance"
            dirs['instance'].mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nInput: {input_path}")
        print(f"Resolution: {w}x{h}, FPS: {fps:.2f}, Frames: {total}")
        
        frame_indices = list(range(self.start_frame, total, self.frame_interval))
        if self.max_frames:
            frame_indices = frame_indices[:self.max_frames]
        
        print(f"Processing {len(frame_indices)} frames")
        
        img_fmt = self.config['output']['image_format']
        start_time = time.time()
        
        if self.tracker:
            self.tracker.reset()
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            detections = self.model(frame)
            
            if self.tracker:
                detections = self.tracker.update(detections)
            else:
                for j, det in enumerate(detections):
                    det.track_id = j + 1
            
            frame_name = f"frame_{frame_idx:06d}"
            
            if 'overlay' in dirs:
                overlay = self.visualizer.create_overlay(frame, detections)
                cv2.imwrite(str(dirs['overlay'] / f"{frame_name}.{img_fmt}"), overlay)
            
            if 'panoptic' in dirs:
                panoptic = self.visualizer.create_panoptic_mask(frame.shape[:2], detections)
                cv2.imwrite(str(dirs['panoptic'] / f"{frame_name}.{img_fmt}"), panoptic)
            
            if 'instance' in dirs:
                instance = self.visualizer.create_instance_mask(frame.shape[:2], detections)
                cv2.imwrite(str(dirs['instance'] / f"{frame_name}.png"), instance)
            
            elapsed = time.time() - start_time
            fps_actual = (i + 1) / elapsed if elapsed > 0 else 0
            
            print(f"Frame {frame_idx:6d} | Objects: {len(detections):3d} | FPS: {fps_actual:.2f}")
        
        cap.release()
        
        elapsed = time.time() - start_time
        print(f"\nDone! {len(frame_indices)} frames in {elapsed:.1f}s")
        print(f"Output: {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.yaml")
    args = parser.parse_args()
    
    processor = VideoProcessor(args.config)
    processor.process()


if __name__ == "__main__":
    main()
