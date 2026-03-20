#!/usr/bin/env python3
"""
YOLOE Text-Prompted Segmentation Pipeline
==========================================
Uses YOLOE with text prompts (RepRTA) for open-vocabulary detection + segmentation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import time
import warnings

from lib.config import load_config, merge_global_config, get_colors
from lib.detection import Detection, Track, compute_iou
from lib.tracker import SimpleTracker
from lib.visualizer import Visualizer

warnings.filterwarnings('ignore')


class YOLOESegmenter:
    def __init__(self, config: dict):
        self.config = config
        self.conf_thresh = config['inference']['confidence_threshold']
        self.iou_thresh = config['inference']['iou_threshold']
        self.class_names = config.get('class_names', ['person', 'laptop'])

        device = config['inference']['device']
        if device == 'auto':
            import torch
            if torch.cuda.is_available(): device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): device = 'mps'
            else: device = 'cpu'
        self.device = device
        
        from ultralytics import YOLOE
        
        model_path = config['paths']['yoloe']
        print(f"Loading YOLOE model: {model_path}")
        self.model = YOLOE(model_path)
        self.model.to(self.device)
        self.model.set_classes(self.class_names)
        
        print(f"Model loaded on device: {self.device}")
    
    def __call__(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        
        results = self.model.predict(
            image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False
        )
        
        if len(results) == 0: return []
        
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0: return []
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        masks = result.masks
        mask_data = masks.data.cpu().numpy() if masks is not None else None
        
        detections = []
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
            
            if mask_data is not None and i < len(mask_data):
                mask = mask_data[i]
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)
            else:
                x1, y1, x2, y2 = map(int, box)
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 1
            
            detections.append(Detection(
                bbox=box,
                confidence=float(conf),
                class_id=cls_id,
                class_name=cls_name,
                instance_id=i + 1,
                mask=mask
            ))
        
        return detections


class VideoProcessor:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        merge_global_config(self.config, Path(__file__).resolve().parent)

        self.segmenter = YOLOESegmenter(self.config)
        self.tracker = SimpleTracker.from_config(self.config)
        self.visualizer = Visualizer.from_config(self.config, get_colors(self.config))
        
        self.tracking_enabled = self.config.get('tracking', {}).get('enabled', False)
        
        sampling = self.config['sampling']
        self.frame_interval = sampling['frame_interval']
        self.start_frame = sampling['start_frame']
        self.max_frames = sampling.get('max_frames')
    
    def process(self):
        paths = self.config['paths']
        input_path = Path(paths['input_video'])
        output_dir = Path(paths['output_dir'])
        output_cfg = self.config.get('output', {})
        
        dirs = {}
        if output_cfg.get('save_overlay', True):
            dirs['overlay'] = output_dir / 'overlay'
            dirs['overlay'].mkdir(parents=True, exist_ok=True)
        if output_cfg.get('save_panoptic_mask', True):
            dirs['panoptic'] = output_dir / 'panoptic'
            dirs['panoptic'].mkdir(parents=True, exist_ok=True)
        if output_cfg.get('save_instance_mask', True):
            dirs['instance'] = output_dir / 'instance'
            dirs['instance'].mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nInput: {input_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")
        
        frame_indices = list(range(self.start_frame, total_frames, self.frame_interval))
        if self.max_frames:
            frame_indices = frame_indices[:self.max_frames]
        
        print(f"Processing {len(frame_indices)} frames")
        print(f"Output: {output_dir}\n")
        
        img_fmt = output_cfg.get('image_format', 'png')
        start_time = time.time()
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            detections = self.segmenter(frame)
            
            if self.tracking_enabled:
                detections = self.tracker.update(detections)
            
            frame_name = f"frame_{frame_idx:06d}"
            
            if 'overlay' in dirs:
                overlay = self.visualizer.create_overlay(frame, detections)
                cv2.imwrite(str(dirs['overlay'] / f"{frame_name}.{img_fmt}"), overlay)
            
            if 'panoptic' in dirs:
                panoptic = self.visualizer.create_panoptic_mask(frame.shape, detections)
                cv2.imwrite(str(dirs['panoptic'] / f"{frame_name}.{img_fmt}"), panoptic)
            
            if 'instance' in dirs:
                instance = self.visualizer.create_instance_mask(frame.shape, detections)
                cv2.imwrite(str(dirs['instance'] / f"{frame_name}.png"), instance)
            
            class_names = list(set(d.class_name for d in detections))
            elapsed = time.time() - start_time
            fps_actual = (i + 1) / elapsed if elapsed > 0 else 0
            
            print(f"Frame {frame_idx:6d} | Detections: {len(detections):3d} | Classes: {class_names} | FPS: {fps_actual:.2f}")
        
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
