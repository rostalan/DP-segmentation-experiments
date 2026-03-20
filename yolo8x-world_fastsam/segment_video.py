#!/usr/bin/env python3
"""
YOLO-World + FastSAM Video Segmentation
========================================
Fast zero-shot segmentation using:
- YOLO-World: Open-vocabulary detection
- FastSAM: Fast segment anything
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
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


class YOLOWorldFastSAMSegmenter:
    """YOLO-World + FastSAM for fast zero-shot segmentation."""
    
    def __init__(self, config: dict):
        self.config = config
        self.class_names = config.get('class_names', ['object'])
        self.conf_thresh = config['yolo']['confidence_threshold']
        self.iou_thresh = config['yolo']['iou_threshold']
        
        fastsam_cfg = config.get('fastsam', {})
        self.fastsam_conf = fastsam_cfg.get('conf', 0.4)
        self.fastsam_iou = fastsam_cfg.get('iou', 0.9)
        self.retina_masks = fastsam_cfg.get('retina_masks', True)
        
        device = config['inference']['device']
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        from ultralytics import YOLOWorld, FastSAM
        
        yolo_path = config['paths']['yolo_model']
        print(f"Loading YOLO-World: {yolo_path}")
        self.yolo = YOLOWorld(yolo_path)
        self.yolo.set_classes(self.class_names)
        
        fastsam_path = config['paths']['fastsam_model']
        print(f"Loading FastSAM: {fastsam_path}")
        self.fastsam = FastSAM(fastsam_path)
        
        print(f"Models loaded on: {self.device}")
    
    def __call__(self, image: np.ndarray) -> List[Detection]:
        h, w = image.shape[:2]
        max_size = 640
        scale = 1.0
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image_small = cv2.resize(image, (new_w, new_h))
        else:
            image_small = image
            new_h, new_w = h, w
        
        yolo_results = self.yolo.predict(
            image_small,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False
        )
        
        if not yolo_results or len(yolo_results[0].boxes) == 0:
            return []
        
        result = yolo_results[0]
        boxes_small = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        if len(boxes_small) == 0:
            return []
        
        fastsam_results = self.fastsam(
            image_small,
            bboxes=boxes_small,
            device=self.device,
            retina_masks=self.retina_masks,
            conf=self.fastsam_conf,
            iou=self.fastsam_iou,
            verbose=False
        )
        
        detections = []
        masks = None
        if fastsam_results and len(fastsam_results) > 0:
            if fastsam_results[0].masks is not None:
                masks = fastsam_results[0].masks.data.cpu().numpy()
        
        for i, (box_small, conf, cls_id) in enumerate(zip(boxes_small, confs, cls_ids)):
            box = box_small / scale
            
            if masks is not None and i < len(masks):
                mask_small = masks[i]
                mask = cv2.resize(mask_small.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0.5).astype(np.uint8)
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask[y1:y2, x1:x2] = 1
                
            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                
            detections.append(Detection(
                bbox=box,
                confidence=float(conf),
                mask=mask,
                class_id=int(cls_id),
                class_name=class_name
            ))
        
        return detections


class VideoProcessor:
    def __init__(self, config_path: str = "config.yaml"):
        script_dir = Path(__file__).resolve().parent
        self.config = load_config(config_path)
        merge_global_config(self.config, script_dir)
        colors = get_colors(self.config)

        self.model = YOLOWorldFastSAMSegmenter(self.config)
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
            
            print(f"Frame {frame_idx:6d} | Dets: {len(detections):3d} | FPS: {fps_actual:.2f}")
        
        cap.release()
        
        elapsed = time.time() - start_time
        print(f"\nDone! {len(frame_indices)} frames in {elapsed:.1f}s")
        print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.yaml")
    args = parser.parse_args()
    
    processor = VideoProcessor(args.config)
    processor.process()


if __name__ == "__main__":
    main()
