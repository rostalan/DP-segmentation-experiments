#!/usr/bin/env python3
"""
YOLO-World + YOLOE Segmentation Pipeline
=========================================
Combines YOLO-World (open-vocabulary detection) with YOLOE (efficient segmentation).
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.config import load_config, merge_global_config, get_colors
from lib.detection import Detection, compute_iou
from lib.visualizer import Visualizer

import time
import warnings

warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLOWorld, YOLOE
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics>=8.3.0")
    exit(1)


class YOLOWorldYOLOESegmenter:
    """YOLO-World detection + YOLOE segmentation pipeline."""
    
    def __init__(self, config: dict):
        self.config = config

        device = config['inference']['device']
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        yolo_world_path = config['paths']['yolo_world']
        print(f"Loading YOLO-World: {yolo_world_path}")
        self.yolo_world = YOLOWorld(yolo_world_path)
        self.yolo_world.to(self.device)
        
        self.class_names = config.get('custom_classes', ['object'])
        self.yolo_world.set_classes(self.class_names)
        
        yoloe_path = config['paths']['yoloe']
        print(f"Loading YOLOE: {yoloe_path}")
        self.yoloe = YOLOE(yoloe_path)
        self.yoloe.to(self.device)
        self.yoloe.set_classes(self.class_names)
        
        yolo_cfg = config.get('yolo_world', {})
        self.det_conf = yolo_cfg.get('confidence_threshold', 0.25)
        self.det_iou = yolo_cfg.get('iou_threshold', 0.45)
        self.det_imgsz = yolo_cfg.get('imgsz', 640)
        
        yoloe_cfg = config.get('yoloe', {})
        self.seg_imgsz = yoloe_cfg.get('imgsz', 640)
        self.seg_conf = yoloe_cfg.get('conf', 0.1)
        
        print(f"Models loaded on: {self.device}")
    
    def __call__(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        
        det_results = self.yolo_world.predict(
            image,
            conf=self.det_conf,
            iou=self.det_iou,
            imgsz=self.det_imgsz,
            verbose=False
        )
        
        if len(det_results) == 0 or det_results[0].boxes is None or len(det_results[0].boxes) == 0:
            return []
        
        det_result = det_results[0]
        boxes = det_result.boxes.xyxy.cpu().numpy()
        confs = det_result.boxes.conf.cpu().numpy()
        cls_ids = det_result.boxes.cls.cpu().numpy().astype(int)
        
        seg_results = self.yoloe.predict(
            image,
            conf=self.seg_conf,
            imgsz=self.seg_imgsz,
            verbose=False
        )
        
        yoloe_masks = None
        yoloe_boxes = None
        if seg_results and seg_results[0].masks is not None:
            yoloe_masks = seg_results[0].masks.data.cpu().numpy()
            yoloe_boxes = seg_results[0].boxes.xyxy.cpu().numpy()
        
        detections = []
        used_masks = set()
        
        for i, (bbox, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
            
            if cls_name == "person":
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > 0.2 * (w * h):
                    continue

            mask = np.zeros((h, w), dtype=np.uint8)
            
            if yoloe_masks is not None and yoloe_boxes is not None:
                best_iou = 0
                best_mask_idx = -1
                
                for m_idx, yoloe_box in enumerate(yoloe_boxes):
                    if m_idx in used_masks: continue
                    iou = compute_iou(bbox, yoloe_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_mask_idx = m_idx
                
                if best_mask_idx >= 0 and best_iou > 0.2:
                    yoloe_mask = yoloe_masks[best_mask_idx]
                    yoloe_mask = cv2.resize(yoloe_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    mask = (yoloe_mask > 0.5).astype(np.uint8)
                    used_masks.add(best_mask_idx)
            
            if mask.sum() == 0:
                x1, y1, x2, y2 = bbox.astype(int)
                mask[y1:y2, x1:x2] = 1
            
            detections.append(Detection(
                bbox=bbox,
                confidence=float(conf),
                class_id=cls_id,
                class_name=cls_name,
                mask=mask,
                instance_id=i + 1
            ))
        
        return detections


class VideoProcessor:
    """Process video with YOLO-World + YOLOE pipeline."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        merge_global_config(self.config, Path(__file__).resolve().parent)

        self.model = YOLOWorldYOLOESegmenter(self.config)
        self.visualizer = Visualizer.from_config(self.config, get_colors(self.config))
        
        self.frame_interval = self.config['sampling']['frame_interval']
        self.start_frame = self.config['sampling']['start_frame']
        self.max_frames = self.config['sampling'].get('max_frames')
    
    def process(self):
        paths = self.config['paths']
        input_path = Path(paths['input_video'])
        output_dir = Path(paths['output_dir'])
        output_cfg = self.config.get('output', {})
        
        dirs = {}
        if output_cfg.get('save_overlay', True):
            dirs['overlay'] = output_dir / "overlay"
            dirs['overlay'].mkdir(parents=True, exist_ok=True)
        if output_cfg.get('save_panoptic_mask', True):
            dirs['panoptic'] = output_dir / "panoptic"
            dirs['panoptic'].mkdir(parents=True, exist_ok=True)
        if output_cfg.get('save_instance_mask', True):
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
        print(f"Output: {output_dir}\n")
        
        img_fmt = output_cfg.get('image_format', 'png')
        start_time = time.time()
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            detections = self.model(frame)
            
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
            
            elapsed = time.time() - start_time
            fps_actual = (i + 1) / elapsed if elapsed > 0 else 0
            
            print(f"Frame {frame_idx:6d} | Detections: {len(detections):3d} | FPS: {fps_actual:.2f}")
        
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
