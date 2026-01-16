#!/usr/bin/env python3
"""
YOLO-World + FastSAM Video Segmentation
========================================
Fast zero-shot segmentation using:
- YOLO-World: Open-vocabulary detection (any class by text)
- FastSAM: Fast segment anything (~50x faster than SAM)

FastSAM uses a YOLO-based architecture for real-time segmentation.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import time
import warnings

warnings.filterwarnings('ignore')


@dataclass
class Detection:
    """Detection with mask."""
    bbox: np.ndarray       # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    mask: np.ndarray       # Binary mask
    track_id: int = -1


class SimpleTracker:
    """IoU-based tracker for temporal consistency."""
    
    def __init__(self, config: dict):
        tracking_cfg = config.get('tracking', {})
        self.iou_threshold = tracking_cfg.get('iou_threshold', 0.3)
        self.max_age = tracking_cfg.get('max_age', 3)
        self.tracks: Dict[int, dict] = {}
        self.next_id = 1
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]
            return detections
        
        matched_det = set()
        matched_track = set()
        
        for det in detections:
            best_iou = 0
            best_tid = None
            
            for tid, track in self.tracks.items():
                if tid in matched_track:
                    continue
                if track['class_id'] != det.class_id:
                    continue
                
                iou = self._compute_iou(det.bbox, track['bbox'])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_tid = tid
            
            if best_tid is not None:
                det.track_id = best_tid
                self.tracks[best_tid]['bbox'] = det.bbox.copy()
                self.tracks[best_tid]['age'] = 0
                matched_track.add(best_tid)
                matched_det.add(id(det))
        
        for det in detections:
            if id(det) not in matched_det:
                det.track_id = self.next_id
                self.tracks[self.next_id] = {
                    'bbox': det.bbox.copy(),
                    'class_id': det.class_id,
                    'age': 0
                }
                self.next_id += 1
        
        for tid in list(self.tracks.keys()):
            if tid not in matched_track:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]
        
        return detections
    
    def reset(self):
        self.tracks = {}
        self.next_id = 1


class YOLOWorldFastSAMSegmenter:
    """YOLO-World + FastSAM for fast zero-shot segmentation."""
    
    def __init__(self, config: dict):
        self.config = config
        self.class_names = config.get('class_names', ['object'])
        self.conf_thresh = config['yolo']['confidence_threshold']
        self.iou_thresh = config['yolo']['iou_threshold']
        
        # FastSAM settings
        fastsam_cfg = config.get('fastsam', {})
        self.fastsam_conf = fastsam_cfg.get('conf', 0.4)
        self.fastsam_iou = fastsam_cfg.get('iou', 0.9)
        self.retina_masks = fastsam_cfg.get('retina_masks', True)
        
        # Device
        device = config['inference']['device']
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        from ultralytics import YOLOWorld, FastSAM
        
        # Load YOLO-World
        yolo_path = config['paths']['yolo_model']
        print(f"Loading YOLO-World: {yolo_path}")
        self.yolo = YOLOWorld(yolo_path)
        self.yolo.set_classes(self.class_names)
        print(f"Custom classes: {self.class_names}")
        
        # Load FastSAM
        fastsam_path = config['paths']['fastsam_model']
        print(f"Loading FastSAM: {fastsam_path}")
        self.fastsam = FastSAM(fastsam_path)
        
        print(f"Models loaded on: {self.device}")
    
    def __call__(self, image: np.ndarray) -> List[Detection]:
        """Run detection + segmentation."""
        h, w = image.shape[:2]
        
        # Resize for faster processing (if image is large)
        max_size = 640
        scale = 1.0
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image_small = cv2.resize(image, (new_w, new_h))
        else:
            image_small = image
            new_h, new_w = h, w
        
        # 1. YOLO-World detection (on resized image)
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
        
        # 2. FastSAM with bbox prompts (on resized image)
        fastsam_results = self.fastsam(
            image_small,
            bboxes=boxes_small,
            device=self.device,
            retina_masks=self.retina_masks,
            conf=self.fastsam_conf,
            iou=self.fastsam_iou,
            verbose=False
        )
        
        # 3. Build detections and scale back to original size
        detections = []
        
        # Get masks from FastSAM
        masks = None
        if fastsam_results and len(fastsam_results) > 0:
            if fastsam_results[0].masks is not None:
                masks = fastsam_results[0].masks.data.cpu().numpy()
        
        for i, (box_small, conf, cls_id) in enumerate(zip(boxes_small, confs, cls_ids)):
            # Scale box back to original size
            box = box_small / scale
            
            # Get mask for this detection
            if masks is not None and i < len(masks):
                mask_small = masks[i]
                # Resize mask to original image size
                mask = cv2.resize(mask_small.astype(np.float32), (w, h), 
                                 interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0.5).astype(np.uint8)
            else:
                # Fallback: box mask
                mask = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask[y1:y2, x1:x2] = 1
                
            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                
                detections.append(Detection(
                    bbox=box,
                    confidence=float(conf),
                    class_id=int(cls_id),
                class_name=class_name,
                mask=mask
                ))
        
        return detections


class Visualizer:
    """Create output visualizations."""
    
    def __init__(self, config: dict):
        self.colors = [tuple(c) for c in config.get('colors', [(255, 0, 0)])]
        self.alpha = config['output']['overlay_alpha']
        self.show_labels = config['output']['show_labels']
        self.show_conf = config['output']['show_confidence']
        self.show_track = config['output'].get('show_track_id', True)
    
    def _get_color(self, class_id: int, track_id: int) -> Tuple[int, int, int]:
        base = self.colors[class_id % len(self.colors)]
        offset = (track_id * 17) % 30 - 15
        return tuple(max(0, min(255, c + offset)) for c in base)
    
    def create_overlay(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        output = image.copy()
        overlay = image.copy()
        
        for det in detections:
            color = self._get_color(det.class_id, det.track_id)
            overlay[det.mask.astype(bool)] = color
            
            if self.show_labels or self.show_track:
                x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                parts = []
                if self.show_track:
                    parts.append(f"#{det.track_id}")
                if self.show_labels:
                    parts.append(det.class_name)
                if self.show_conf:
                    parts.append(f"{det.confidence:.2f}")
                
                label = " ".join(parts)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(output, label, (x1 + 2, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0)
    
    def create_panoptic(self, shape: Tuple[int, int], detections: List[Detection]) -> np.ndarray:
        h, w = shape
        panoptic = np.zeros((h, w, 3), dtype=np.uint8)
        
        sorted_dets = sorted(detections, key=lambda d: d.mask.sum(), reverse=True)
        
        for det in sorted_dets:
            color = self._get_color(det.class_id, det.track_id)
            panoptic[det.mask.astype(bool)] = color
        
        return panoptic
    
    def create_instance(self, shape: Tuple[int, int], detections: List[Detection]) -> np.ndarray:
        h, w = shape
        instance = np.zeros((h, w), dtype=np.uint16)
        
        sorted_dets = sorted(detections, key=lambda d: d.mask.sum(), reverse=True)
        
        for det in sorted_dets:
            pixel = min(det.class_id, 255) * 256 + min(det.track_id, 255)
            instance[det.mask.astype(bool)] = pixel
        
        return instance


class VideoProcessor:
    """Process video with YOLO-World + FastSAM."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = YOLOWorldFastSAMSegmenter(self.config)
        self.visualizer = Visualizer(self.config)
        
        if self.config['tracking']['enabled']:
            self.tracker = SimpleTracker(self.config)
        else:
            self.tracker = None
        
        self.frame_interval = self.config['sampling']['frame_interval']
        self.start_frame = self.config['sampling']['start_frame']
        self.max_frames = self.config['sampling'].get('max_frames')
    
    def process(self):
        input_path = Path(self.config['paths']['input_video'])
        output_dir = Path(self.config['paths']['output_dir'])
        
        # Create output dirs
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
        print(f"Classes: {self.config.get('class_names', [])}")
        
        frame_indices = list(range(self.start_frame, total, self.frame_interval))
        if self.max_frames:
            frame_indices = frame_indices[:self.max_frames]
        
        print(f"Processing {len(frame_indices)} frames")
        print(f"Output: {output_dir}\n")
        
        img_fmt = self.config['output']['image_format']
        start_time = time.time()
        
        if self.tracker:
            self.tracker.reset()
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect + Segment
            detections = self.model(frame)
            
            # Track
            if self.tracker:
                detections = self.tracker.update(detections)
            else:
                for j, det in enumerate(detections):
                    det.track_id = j + 1
            
            # Save outputs
            frame_name = f"frame_{frame_idx:06d}"
            
            if 'overlay' in dirs:
                overlay = self.visualizer.create_overlay(frame, detections)
                cv2.imwrite(str(dirs['overlay'] / f"{frame_name}.{img_fmt}"), overlay)
            
            if 'panoptic' in dirs:
                panoptic = self.visualizer.create_panoptic(frame.shape[:2], detections)
                cv2.imwrite(str(dirs['panoptic'] / f"{frame_name}.{img_fmt}"), panoptic)
            
            if 'instance' in dirs:
                instance = self.visualizer.create_instance(frame.shape[:2], detections)
                cv2.imwrite(str(dirs['instance'] / f"{frame_name}.png"), instance)
            
            elapsed = time.time() - start_time
            fps_actual = (i + 1) / elapsed if elapsed > 0 else 0
            
            print(f"Frame {frame_idx:6d} | Dets: {len(detections):3d} | "
                  f"Progress: {i+1}/{len(frame_indices)} | FPS: {fps_actual:.2f}")
        
        cap.release()
        
        elapsed = time.time() - start_time
        print(f"\nDone! {len(frame_indices)} frames in {elapsed:.1f}s")
        print(f"Average FPS: {len(frame_indices)/elapsed:.2f}")
        print(f"Output: {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.yaml")
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLO-World + FastSAM Video Segmentation")
    print("=" * 60)
    print("Fast zero-shot segmentation (~50x faster than SAM)")
    print("=" * 60)
    
    processor = VideoProcessor(args.config)
    processor.process()


if __name__ == "__main__":
    main()
