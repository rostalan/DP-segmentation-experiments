#!/usr/bin/env python3
"""
SAM2-Only Video Segmentation
============================
Uses SAM2 for automatic segmentation without a detection model.
SAM2 segments ALL objects in the image automatically.
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
    """Segmented object."""
    bbox: np.ndarray       # [x1, y1, x2, y2]
    confidence: float
    mask: np.ndarray       # Binary mask
    track_id: int = -1
    instance_id: int = 0


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
    
    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0
    
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
                
                # Use mask IoU for better matching
                iou = self._compute_mask_iou(det.mask, track['mask'])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_tid = tid
            
            if best_tid is not None:
                det.track_id = best_tid
                self.tracks[best_tid]['bbox'] = det.bbox.copy()
                self.tracks[best_tid]['mask'] = det.mask.copy()
                self.tracks[best_tid]['age'] = 0
                matched_track.add(best_tid)
                matched_det.add(id(det))
        
        for det in detections:
            if id(det) not in matched_det:
                det.track_id = self.next_id
                self.tracks[self.next_id] = {
                    'bbox': det.bbox.copy(),
                    'mask': det.mask.copy(),
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


class SAM2Segmenter:
    """SAM2 automatic segmentation."""
    
    def __init__(self, config: dict):
        self.config = config
        
        sam2_cfg = config.get('sam2', {})
        self.points_per_side = sam2_cfg.get('points_per_side', 32)
        self.pred_iou_thresh = sam2_cfg.get('pred_iou_thresh', 0.88)
        self.stability_score_thresh = sam2_cfg.get('stability_score_thresh', 0.95)
        self.min_mask_region_area = sam2_cfg.get('min_mask_region_area', 100)
        
        self.imgsz = config['inference'].get('imgsz', 640)
        
        # Device
        device = config['inference']['device']
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        from ultralytics import SAM
        
        # Load SAM2
        sam_path = config['paths']['sam_model']
        print(f"Loading SAM2: {sam_path}")
        self.sam = SAM(sam_path)
        
        print(f"SAM2 loaded on: {self.device}")
    
    def __call__(self, image: np.ndarray) -> List[Detection]:
        """Run automatic segmentation."""
        h, w = image.shape[:2]
        
        # Force resize to small size for speed
        target_size = self.imgsz
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Run SAM2 - simple call, let it process the small image
        results = self.sam.predict(
            image_small,
            verbose=False
        )
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                
                # Get confidence scores if available
                if hasattr(result.boxes, 'conf') and result.boxes.conf is not None:
                    confs = result.boxes.conf.cpu().numpy()
                else:
                    confs = np.ones(len(masks))
                
                for i, (mask, conf) in enumerate(zip(masks, confs)):
                    # Resize mask to original size
                    if scale != 1.0:
                        mask = cv2.resize(mask.astype(np.float32), (w, h),
                                         interpolation=cv2.INTER_NEAREST)
                    
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    
                    # Filter small masks
                    if mask_binary.sum() < self.min_mask_region_area:
                        continue
                    
                    # Compute bbox from mask
                    ys, xs = np.where(mask_binary > 0)
                    if len(xs) == 0:
                        continue
                    
                    bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
                    
                    detections.append(Detection(
                        bbox=bbox,
                        confidence=float(conf),
                        mask=mask_binary,
                        instance_id=i + 1
                    ))
        
        return detections


class Visualizer:
    """Create output visualizations."""
    
    def __init__(self, config: dict):
        self.colors = [tuple(c) for c in config.get('colors', [(255, 0, 0)])]
        self.alpha = config['output']['overlay_alpha']
        self.show_track = config['output'].get('show_track_id', True)
        self.show_conf = config['output'].get('show_confidence', False)
    
    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        return self.colors[track_id % len(self.colors)]
    
    def create_overlay(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        output = image.copy()
        overlay = image.copy()
        
        for det in detections:
            color = self._get_color(det.track_id)
            overlay[det.mask.astype(bool)] = color
            
            if self.show_track:
                x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                label = f"#{det.track_id}"
                if self.show_conf:
                    label += f" {det.confidence:.2f}"
                
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
            color = self._get_color(det.track_id)
            panoptic[det.mask.astype(bool)] = color
        
        return panoptic
    
    def create_instance(self, shape: Tuple[int, int], detections: List[Detection]) -> np.ndarray:
        h, w = shape
        instance = np.zeros((h, w), dtype=np.uint16)
        
        sorted_dets = sorted(detections, key=lambda d: d.mask.sum(), reverse=True)
        
        for det in sorted_dets:
            instance[det.mask.astype(bool)] = det.track_id
        
        return instance


class VideoProcessor:
    """Process video with SAM2."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = SAM2Segmenter(self.config)
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
            
            # Segment
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
            
            print(f"Frame {frame_idx:6d} | Objects: {len(detections):3d} | "
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
    print("SAM2-Only Video Segmentation")
    print("=" * 60)
    print("Automatic segmentation of ALL objects")
    print("=" * 60)
    
    processor = VideoProcessor(args.config)
    processor.process()


if __name__ == "__main__":
    main()

