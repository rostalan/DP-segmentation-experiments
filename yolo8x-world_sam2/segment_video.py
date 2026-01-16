#!/usr/bin/env python3
"""
Zero-Shot Video Segmentation using YOLO-World + SAM
===================================================
Detect ANY object by text description and generate high-quality segmentation masks.

This approach combines:
- YOLO-World: Open-vocabulary object detection (detect any class by text)
- SAM (Segment Anything Model): High-quality mask generation from bounding boxes
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time


@dataclass
class Detection:
    """Represents a single detection with mask."""
    bbox: np.ndarray       # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    instance_id: int       # Unique instance ID for panoptic segmentation
    mask: np.ndarray       # Binary mask at original resolution
    track_id: int = -1     # Persistent track ID across frames


@dataclass 
class Track:
    """Represents a tracked object across frames."""
    track_id: int
    bbox: np.ndarray
    class_id: int
    class_name: str
    mask: np.ndarray
    confidence: float
    age: int = 0           # Frames since track was created
    hits: int = 1          # Number of frames with detections
    misses: int = 0        # Consecutive frames without detection


class SimpleTracker:
    """IoU-based tracker for temporal consistency."""
    
    def __init__(self, config: dict):
        tracking_cfg = config.get('tracking', {})
        self.iou_threshold = tracking_cfg.get('iou_threshold', 0.3)
        self.max_age = tracking_cfg.get('max_age', 5)
        self.min_hits = tracking_cfg.get('min_hits', 1)
        
        self.tracks: List[Track] = []
        self.next_track_id = 1
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def _compute_iou_matrix(self, detections: List[Detection]) -> np.ndarray:
        """Compute IoU matrix between tracks and detections."""
        if not self.tracks or not detections:
            return np.zeros((len(self.tracks), len(detections)))
        
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                # Only match same class
                if track.class_id == det.class_id:
                    iou_matrix[i, j] = self._compute_iou(track.bbox, det.bbox)
        return iou_matrix
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update tracks with new detections and return tracked detections."""
        iou_matrix = self._compute_iou_matrix(detections)
        
        matched_tracks = set()
        matched_dets = set()
        matches = []
        
        if iou_matrix.size > 0:
            while True:
                max_iou = iou_matrix.max()
                if max_iou < self.iou_threshold:
                    break
                
                idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                track_idx, det_idx = idx
                
                matches.append((track_idx, det_idx))
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)
                
                iou_matrix[track_idx, :] = 0
                iou_matrix[:, det_idx] = 0
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            det = detections[det_idx]
            track.bbox = det.bbox
            track.mask = det.mask
            track.confidence = det.confidence
            track.hits += 1
            track.misses = 0
            track.age += 1
        
        # Update unmatched tracks
        num_existing_tracks = len(self.tracks)
        for track_idx in range(num_existing_tracks):
            if track_idx not in matched_tracks:
                self.tracks[track_idx].misses += 1
                self.tracks[track_idx].age += 1
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                new_track = Track(
                    track_id=self.next_track_id,
                    bbox=det.bbox,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    mask=det.mask,
                    confidence=det.confidence
                )
                self.tracks.append(new_track)
                self.next_track_id += 1
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        
        # Build output detections
        output_detections = []
        for track in self.tracks:
            if track.hits >= self.min_hits:
                adjusted_conf = track.confidence * (0.9 ** track.misses)
                output_detections.append(Detection(
                    bbox=track.bbox,
                    confidence=adjusted_conf,
                    class_id=track.class_id,
                    class_name=track.class_name,
                    instance_id=track.track_id,
                    mask=track.mask,
                    track_id=track.track_id
                ))
        
        return output_detections
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_track_id = 1


class YOLOWorldSAMSegmenter:
    """Zero-shot segmentation using YOLO-World + SAM."""
    
    def __init__(self, config: dict):
        self.config = config
        self.class_names = config.get('class_names', ['object'])
        self.conf_thresh = config['yolo']['confidence_threshold']
        self.iou_thresh = config['yolo']['iou_threshold']
        
        # Determine device
        device = config['inference']['device']
        if device == 'auto':
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        
        # Import ultralytics here to avoid import errors if not installed
        from ultralytics import YOLOWorld, SAM
        
        # Load YOLO-World model
        yolo_path = config['paths']['yolo_model']
        print(f"Loading YOLO-World model: {yolo_path}")
        self.yolo = YOLOWorld(yolo_path)
        self.yolo.to(self.device)
        
        # Set custom classes for zero-shot detection
        print(f"Setting custom classes: {self.class_names}")
        self.yolo.set_classes(self.class_names)
        
        # Load SAM model
        sam_path = config['paths']['sam_model']
        print(f"Loading SAM model: {sam_path}")
        self.sam = SAM(sam_path)
        self.sam.to(self.device)
        
        print(f"Models loaded on device: {self.device}")
    
    def __call__(self, image: np.ndarray) -> List[Detection]:
        """Run zero-shot segmentation on an image."""
        # Run YOLO-World detection
        yolo_results = self.yolo.predict(
            image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False
        )
        
        if len(yolo_results) == 0 or len(yolo_results[0].boxes) == 0:
            return []
        
        result = yolo_results[0]
        boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4]
        confidences = result.boxes.conf.cpu().numpy()  # [N]
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # [N]
        
        if len(boxes) == 0:
            return []
        
        # Run SAM to generate masks from boxes
        sam_results = self.sam.predict(
            image,
            bboxes=boxes,
            verbose=False
        )
        
        # Build detections
        detections = []
        masks = sam_results[0].masks
        
        if masks is None:
            # Fallback: create box masks if SAM fails
            h, w = image.shape[:2]
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = map(int, box)
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 1
                
                detections.append(Detection(
                    bbox=box,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                    instance_id=i + 1,
                    mask=mask
                ))
        else:
            masks_data = masks.data.cpu().numpy()  # [N, H, W]
            
            for i, (box, conf, cls_id, mask) in enumerate(zip(boxes, confidences, class_ids, masks_data)):
                # Convert mask to uint8
                binary_mask = (mask > 0.5).astype(np.uint8)
                
                detections.append(Detection(
                    bbox=box,
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                    instance_id=i + 1,
                    mask=binary_mask
                ))
        
        return detections


class PanopticVisualizer:
    """Generate panoptic segmentation visualizations."""
    
    def __init__(self, config: dict):
        self.colors = [tuple(c) for c in config.get('colors', [(255, 0, 0)])]
        self.alpha = config['output']['overlay_alpha']
        self.show_labels = config['output']['show_labels']
        self.show_confidence = config['output']['show_confidence']
        self.class_names = config.get('class_names', [])
    
    def _get_instance_color(self, class_id: int, instance_id: int) -> Tuple[int, int, int]:
        """Generate unique color for each instance."""
        base_color = self.colors[class_id % len(self.colors)]
        offset = (instance_id * 37) % 60 - 30
        return tuple(max(0, min(255, c + offset)) for c in base_color)
    
    def create_overlay(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Create RGB image with mask overlay and labels."""
        output = image.copy()
        overlay = image.copy()
        
        for det in detections:
            color = self._get_instance_color(det.class_id, det.instance_id)
            mask_bool = det.mask.astype(bool)
            overlay[mask_bool] = color
            
            if self.show_labels:
                x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                label = f"{det.class_name}: {det.confidence:.2f}" if self.show_confidence else det.class_name
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(output, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0)
    
    def create_panoptic_mask(self, shape: Tuple[int, int], detections: List[Detection]) -> np.ndarray:
        """Create colored panoptic segmentation mask."""
        h, w = shape
        panoptic = np.zeros((h, w, 3), dtype=np.uint8)
        
        sorted_dets = sorted(detections, key=lambda d: d.mask.sum(), reverse=True)
        
        for det in sorted_dets:
            color = self._get_instance_color(det.class_id, det.instance_id)
            panoptic[det.mask.astype(bool)] = color
        
        return panoptic
    
    def create_instance_mask(self, shape: Tuple[int, int], detections: List[Detection]) -> np.ndarray:
        """Create instance ID mask (each instance has unique grayscale value)."""
        h, w = shape
        instance_mask = np.zeros((h, w), dtype=np.uint16)
        
        sorted_dets = sorted(detections, key=lambda d: d.mask.sum(), reverse=True)
        
        for det in sorted_dets:
            pixel_value = min(det.class_id, 255) * 256 + min(det.instance_id, 255)
            instance_mask[det.mask.astype(bool)] = pixel_value
        
        return instance_mask


class VideoFrameSegmenter:
    """Process video frames with zero-shot segmentation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = YOLOWorldSAMSegmenter(self.config)
        self.visualizer = PanopticVisualizer(self.config)
        
        # Initialize tracker if enabled
        self.use_tracking = self.config.get('tracking', {}).get('enabled', False)
        self.tracker = SimpleTracker(self.config) if self.use_tracking else None
        
        # Sampling settings
        self.frame_interval = self.config['sampling']['frame_interval']
        self.start_frame = self.config['sampling']['start_frame']
        self.max_frames = self.config['sampling'].get('max_frames')
    
    def process(self):
        """Process video and save individual frames."""
        input_path = Path(self.config['paths']['input_video'])
        output_dir = Path(self.config['paths']['output_dir'])
        
        # Create output subdirectories
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
        
        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nInput: {input_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Total frames: {total_frames}")
        print(f"Sampling: every {self.frame_interval} frames (starting at {self.start_frame})")
        print(f"Classes to detect: {self.config.get('class_names', [])}")
        
        # Calculate frames to process
        frames_to_process = list(range(self.start_frame, total_frames, self.frame_interval))
        if self.max_frames:
            frames_to_process = frames_to_process[:self.max_frames]
        
        print(f"Frames to process: {len(frames_to_process)}")
        print(f"Output directory: {output_dir}")
        if self.use_tracking:
            print("Tracking: ENABLED (temporal consistency)")
        print()
        
        img_fmt = self.config['output']['image_format']
        start_time = time.time()
        processed = 0
        
        # Reset tracker for new video
        if self.tracker:
            self.tracker.reset()
        
        for frame_idx in frames_to_process:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue
            
            # Run zero-shot segmentation
            detections = self.model(frame)
            
            # Apply tracking for temporal consistency
            if self.tracker:
                detections = self.tracker.update(detections)
            
            # Generate and save outputs
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
            
            processed += 1
            
            # Progress
            elapsed = time.time() - start_time
            fps_actual = processed / elapsed if elapsed > 0 else 0
            print(f"Frame {frame_idx:6d} | Detections: {len(detections):3d} | "
                  f"Progress: {processed}/{len(frames_to_process)} | FPS: {fps_actual:.2f}")
        
        cap.release()
        
        elapsed = time.time() - start_time
        print(f"\nComplete! Processed {processed} frames in {elapsed:.1f}s")
        print(f"Outputs saved to: {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Zero-Shot Video Segmentation (YOLO-World + SAM)")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Zero-Shot Video Segmentation (YOLO-World + SAM)")
    print("=" * 60)
    
    segmenter = VideoFrameSegmenter(args.config)
    segmenter.process()


if __name__ == "__main__":
    main()

