#!/usr/bin/env python3
"""
YOLOE Text-Prompted Segmentation Pipeline
==========================================
Uses YOLOE with text prompts (RepRTA) for open-vocabulary detection + segmentation.
Single model handles both detection and mask generation.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import time
import warnings

warnings.filterwarnings('ignore')


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
    age: int = 0
    hits: int = 1
    misses: int = 0


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
        if not self.tracks or not detections:
            return np.zeros((len(self.tracks), len(detections)))
        
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                if track.class_id == det.class_id:
                    iou_matrix[i, j] = self._compute_iou(track.bbox, det.bbox)
        
        return iou_matrix
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            for track in self.tracks:
                track.misses += 1
                track.age += 1
            self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
            return []
        
        iou_matrix = self._compute_iou_matrix(detections)
        
        matched_tracks = set()
        matched_dets = set()
        
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break
            
            track_idx, det_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            
            self.tracks[track_idx].bbox = detections[det_idx].bbox
            self.tracks[track_idx].mask = detections[det_idx].mask
            self.tracks[track_idx].confidence = detections[det_idx].confidence
            self.tracks[track_idx].hits += 1
            self.tracks[track_idx].misses = 0
            self.tracks[track_idx].age += 1
            
            detections[det_idx].track_id = self.tracks[track_idx].track_id
            
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)
            
            iou_matrix[track_idx, :] = 0
            iou_matrix[:, det_idx] = 0
        
        num_existing_tracks = len(self.tracks)
        for track_idx in range(num_existing_tracks):
            if track_idx not in matched_tracks:
                self.tracks[track_idx].misses += 1
                self.tracks[track_idx].age += 1
        
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
                det.track_id = self.next_track_id
                self.next_track_id += 1
        
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        
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
        self.tracks = []
        self.next_track_id = 1


class YOLOESegmenter:
    """YOLOE text-prompted segmentation."""
    
    def __init__(self, config: dict):
        self.config = config
        self.conf_thresh = config['inference']['confidence_threshold']
        self.iou_thresh = config['inference']['iou_threshold']
        self.class_names = config.get('class_names', ['person', 'laptop'])
        
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
        
        # Import ultralytics
        from ultralytics import YOLOE
        
        # Load YOLOE model
        model_path = config['paths']['model']
        print(f"Loading YOLOE model: {model_path}")
        self.model = YOLOE(model_path)
        self.model.to(self.device)
        
        # Set text prompts (classes to detect)
        print(f"Setting text prompts: {self.class_names}")
        self.model.set_classes(self.class_names)
        
        print(f"Model loaded on device: {self.device}")
    
    def __call__(self, image: np.ndarray) -> List[Detection]:
        """Run YOLOE prompt-free segmentation on an image."""
        h, w = image.shape[:2]
        
        # Run YOLOE in prompt-free mode (no classes specified)
        results = self.model.predict(
            image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False
        )
        
        if len(results) == 0:
            return []
        
        result = results[0]
        
        # Get boxes
        if result.boxes is None or len(result.boxes) == 0:
            return []
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Get masks
        masks = result.masks
        mask_data = None
        if masks is not None:
            mask_data = masks.data.cpu().numpy()
        
        # Build detections
        detections = []
        
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
            
            # Get mask
            if mask_data is not None and i < len(mask_data):
                mask = mask_data[i]
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)
            else:
                # Fallback to box mask
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


class Visualizer:
    """Visualize segmentation results."""
    
    def __init__(self, config: dict):
        output_cfg = config.get('output', {})
        self.overlay_alpha = output_cfg.get('overlay_alpha', 0.6)
        self.show_labels = output_cfg.get('show_labels', True)
        self.show_confidence = output_cfg.get('show_confidence', False)
        self.show_boxes = output_cfg.get('show_boxes', False)
        self.colors = config.get('colors', [[230, 25, 75], [60, 180, 75], [255, 225, 25]])
    
    def draw_overlay(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        overlay = image.copy()
        
        for det in detections:
            color = self.colors[det.class_id % len(self.colors)]
            
            # Draw mask
            mask_area = det.mask > 0
            overlay[mask_area] = (
                overlay[mask_area] * (1 - self.overlay_alpha) + 
                np.array(color) * self.overlay_alpha
            ).astype(np.uint8)
            
            # Draw box
            if self.show_boxes:
                x1, y1, x2, y2 = det.bbox.astype(int)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if self.show_labels:
                label = det.class_name
                if self.show_confidence:
                    label += f" {det.confidence:.2f}"
                
                x1, y1 = int(det.bbox[0]), int(det.bbox[1])
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return overlay
    
    def create_panoptic_mask(self, shape: tuple, detections: List[Detection]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        for det in detections:
            color = self.colors[det.class_id % len(self.colors)]
            mask[det.mask > 0] = color
        return mask
    
    def create_instance_mask(self, shape: tuple, detections: List[Detection]) -> np.ndarray:
        """Create instance mask with unique ID per instance (8-bit).
        
        Each instance gets a unique value: 1, 2, 3, ... (0 = background)
        Values are scaled for visibility when viewing.
        """
        mask = np.zeros(shape[:2], dtype=np.uint8)
        n_det = len(detections)
        for i, det in enumerate(detections):
            # Scale values to be visible (spread across 0-255 range)
            if n_det > 0:
                instance_value = int((i + 1) * 255 / (n_det + 1))
            else:
                instance_value = i + 1
            mask[det.mask > 0] = instance_value
        return mask


class VideoProcessor:
    """Process video with YOLOE segmentation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.segmenter = YOLOESegmenter(self.config)
        self.tracker = SimpleTracker(self.config)
        self.visualizer = Visualizer(self.config)
        
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
        
        # Create output directories
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
            if not ret:
                continue
            
            # Run segmentation
            detections = self.segmenter(frame)
            
            # Apply tracking if enabled
            if self.tracking_enabled:
                detections = self.tracker.update(detections)
            
            # Save outputs
            frame_name = f"frame_{frame_idx:06d}"
            
            if 'overlay' in dirs:
                overlay = self.visualizer.draw_overlay(frame, detections)
                cv2.imwrite(str(dirs['overlay'] / f"{frame_name}.{img_fmt}"), overlay)
            
            if 'panoptic' in dirs:
                panoptic = self.visualizer.create_panoptic_mask(frame.shape, detections)
                cv2.imwrite(str(dirs['panoptic'] / f"{frame_name}.{img_fmt}"), panoptic)
            
            if 'instance' in dirs:
                instance = self.visualizer.create_instance_mask(frame.shape, detections)
                cv2.imwrite(str(dirs['instance'] / f"{frame_name}.png"), instance)
            
            # Get unique classes detected
            class_names = list(set(d.class_name for d in detections))
            
            elapsed = time.time() - start_time
            fps_actual = (i + 1) / elapsed if elapsed > 0 else 0
            
            print(f"Frame {frame_idx:6d} | Detections: {len(detections):3d} | "
                  f"Classes: {class_names} | FPS: {fps_actual:.2f}")
        
        cap.release()
        
        elapsed = time.time() - start_time
        print(f"\nDone! {len(frame_indices)} frames in {elapsed:.1f}s")
        print(f"Average FPS: {len(frame_indices)/elapsed:.2f}")
        print(f"Output: {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="YOLOE Prompt-Free Segmentation")
    parser.add_argument("--config", "-c", default="config.yaml")
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLOE Text-Prompted Segmentation")
    print("=" * 60)
    
    processor = VideoProcessor(args.config)
    processor.process()


if __name__ == "__main__":
    main()
