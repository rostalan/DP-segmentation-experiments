#!/usr/bin/env python3
"""
Panoptic Video Segmentation using YOLO26 ONNX Model
====================================================
Processes selected frames from input video and generates panoptic segmentation outputs.
"""

import cv2
import numpy as np
import onnxruntime as ort
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
    instance_id: int       # Unique instance ID for panoptic segmentation
    mask: np.ndarray       # Binary mask at original resolution
    track_id: int = -1     # Persistent track ID across frames


@dataclass 
class Track:
    """Represents a tracked object across frames."""
    track_id: int
    bbox: np.ndarray
    class_id: int
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
        self.max_age = tracking_cfg.get('max_age', 5)  # Max frames to keep lost track
        self.min_hits = tracking_cfg.get('min_hits', 1)  # Min hits before track is confirmed
        
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
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(detections)
        
        # Greedy matching
        matched_tracks = set()
        matched_dets = set()
        matches = []
        
        if iou_matrix.size > 0:
            # Sort by IoU descending
            while True:
                max_iou = iou_matrix.max()
                if max_iou < self.iou_threshold:
                    break
                
                idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                track_idx, det_idx = idx
                
                matches.append((track_idx, det_idx))
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)
                
                # Invalidate row and column
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
        
        # Update unmatched tracks (increment misses) - do this BEFORE adding new tracks
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
                    mask=det.mask,
                    confidence=det.confidence
                )
                self.tracks.append(new_track)
                self.next_track_id += 1
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        
        # Build output detections from confirmed tracks
        # Include tracks that are "coasting" (recently lost but within max_age)
        output_detections = []
        for track in self.tracks:
            if track.hits >= self.min_hits:
                # Fade confidence for coasting tracks
                adjusted_conf = track.confidence * (0.9 ** track.misses)
                output_detections.append(Detection(
                    bbox=track.bbox,
                    confidence=adjusted_conf,
                    class_id=track.class_id,
                    instance_id=track.track_id,
                    mask=track.mask,
                    track_id=track.track_id
                ))
        
        return output_detections
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_track_id = 1


class YOLOPanopticSegmenter:
    """YOLO26 Panoptic Segmentation pipeline."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_path = config['paths']['model']
        self.input_size = config['model']['input_size']
        self.conf_thresh = config['model']['confidence_threshold']
        self.iou_thresh = config['model']['iou_threshold']
        self.mask_thresh = config['model']['mask_threshold']
        self.class_names = config.get('class_names', [])
        
        # Initialize ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
            if config['inference']['device'] == 'cuda' else ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        print(f"Model loaded: {self.model_path}")
        print(f"Input: {self.input_name}, Outputs: {self.output_names}")
        print(f"Providers: {self.session.get_providers()}")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for model input with letterboxing."""
        orig_h, orig_w = image.shape[:2]
        
        # Calculate scale to maintain aspect ratio
        scale = min(self.input_size / orig_w, self.input_size / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image (letterbox)
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_w, pad_h = (self.input_size - new_w) // 2, (self.input_size - new_h) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # Convert to model input format: NCHW, float32, normalized
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, 0)
        
        return blob, scale, (pad_w, pad_h)
    
    def postprocess(
        self, 
        outputs: List[np.ndarray], 
        orig_shape: Tuple[int, int],
        scale: float,
        padding: Tuple[int, int]
    ) -> List[Detection]:
        """Process model outputs into panoptic detections with unique instance IDs."""
        orig_h, orig_w = orig_shape
        pad_w, pad_h = padding
        
        # Parse outputs based on YOLO segmentation format
        # output0: [1, 300, 38] = [batch, num_detections, features]
        # output1: [1, 32, 160, 160] = [batch, mask_channels, mask_h, mask_w]
        if len(outputs) >= 2:
            det_output = outputs[0]
            proto_output = outputs[1]
        else:
            det_output = outputs[0]
            proto_output = None
        
        det_output = det_output[0]  # Remove batch dimension: [300, 38]
        
        # Model output format: [x1, y1, x2, y2, confidence, class_id, 32 mask_coeffs]
        # Coordinates are in input image space (640x640 with padding)
        
        detections = []
        instance_counter = 1  # Start instance IDs from 1 (0 = background)
        
        for det in det_output:
            # Parse detection: already in corner format (x1, y1, x2, y2)
            x1_raw, y1_raw, x2_raw, y2_raw = det[:4]
            confidence = float(det[4])
            class_id = int(det[5])
            mask_coeffs = det[6:] if len(det) > 6 else None
            
            if confidence < self.conf_thresh:
                continue
            
            # Convert from padded input space to original image space
            x1 = (x1_raw - pad_w) / scale
            y1 = (y1_raw - pad_h) / scale
            x2 = (x2_raw - pad_w) / scale
            y2 = (y2_raw - pad_h) / scale
            
            # Clip to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Generate instance mask
            if proto_output is not None and mask_coeffs is not None:
                mask = self._generate_mask(
                    mask_coeffs, proto_output[0],
                    (x1, y1, x2, y2), orig_shape, scale, padding
                )
            else:
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                mask[int(y1):int(y2), int(x1):int(x2)] = 1
            
            detections.append(Detection(
                bbox=np.array([x1, y1, x2, y2]),
                confidence=confidence,
                class_id=class_id,
                instance_id=instance_counter,
                mask=mask
            ))
            instance_counter += 1
        
        # Apply NMS
        if detections:
            detections = self._nms(detections)
            # Reassign instance IDs after NMS
            for i, det in enumerate(detections):
                det.instance_id = i + 1
        
        return detections
    
    def _generate_mask(
        self,
        mask_coeffs: np.ndarray,
        proto: np.ndarray,
        bbox: Tuple[float, float, float, float],
        orig_shape: Tuple[int, int],
        scale: float,
        padding: Tuple[int, int]
    ) -> np.ndarray:
        """Generate instance mask from prototype and coefficients."""
        orig_h, orig_w = orig_shape
        pad_w, pad_h = padding
        c, mask_h, mask_w = proto.shape
        
        mask_coeffs = mask_coeffs[:c]
        mask = np.sum(mask_coeffs.reshape(-1, 1, 1) * proto, axis=0)
        mask = 1 / (1 + np.exp(-mask))  # Sigmoid
        
        # Resize to input size then crop padding
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        
        # Calculate valid region after padding removal
        valid_h = self.input_size - 2 * pad_h
        valid_w = self.input_size - 2 * pad_w
        
        if valid_h > 0 and valid_w > 0:
            mask = mask[pad_h:pad_h + valid_h, pad_w:pad_w + valid_w]
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        mask = (mask > self.mask_thresh).astype(np.uint8)
        
        # Crop to bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cropped = np.zeros_like(mask)
        cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        
        return cropped
    
    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """Non-Maximum Suppression."""
        if not detections:
            return []
        
        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            order = order[np.where(iou <= self.iou_thresh)[0] + 1]
        
        return [detections[i] for i in keep]
    
    def __call__(self, image: np.ndarray) -> List[Detection]:
        """Run panoptic segmentation inference."""
        orig_shape = image.shape[:2]
        blob, scale, padding = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        return self.postprocess(outputs, orig_shape, scale, padding)


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
        # Add variation based on instance ID
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
                x1, y1 = det.bbox[:2].astype(int)
                name = self.class_names[det.class_id] if det.class_id < len(self.class_names) else f"cls_{det.class_id}"
                label = f"{name}: {det.confidence:.2f}" if self.show_confidence else name
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(output, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0)
    
    def create_panoptic_mask(self, shape: Tuple[int, int], detections: List[Detection]) -> np.ndarray:
        """Create colored panoptic segmentation mask."""
        h, w = shape
        panoptic = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Sort by area (largest first) so smaller objects appear on top
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
            # Encode class_id and instance_id: class_id * 256 + instance_id
            # This fits in uint16 (max 65535) for up to 255 classes and 255 instances
            pixel_value = min(det.class_id, 255) * 256 + min(det.instance_id, 255)
            instance_mask[det.mask.astype(bool)] = pixel_value
        
        return instance_mask


class VideoFrameSegmenter:
    """Process video frames with panoptic segmentation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = YOLOPanopticSegmenter(self.config)
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
            
            # Run panoptic segmentation
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
                cv2.imwrite(str(dirs['instance'] / f"{frame_name}.png"), instance)  # Always PNG for 16-bit
            
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
    parser = argparse.ArgumentParser(description="Panoptic Video Segmentation with YOLO26")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLO26 Panoptic Video Segmentation")
    print("=" * 60)
    
    segmenter = VideoFrameSegmenter(args.config)
    segmenter.process()


if __name__ == "__main__":
    main()

