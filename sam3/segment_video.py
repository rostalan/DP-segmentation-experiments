#!/usr/bin/env python3
"""
YOLO11-World + SAM3 Hybrid Video Segmentation
==============================================
Pipeline:
1. YOLO11-World - Open-vocabulary detection (text prompts → bounding boxes)
2. SAM3 (Image mode) - High-quality segmentation from boxes
3. IoU Tracker - Temporal consistency

CPU-friendly: Processes one frame at a time, low memory usage.
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
    """Detection from YOLO."""
    bbox: np.ndarray      # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    mask: Optional[np.ndarray] = None
    track_id: int = -1


class YOLO11WorldDetector:
    """YOLO11-World open-vocabulary detector."""
    
    def __init__(self, model_path: str, config: dict):
        from ultralytics import YOLO
        
        self.conf_thresh = config['yolo']['confidence_threshold']
        self.iou_thresh = config['yolo']['iou_threshold']
        self.custom_classes = config.get('custom_classes', ['object'])
        
        # Load YOLO11-World model
        print(f"Loading YOLO11-World from: {model_path}")
        self.model = YOLO(model_path)
        
        # Set custom classes for open-vocabulary detection
        self.model.set_classes(self.custom_classes)
        print(f"Custom classes set: {self.custom_classes}")
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection and return bounding boxes."""
        results = self.model(
            image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False
        )
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    class_id = int(cls_ids[i])
                    class_name = self.custom_classes[class_id] if class_id < len(self.custom_classes) else f"class_{class_id}"
                    
                    detections.append(Detection(
                        bbox=boxes[i],
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(confs[i])
                    ))
        
        return detections


class SAM3Segmenter:
    """SAM3 image segmentation from bounding boxes - processes each box individually."""
    
    def __init__(self, config: dict):
        self.threshold = config['sam3'].get('threshold', 0.5)
        self.mask_threshold = config['sam3'].get('mask_threshold', 0.5)
        
        device = config['inference']['device']
        
        print("Loading SAM3 model from HuggingFace...")
        
        try:
            from transformers import Sam3TrackerProcessor, Sam3TrackerModel
            import torch
            
            self.torch = torch
            self.device = device if device != "cpu" else "cpu"
            self.dtype = torch.float32
            
            # Use Sam3Tracker for box-prompted segmentation (more reliable than Sam3Model for boxes)
            self.model = Sam3TrackerModel.from_pretrained("facebook/sam3")
            self.processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
            
            if self.device != "cpu":
                self.model = self.model.to(self.device)
            
            print(f"SAM3 Tracker loaded on {self.device}")
            self.available = True
            self.use_tracker = True
            
        except Exception as e:
            print(f"Sam3Tracker not available: {e}, trying Sam3Model...")
            try:
                from transformers import Sam3Processor, Sam3Model
                import torch
                
                self.torch = torch
                self.device = device if device != "cpu" else "cpu"
                self.dtype = torch.float32
                
                self.model = Sam3Model.from_pretrained("facebook/sam3")
                self.processor = Sam3Processor.from_pretrained("facebook/sam3")
                
                if self.device != "cpu":
                    self.model = self.model.to(self.device)
                
                print(f"SAM3 Model loaded on {self.device}")
                self.available = True
                self.use_tracker = False
                
            except Exception as e2:
                print(f"SAM3 not available: {e2}")
                print("Falling back to simple mask from bounding box")
                self.available = False
                self.use_tracker = False
    
    def _create_box_mask(self, h: int, w: int, bbox: np.ndarray) -> np.ndarray:
        """Create a simple rectangular mask from bounding box."""
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = 1
        return mask
    
    def _segment_single_box_tracker(self, pil_image, bbox: list, h: int, w: int) -> np.ndarray:
        """Segment a single box using Sam3Tracker."""
        # Correct format: [[box]] - 3 levels: [image_level, box_level, coordinates]
        input_boxes = [[bbox]]
        
        inputs = self.processor(
            images=pil_image,
            input_boxes=input_boxes,
            return_tensors="pt"
        )
        
        if self.device != "cpu":
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                     for k, v in inputs.items()}
        
        with self.torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        
        # Get mask from pred_masks: [batch, num_objects, num_masks, H, W]
        pred_masks = outputs.pred_masks
        
        if pred_masks is not None and pred_masks.numel() > 0:
            # Post-process to original size
            masks = self.processor.post_process_masks(
                pred_masks.cpu(), 
                inputs['original_sizes']
            )[0]  # [num_objects, num_masks, H, W]
            
            # Take the first mask
            mask = masks[0, 0].numpy()  # [H, W] - boolean
            
            return mask.astype(np.uint8)
        
        return None
    
    def _segment_single_box_model(self, pil_image, bbox: list, h: int, w: int) -> np.ndarray:
        """Segment a single box using Sam3Model."""
        # Format for Sam3Model: [[bbox]] for single image
        input_boxes = [[bbox]]
        input_boxes_labels = [[1]]  # Positive
        
        inputs = self.processor(
            images=pil_image,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt"
        )
        
        if self.device != "cpu":
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                     for k, v in inputs.items()}
        
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process to get mask
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=[[h, w]]
        )[0]
        
        if 'masks' in results and len(results['masks']) > 0:
            mask = results['masks'][0].cpu().numpy()
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h))
            return (mask > 0.5).astype(np.uint8)
        
        return None
    
    def segment(self, image: np.ndarray, detections: List[Detection]) -> List[Detection]:
        """Generate masks for detections using SAM3 - one box at a time."""
        if not detections:
            return detections
        
        h, w = image.shape[:2]
        
        if not self.available:
            for det in detections:
                det.mask = self._create_box_mask(h, w, det.bbox)
            return detections
        
        from PIL import Image
        
        # Convert to PIL once
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Process each detection individually for reliable mask matching
        for det in detections:
            try:
                bbox = det.bbox.tolist()
                
                if self.use_tracker:
                    mask = self._segment_single_box_tracker(pil_image, bbox, h, w)
                else:
                    mask = self._segment_single_box_model(pil_image, bbox, h, w)
                
                if mask is not None:
                    det.mask = mask
                else:
                    det.mask = self._create_box_mask(h, w, det.bbox)
                    
            except Exception as e:
                # Fallback to box mask for this detection
                det.mask = self._create_box_mask(h, w, det.bbox)
        
        return detections


class SimpleTracker:
    """IoU-based tracker for temporal consistency."""
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
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
            if det.mask is None:
                continue
            
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
        
        sorted_dets = sorted(detections, key=lambda d: d.mask.sum() if d.mask is not None else 0, reverse=True)
        
        for det in sorted_dets:
            if det.mask is None:
                continue
            color = self._get_color(det.class_id, det.track_id)
            panoptic[det.mask.astype(bool)] = color
        
        return panoptic
    
    def create_instance(self, shape: Tuple[int, int], detections: List[Detection]) -> np.ndarray:
        h, w = shape
        instance = np.zeros((h, w), dtype=np.uint16)
        
        sorted_dets = sorted(detections, key=lambda d: d.mask.sum() if d.mask is not None else 0, reverse=True)
        
        for det in sorted_dets:
            if det.mask is None:
                continue
            pixel = min(det.class_id, 255) * 256 + min(det.track_id, 255)
            instance[det.mask.astype(bool)] = pixel
        
        return instance


class VideoProcessor:
    """Main video processing pipeline."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        yolo_path = self.config['paths']['yolo_model']
        self.detector = YOLO11WorldDetector(yolo_path, self.config)
        self.segmenter = SAM3Segmenter(self.config)
        self.visualizer = Visualizer(self.config)
        
        if self.config['tracking']['enabled']:
            self.tracker = SimpleTracker(
                iou_threshold=self.config['tracking']['iou_threshold'],
                max_age=self.config['tracking']['max_age']
            )
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
        
        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nInput: {input_path}")
        print(f"Resolution: {w}x{h}, FPS: {fps:.2f}, Frames: {total}")
        print(f"Custom classes: {self.config.get('custom_classes', [])}")
        
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
            
            # 1. Detect with YOLO11-World
            detections = self.detector.detect(frame)
            
            # 2. Segment with SAM3
            detections = self.segmenter.segment(frame, detections)
            
            # 3. Track
            if self.tracker:
                detections = self.tracker.update(detections)
            else:
                for j, det in enumerate(detections):
                    det.track_id = j + 1
            
            # 4. Save outputs
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
        print(f"Output: {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.yaml")
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLO8-World + SAM3 Hybrid Video Segmentation")
    print("=" * 60)
    print("Pipeline: YOLO8-World (detection) → SAM3 (segmentation)")
    print("Open-vocabulary: Detect ANY object by text description")
    print("CPU-friendly: Frame-by-frame processing")
    print("=" * 60)
    
    processor = VideoProcessor(args.config)
    processor.process()


if __name__ == "__main__":
    main()
