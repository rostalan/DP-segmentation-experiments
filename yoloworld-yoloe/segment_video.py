#!/usr/bin/env python3
"""
YOLO-World + YOLOE Segmentation Pipeline
=========================================
Combines YOLO-World (open-vocabulary detection) with YOLOE (efficient segmentation).

- YOLO-World: Detects objects using text prompts (any description works)
- YOLOE: Generates high-quality segmentation masks from bounding boxes
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
import time
import warnings
from dataclasses import dataclass

warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLOWorld, YOLOE
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics>=8.3.0")
    exit(1)


@dataclass
class Detection:
    """Single detection with mask."""
    bbox: np.ndarray        # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    mask: np.ndarray        # Binary mask (H, W)
    instance_id: int = 0


class YOLOWorldYOLOESegmenter:
    """YOLO-World detection + YOLOE segmentation pipeline."""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Device
        device = config['inference']['device']
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Load YOLO-World for detection
        yolo_world_path = config['paths']['yolo_world_model']
        print(f"Loading YOLO-World: {yolo_world_path}")
        self.yolo_world = YOLOWorld(yolo_world_path)
        self.yolo_world.to(self.device)
        
        # Set custom classes for open-vocabulary detection
        self.class_names = config.get('custom_classes', ['object'])
        print(f"Setting classes: {self.class_names}")
        self.yolo_world.set_classes(self.class_names)
        
        # Load YOLOE for segmentation (with same text prompts)
        yoloe_path = config['paths']['yoloe_model']
        print(f"Loading YOLOE: {yoloe_path}")
        self.yoloe = YOLOE(yoloe_path)
        self.yoloe.to(self.device)
        
        # Set text prompts for YOLOE too
        print(f"Setting YOLOE classes: {self.class_names}")
        self.yoloe.set_classes(self.class_names)
        
        # Detection settings
        yolo_cfg = config.get('yolo_world', {})
        self.det_conf = yolo_cfg.get('confidence_threshold', 0.25)
        self.det_iou = yolo_cfg.get('iou_threshold', 0.45)
        self.det_imgsz = yolo_cfg.get('imgsz', 640)
        
        # Segmentation settings
        yoloe_cfg = config.get('yoloe', {})
        self.seg_imgsz = yoloe_cfg.get('imgsz', 640)
        self.seg_conf = yoloe_cfg.get('conf', 0.1)  # Lower for more masks
        
        print(f"Models loaded on: {self.device}")
        print(f"Classes: {len(self.class_names)}")
    
    def __call__(self, image: np.ndarray) -> list[Detection]:
        """Run detection + segmentation pipeline."""
        h, w = image.shape[:2]
        
        # Step 1: Detect objects with YOLO-World (for boxes + class names)
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
        
        # Step 2: Run YOLOE on FULL image with text prompts to get masks
        seg_results = self.yoloe.predict(
            image,
            conf=self.seg_conf,
            imgsz=self.seg_imgsz,
            verbose=False
        )
        
        # Get YOLOE masks and boxes
        yoloe_masks = None
        yoloe_boxes = None
        if seg_results and seg_results[0].masks is not None:
            yoloe_masks = seg_results[0].masks.data.cpu().numpy()
            yoloe_boxes = seg_results[0].boxes.xyxy.cpu().numpy()
        
        # Step 3: Match YOLOE masks to YOLO-World boxes
        detections = []
        used_masks = set()
        
        for i, (bbox, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
            
            # Filter large "person" detections (artifacts)
            # Hands are detected as "person" but are small. Background artifacts are huge.
            if cls_name == "person":
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > 0.2 * (w * h):  # Filter if > 20% of image
                    continue

            mask = np.zeros((h, w), dtype=np.uint8)
            
            if yoloe_masks is not None and yoloe_boxes is not None:
                # Find best matching YOLOE mask by IoU
                best_iou = 0
                best_mask_idx = -1
                
                for m_idx, yoloe_box in enumerate(yoloe_boxes):
                    if m_idx in used_masks:
                        continue
                    iou = self._compute_iou(bbox, yoloe_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_mask_idx = m_idx
                
                if best_mask_idx >= 0 and best_iou > 0.2:
                    # Use the matching YOLOE mask
                    yoloe_mask = yoloe_masks[best_mask_idx]
                    yoloe_mask = cv2.resize(yoloe_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    mask = (yoloe_mask > 0.5).astype(np.uint8)
                    used_masks.add(best_mask_idx)
            
            # Fallback: use bounding box as mask
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
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0


class Visualizer:
    """Visualize segmentation results."""
    
    def __init__(self, config: dict):
        self.config = config
        output_cfg = config.get('output', {})
        
        self.overlay_alpha = output_cfg.get('overlay_alpha', 0.5)
        self.show_labels = output_cfg.get('show_labels', True)
        self.show_confidence = output_cfg.get('show_confidence', True)
        self.show_boxes = output_cfg.get('show_boxes', True)
        
        self.colors = config.get('colors', [
            [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200]
        ])
    
    def draw_overlay(self, image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw masks, boxes, and labels on image."""
        overlay = image.copy()
        
        for det in detections:
            color = self.colors[det.class_id % len(self.colors)]
            
            # Draw mask
            mask_colored = np.zeros_like(image)
            mask_colored[det.mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1, mask_colored, self.overlay_alpha, 0)
            
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
    
    def create_panoptic_mask(self, shape: tuple, detections: list[Detection]) -> np.ndarray:
        """Create colored panoptic mask."""
        mask = np.zeros(shape, dtype=np.uint8)
        
        for det in detections:
            color = self.colors[det.class_id % len(self.colors)]
            mask[det.mask > 0] = color
        
        return mask
    
    def create_instance_mask(self, shape: tuple, detections: list[Detection]) -> np.ndarray:
        """Create instance mask with unique ID per instance (8-bit).
        
        Each instance gets a unique value.
        Draws larger masks first so smaller masks (fine details) appear on top.
        """
        mask = np.zeros(shape[:2], dtype=np.uint8)
        n_det = len(detections)
        
        # Sort by area (descending) so large masks are drawn first, 
        # and smaller masks (like hands) are drawn on top.
        sorted_dets = sorted(detections, key=lambda x: np.sum(x.mask > 0), reverse=True)
        
        for i, det in enumerate(sorted_dets):
            # Scale values to be visible (spread across 0-255 range)
            if n_det > 0:
                instance_value = int((i + 1) * 255 / (n_det + 1))
            else:
                instance_value = i + 1
            mask[det.mask > 0] = instance_value
        return mask


class VideoProcessor:
    """Process video with YOLO-World + YOLOE pipeline."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = YOLOWorldYOLOESegmenter(self.config)
        self.visualizer = Visualizer(self.config)
        
        self.frame_interval = self.config['sampling']['frame_interval']
        self.start_frame = self.config['sampling']['start_frame']
        self.max_frames = self.config['sampling'].get('max_frames')
    
    def process(self):
        input_path = Path(self.config['paths']['input_video'])
        output_dir = Path(self.config['paths']['output_dir'])
        output_cfg = self.config.get('output', {})
        
        # Create output directories
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
            if not ret:
                continue
            
            # Run pipeline
            detections = self.model(frame)
            
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
            
            elapsed = time.time() - start_time
            fps_actual = (i + 1) / elapsed if elapsed > 0 else 0
            
            print(f"Frame {frame_idx:6d} | Detections: {len(detections):3d} | "
                  f"Progress: {i+1}/{len(frame_indices)} | FPS: {fps_actual:.2f}")
        
        cap.release()
        
        elapsed = time.time() - start_time
        print(f"\nDone! {len(frame_indices)} frames in {elapsed:.1f}s")
        print(f"Average FPS: {len(frame_indices)/elapsed:.2f}")
        print(f"Output: {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="YOLO-World + YOLOE Segmentation")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file")
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLO-World + YOLOE Segmentation Pipeline")
    print("=" * 60)
    
    processor = VideoProcessor(args.config)
    processor.process()


if __name__ == "__main__":
    main()

