#!/usr/bin/env python3
"""
Zero-Shot Video Segmentation using YOLO-World + SAM
===================================================
Detect ANY object by text description and generate high-quality segmentation masks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from typing import List, Tuple
import time
from collections import defaultdict, deque

from lib.detection import Detection
from lib.visualizer import Visualizer
from lib.config import load_config, merge_global_config, get_colors


class YOLOWorldSAMSegmenter:
    """Zero-shot segmentation using YOLO-World + SAM."""
    
    def __init__(self, config: dict):
        self.config = config
        self.class_names = config.get('class_names', ['object'])
        self.conf_thresh = config['yolo']['confidence_threshold']
        self.iou_thresh = config['yolo']['iou_threshold']
        
        tracking_cfg = config.get('tracking', {})
        self.tracking_enabled = tracking_cfg.get('enabled', False)
        self.tracker_type = tracking_cfg.get('tracker_type', 'bytetrack.yaml')
        self.persist = tracking_cfg.get('persist', True)
        
        device = config['inference']['device']
        if device == 'auto':
            import torch
            if torch.cuda.is_available(): device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): device = 'mps'
            else: device = 'cpu'
        self.device = device
        
        from ultralytics import YOLOWorld, SAM
        
        yolo_path = config['paths']['yolo_world']
        print(f"Loading YOLO-World model: {yolo_path}")
        self.yolo = YOLOWorld(yolo_path)
        self.yolo.to(self.device)
        self.yolo.set_classes(self.class_names)
        
        sam_path = config['paths']['sam2']
        print(f"Loading SAM model: {sam_path}")
        self.sam = SAM(sam_path)
        self.sam.to(self.device)
        
        print(f"Models loaded on device: {self.device}")
    
    def __call__(self, image: np.ndarray) -> List[Detection]:
        if self.tracking_enabled:
            yolo_results = self.yolo.track(
                image,
                persist=self.persist,
                tracker=self.tracker_type,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                verbose=False
            )
        else:
            yolo_results = self.yolo.predict(
                image,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                verbose=False
            )
        
        if len(yolo_results) == 0 or len(yolo_results[0].boxes) == 0:
            return []
        
        result = yolo_results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        track_ids = None
        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().numpy()
        
        if len(boxes) == 0:
            return []
        
        sam_results = self.sam.predict(
            image,
            bboxes=boxes,
            verbose=False
        )
        
        detections = []
        masks = sam_results[0].masks
        
        if masks is None:
            h, w = image.shape[:2]
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                mask = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = map(int, box)
                mask[y1:y2, x1:x2] = 1
                
                tid = int(track_ids[i]) if track_ids is not None else -1
                
                detections.append(Detection(
                    bbox=box,
                    confidence=float(conf),
                    mask=mask,
                    class_id=int(cls_id),
                    class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                    instance_id=tid if tid != -1 else (i + 1),
                    track_id=tid
                ))
        else:
            masks_data = masks.data.cpu().numpy()
            
            for i, (box, conf, cls_id, mask) in enumerate(zip(boxes, confidences, class_ids, masks_data)):
                binary_mask = (mask > 0.5).astype(np.uint8)
                tid = int(track_ids[i]) if track_ids is not None else -1
                
                detections.append(Detection(
                    bbox=box,
                    confidence=float(conf),
                    mask=binary_mask,
                    class_id=int(cls_id),
                    class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                    instance_id=tid if tid != -1 else (i + 1),
                    track_id=tid
                ))
        
        return detections


def _track_color(track_id: int, colors: list) -> Tuple[int, int, int]:
    base = colors[track_id % len(colors)]
    offset = (track_id * 37) % 60 - 30
    return tuple(max(0, min(255, c + offset)) for c in base)


def draw_track_trails(image: np.ndarray, track_history: dict, colors: list, trail_thickness: int = 2) -> np.ndarray:
    output = image.copy()
    for track_id, points in track_history.items():
        if len(points) < 2:
            continue
        color = _track_color(track_id, colors)
        for i in range(1, len(points)):
            cv2.line(output, points[i - 1], points[i], color, trail_thickness)
    return output


class VideoFrameSegmenter:
    def __init__(self, config_path: str = "config.yaml"):
        script_dir = Path(__file__).resolve().parent
        self.config = load_config(config_path)
        merge_global_config(self.config, script_dir)
        colors = get_colors(self.config)

        self.model = YOLOWorldSAMSegmenter(self.config)
        self.visualizer = Visualizer.from_config(self.config, colors)
        self.colors = [tuple(c) for c in colors]
        
        self.frame_interval = self.config['sampling']['frame_interval']
        self.start_frame = self.config['sampling']['start_frame']
        self.max_frames = self.config['sampling'].get('max_frames')
        self.track_history = defaultdict(lambda: deque(maxlen=50))
        self.trail_thickness = 2
    
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
            raise ValueError(f"Could not open video: {input_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nInput: {input_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Total frames: {total_frames}")
        
        frames_to_process = list(range(self.start_frame, total_frames, self.frame_interval))
        if self.max_frames:
            frames_to_process = frames_to_process[:self.max_frames]
        
        print(f"Frames to process: {len(frames_to_process)}")
        
        img_fmt = self.config['output']['image_format']
        start_time = time.time()
        processed = 0
        
        for frame_idx in frames_to_process:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            detections = self.model(frame)
            
            for det in detections:
                if det.track_id == -1: continue
                x1, y1, x2, y2 = det.bbox.astype(int)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                self.track_history[det.track_id].append((cx, cy))

            frame_name = f"frame_{frame_idx:06d}"
            
            if 'overlay' in dirs:
                overlay = self.visualizer.create_overlay(frame, detections)
                overlay = draw_track_trails(overlay, self.track_history, self.colors, self.trail_thickness)
                cv2.imwrite(str(dirs['overlay'] / f"{frame_name}.{img_fmt}"), overlay)
            
            if 'panoptic' in dirs:
                panoptic = self.visualizer.create_panoptic_mask(frame.shape[:2], detections)
                cv2.imwrite(str(dirs['panoptic'] / f"{frame_name}.{img_fmt}"), panoptic)
            
            if 'instance' in dirs:
                instance = self.visualizer.create_instance_mask(frame.shape[:2], detections)
                cv2.imwrite(str(dirs['instance'] / f"{frame_name}.png"), instance)

            frame_obj_dir = output_dir / frame_name
            frame_obj_dir.mkdir(parents=True, exist_ok=True)
            for obj_idx, det in enumerate(detections):
                mask_bool = det.mask.astype(bool)
                obj_img = np.zeros_like(frame)
                obj_img[mask_bool] = frame[mask_bool]
                safe_class = det.class_name.replace(" ", "_").replace("/", "_")
                obj_name = f"obj_{obj_idx:02d}_{safe_class}"
                if det.track_id != -1: obj_name += f"_{det.track_id}"
                cv2.imwrite(str(frame_obj_dir / f"{obj_name}.{img_fmt}"), obj_img)
            
            processed += 1
            elapsed = time.time() - start_time
            fps_actual = processed / elapsed if elapsed > 0 else 0
            class_names = sorted({det.class_name for det in detections})
            class_str = ", ".join(class_names) if class_names else "none"
            print(f"Frame {frame_idx:6d} | Dets: {len(detections):3d} | Classes: {class_str} | FPS: {fps_actual:.2f}")
        
        cap.release()
        elapsed = time.time() - start_time
        print(f"\nComplete! Processed {processed} frames in {elapsed:.1f}s")
        print(f"Outputs saved to: {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.yaml")
    args = parser.parse_args()
    
    segmenter = VideoFrameSegmenter(args.config)
    segmenter.process()


if __name__ == "__main__":
    main()
