#!/usr/bin/env python3
"""
Depth Anything V2 - Depth Map Generation
========================================
Generates high-quality depth maps from video frames using Depth Anything V2.
Uses HuggingFace transformers pipeline for easy inference.
"""

import cv2
import numpy as np
from pathlib import Path
import time
import warnings
import torch

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.config import load_config, merge_global_config
from lib.depth import DepthAnythingV2


class DepthVisualizer:
    """Visualize depth maps."""
    
    def __init__(self, config: dict):
        self.colormap_name = config['output'].get('colormap', 'inferno')
        self.colormaps = {
            'inferno': cv2.COLORMAP_INFERNO,
            'magma': cv2.COLORMAP_MAGMA,
            'plasma': cv2.COLORMAP_PLASMA,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'jet': cv2.COLORMAP_JET,
            'turbo': cv2.COLORMAP_TURBO,
        }
        self.colormap = self.colormaps.get(self.colormap_name, cv2.COLORMAP_INFERNO)
    
    def to_colored(self, depth: np.ndarray) -> np.ndarray:
        # Normalize to 0-255
        depth_uint8 = (depth * 255).astype(np.uint8)
        # Apply colormap
        colored = cv2.applyColorMap(depth_uint8, self.colormap)
        return colored
    
    def to_grayscale(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth map to grayscale image."""
        return (depth * 255).astype(np.uint8)
    
    def to_raw(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth map to 16-bit raw values."""
        return (depth * 65535).astype(np.uint16)


class VideoProcessor:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        merge_global_config(self.config, Path(__file__).resolve().parent)
        self.model = DepthAnythingV2(
            encoder=self.config.get('model', {}).get('encoder', 'vitl'),
            device=self.config.get('inference', {}).get('device', 'auto')
        )
        self.visualizer = DepthVisualizer(self.config)
        
        self.frame_interval = self.config['sampling']['frame_interval']
        self.start_frame = self.config['sampling']['start_frame']
        self.max_frames = self.config['sampling'].get('max_frames')
    
    def process(self):
        input_path = Path(self.config['paths']['input_video'])
        output_dir = Path(self.config['paths']['output_dir'])
        
        # Create output dirs
        dirs = {}
        if self.config['output'].get('save_rgb', True):
            dirs['rgb'] = output_dir / "rgb"
            dirs['rgb'].mkdir(parents=True, exist_ok=True)
        if self.config['output'].get('save_depth_colored', True):
            dirs['colored'] = output_dir / "depth_colored"
            dirs['colored'].mkdir(parents=True, exist_ok=True)
        if self.config['output'].get('save_depth_grayscale', True):
            dirs['grayscale'] = output_dir / "depth_grayscale"
            dirs['grayscale'].mkdir(parents=True, exist_ok=True)
        if self.config['output'].get('save_depth_raw', True):
            dirs['raw'] = output_dir / "depth_raw"
            dirs['raw'].mkdir(parents=True, exist_ok=True)
        
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
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Generate depth map
            depth = self.model(frame)
            
            # Save outputs
            frame_name = f"frame_{frame_idx:06d}"
            
            if 'rgb' in dirs:
                cv2.imwrite(str(dirs['rgb'] / f"{frame_name}.{img_fmt}"), frame)
            
            if 'colored' in dirs:
                colored = self.visualizer.to_colored(depth)
                cv2.imwrite(str(dirs['colored'] / f"{frame_name}.{img_fmt}"), colored)
            
            if 'grayscale' in dirs:
                grayscale = self.visualizer.to_grayscale(depth)
                cv2.imwrite(str(dirs['grayscale'] / f"{frame_name}.{img_fmt}"), grayscale)
            
            if 'raw' in dirs:
                raw = self.visualizer.to_raw(depth)
                cv2.imwrite(str(dirs['raw'] / f"{frame_name}.png"), raw)
            
            elapsed = time.time() - start_time
            fps_actual = (i + 1) / elapsed if elapsed > 0 else 0
            
            print(f"Frame {frame_idx:6d} | Progress: {i+1}/{len(frame_indices)} | FPS: {fps_actual:.2f}")
        
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
    print("Depth Anything V2 - Depth Map Generation")
    print("=" * 60)
    
    processor = VideoProcessor(args.config)
    processor.process()


if __name__ == "__main__":
    main()
