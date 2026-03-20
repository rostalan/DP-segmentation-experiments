#!/usr/bin/env python3
"""
Segmented Depth Pipeline
========================
Combines SAM 3 for interactive segmentation/tracking with Depth Anything V2 for depth estimation.
Outputs depth maps grouped by object ID.
"""

import cv2
import numpy as np
from pathlib import Path
import time
import argparse
import os
import tempfile
import warnings

# Disable torch.compile/Inductor to avoid Python.h dependency issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
try:
    torch.compile = lambda model, *args, **kwargs: model
except Exception:
    pass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.config import load_config, merge_global_config
from lib.sam_utils import InteractiveSelector, make_custom_sam3_predictor
from lib.depth import DepthAnythingV2

warnings.filterwarnings('ignore')


class Pipeline:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        merge_global_config(self.config, Path(__file__).resolve().parent)
        self.depth_model = DepthAnythingV2(
            encoder=self.config.get('depth_anything', {}).get('encoder', 'vitl'),
            device=self.config.get('depth_anything', {}).get('device', 'auto')
        )
        self.sam_model_path = self.config['paths'].get('sam3_model', self.config['paths'].get('sam3', '../models/sam3.pt'))
        self.output_dir = Path(self.config['paths']['output_dir'])
        
        # Prepare output directories
        if self.config['output'].get('save_full_depth', False):
            (self.output_dir / "full").mkdir(parents=True, exist_ok=True)
            
    def normalize_depth(self, depth_map):
        """Normalize depth map to 0-255 uint8 for visualization."""
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 1e-6:
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth_map)
        
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        return depth_uint8

    def run(self):
        input_video = self.config['paths']['input_video']
        
        # 1. Read Frames
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_video}")
            
        start_frame = self.config['sampling'].get('start_frame', 0)
        frame_interval = self.config['sampling'].get('frame_interval', 1)
        max_frames = self.config['sampling'].get('max_frames')
        
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
        ret, first_frame = cap.read()
        if not ret:
            print("Error reading first frame.")
            return

        # 2. Interactive Selection
        selector = InteractiveSelector(first_frame)
        bboxes = selector.select()
        if not bboxes:
            print("No selection. Exiting.")
            return

        # 3. Read Video Content into Memory (for SAM 3 Video Predictor)
        frames = []
        frames.append(first_frame)
        
        print("Reading frames...")
        count = 1
        while cap.isOpened():
            if max_frames and count >= max_frames:
                break
                
            skipped = 0
            while skipped < (frame_interval - 1):
                if not cap.grab(): break
                skipped += 1
                
            ret, frame = cap.read()
            if not ret: break
            
            frames.append(frame)
            count += 1
            if count % 100 == 0: print(f"Read {count} frames...")
            
        cap.release()
        print(f"Loaded {len(frames)} frames.")
        
        # 4. Create Temp Video for SAM 3 (it requires file input)
        height, width = frames[0].shape[:2]
        temp_video_fd, temp_video_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_video_fd)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        
        # 5. Initialize SAM 3 Predictor
        overrides = dict(
            conf=self.config['sam3'].get('conf_threshold', 0.25),
            task="segment",
            mode="predict",
            model=self.sam_model_path,
            half=False,
            device=self.config['sam3'].get('device', 'cpu')
        )
        
        try:
            CustomSAM3VideoPredictor = make_custom_sam3_predictor()
            sam_predictor = CustomSAM3VideoPredictor(overrides=overrides)
            results = sam_predictor(
                source=temp_video_path,
                bboxes=bboxes,
                stream=True
            )
            
            # 6. Processing Loop
            print("Processing frames...")
            start_time = time.time()
            
            for i, result in enumerate(results):
                frame_idx = start_frame + (i * frame_interval)
                frame = frames[i]
                
                # A. Generate Full Depth Map
                full_depth = self.depth_model(frame)
                
                # Save full depth if requested
                if self.config['output'].get('save_full_depth', False):
                    depth_vis = cv2.applyColorMap(self.normalize_depth(full_depth), cv2.COLORMAP_INFERNO)
                    cv2.imwrite(str(self.output_dir / "full" / f"{frame_idx:06d}.png"), depth_vis)

                # B. Process Object Masks
                # result.masks.data is (N, H, W)
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy() # Boolean or float masks
                    
                    for obj_id, mask in enumerate(masks):
                        # Ensure mask is binary
                        mask_binary = (mask > 0.5).astype(np.uint8)
                        
                        # Create Object Directory
                        obj_dir = self.output_dir / f"obj_{obj_id:02d}"
                        rgb_dir = obj_dir / "rgb"
                        depth_dir = obj_dir / "depth"
                        
                        rgb_dir.mkdir(parents=True, exist_ok=True)
                        depth_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Mask Depth Map
                        # Apply mask to depth map (set background to 0 or min?)
                        masked_depth = full_depth.copy()
                        masked_depth[mask_binary == 0] = 0 # Background 0
                        
                        # Save Masked Depth (as grayscale image for 3D usage)
                        # Usually for 3D, raw values are best, but standard image formats clip.
                        # We'll normalize for now (0-255), or use 16-bit png if raw is needed.
                        # Let's use normalized uint8 for visualization/simple 3D for now.
                        if self.config['output'].get('save_depth', True):
                            depth_uint8 = self.normalize_depth(masked_depth)
                            # Keep background black
                            depth_uint8[mask_binary == 0] = 0
                            cv2.imwrite(str(depth_dir / f"{frame_idx:06d}.png"), depth_uint8)
                            
                        # Save Masked RGB (Optional)
                        if self.config['output'].get('save_rgb', True):
                            masked_rgb = frame.copy()
                            masked_rgb[mask_binary == 0] = 0
                            cv2.imwrite(str(rgb_dir / f"{frame_idx:06d}.png"), masked_rgb)

                elapsed = time.time() - start_time
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                if i % 10 == 0:
                    print(f"Processed frame {i} | FPS: {fps:.2f}")

        finally:
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
                
        print(f"\nDone. Output in {self.output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    
    pipeline = Pipeline(args.config)
    pipeline.run()

if __name__ == "__main__":
    main()

