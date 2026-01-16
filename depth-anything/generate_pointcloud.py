#!/usr/bin/env python3
"""
Point Cloud Generation from Depth Maps
=======================================
Converts depth maps + RGB images to colored point clouds (PLY format).
Uses Open3D for point cloud processing.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
import time
import warnings

warnings.filterwarnings('ignore')

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d not installed. Run: pip install open3d")
    exit(1)


class PointCloudGenerator:
    """Generate point clouds from depth maps and RGB images."""
    
    def __init__(self, config: dict):
        self.config = config
        
        pc_cfg = config.get('pointcloud', {})
        
        # Camera intrinsics (default values for typical camera)
        self.fx = pc_cfg.get('focal_length_x', 525.0)
        self.fy = pc_cfg.get('focal_length_y', 525.0)
        self.cx = pc_cfg.get('principal_point_x', None)  # None = image center
        self.cy = pc_cfg.get('principal_point_y', None)
        
        # Depth scaling
        self.depth_scale = pc_cfg.get('depth_scale', 1.0)
        self.depth_offset = pc_cfg.get('depth_offset', 0.0)  # Push scene further from camera
        self.depth_min = pc_cfg.get('depth_min', 0.01)
        self.depth_max = pc_cfg.get('depth_max', 10.0)
        self.invert_depth = pc_cfg.get('invert_depth', True)  # Invert so close=large z
        self.flip_y = pc_cfg.get('flip_y', True)  # Flip Y axis (image Y is inverted)
        
        # Downsampling
        self.downsample = pc_cfg.get('downsample', True)
        self.downsample_factor = pc_cfg.get('downsample_factor', 2)  # 2 = half resolution
        
        # Voxel downsampling (optional, after point cloud creation)
        self.voxel_size = pc_cfg.get('voxel_size', None)  # e.g., 0.01 for 1cm voxels
        
        # Statistical outlier removal
        self.remove_outliers = pc_cfg.get('remove_outliers', False)
        self.outlier_nb_neighbors = pc_cfg.get('outlier_nb_neighbors', 20)
        self.outlier_std_ratio = pc_cfg.get('outlier_std_ratio', 2.0)
    
    def generate(self, depth: np.ndarray, rgb: np.ndarray) -> o3d.geometry.PointCloud:
        """Generate point cloud from depth map and RGB image."""
        h, w = depth.shape[:2]
        
        # Downsample if enabled
        if self.downsample and self.downsample_factor > 1:
            new_h = h // self.downsample_factor
            new_w = w // self.downsample_factor
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = new_h, new_w
        
        # Scale factor for intrinsics when downsampling
        scale = self.downsample_factor if self.downsample else 1
        
        # Scale focal length
        fx = self.fx / scale
        fy = self.fy / scale
        
        # Scale principal point (or use image center if not specified)
        if self.cx is not None:
            cx = self.cx / scale
        else:
            cx = w / 2
        
        if self.cy is not None:
            cy = self.cy / scale
        else:
            cy = h / 2
        
        # Create pixel coordinate grids
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)
        
        # Invert depth if needed (Depth Anything: large=far, we want large=close for projection)
        if self.invert_depth:
            depth = 1.0 - depth
        
        # Convert depth to meters (assuming normalized 0-1 depth)
        z = depth * self.depth_scale + self.depth_offset
        
        # Filter valid depth values
        valid = (z > self.depth_min) & (z < self.depth_max)
        
        # Back-project to 3D
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Flip Y axis (image coordinates: Y down, 3D convention: Y up)
        if self.flip_y:
            y = -y
        
        # Stack coordinates
        points = np.stack([x, y, z], axis=-1)
        points = points[valid]
        
        # Get colors (convert BGR to RGB, normalize to 0-1)
        colors = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        colors = colors[valid]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Optional voxel downsampling
        if self.voxel_size is not None and self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Optional outlier removal
        if self.remove_outliers:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.outlier_nb_neighbors,
                std_ratio=self.outlier_std_ratio
            )
        
        return pcd


class BatchProcessor:
    """Process all depth/RGB pairs to point clouds."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.generator = PointCloudGenerator(self.config)
    
    def process(self):
        paths_cfg = self.config['paths']
        output_base = Path(paths_cfg.get('output_dir', 'output/frames'))
        
        # Input directories
        rgb_dir = output_base / "rgb"
        depth_dir = output_base / "depth_raw"
        
        # Output directory
        pc_dir = output_base / "pointclouds"
        pc_dir.mkdir(parents=True, exist_ok=True)
        
        # Find matching files
        rgb_files = sorted(rgb_dir.glob("*.png"))
        if not rgb_files:
            rgb_files = sorted(rgb_dir.glob("*.jpg"))
        
        if not rgb_files:
            print(f"No RGB images found in {rgb_dir}")
            return
        
        print("=" * 60)
        print("Point Cloud Generation from Depth Maps")
        print("=" * 60)
        print(f"\nInput RGB: {rgb_dir}")
        print(f"Input Depth: {depth_dir}")
        print(f"Output: {pc_dir}")
        print(f"Found {len(rgb_files)} frames")
        
        pc_cfg = self.config.get('pointcloud', {})
        print(f"\nSettings:")
        print(f"  Downsample: {pc_cfg.get('downsample', True)} (factor: {pc_cfg.get('downsample_factor', 2)})")
        print(f"  Voxel size: {pc_cfg.get('voxel_size', 'None')}")
        print(f"  Remove outliers: {pc_cfg.get('remove_outliers', False)}")
        print()
        
        start_time = time.time()
        processed = 0
        
        for i, rgb_path in enumerate(rgb_files):
            # Find corresponding depth file
            depth_path = depth_dir / rgb_path.name
            if not depth_path.exists():
                print(f"Warning: No depth for {rgb_path.name}, skipping")
                continue
            
            # Load images
            rgb = cv2.imread(str(rgb_path))
            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            
            if rgb is None or depth_raw is None:
                print(f"Warning: Failed to load {rgb_path.name}, skipping")
                continue
            
            # Convert 16-bit depth to normalized float
            depth = depth_raw.astype(np.float32) / 65535.0
            
            # Generate point cloud
            pcd = self.generator.generate(depth, rgb)
            
            # Save as PLY
            output_path = pc_dir / f"{rgb_path.stem}.ply"
            o3d.io.write_point_cloud(str(output_path), pcd)
            
            processed += 1
            elapsed = time.time() - start_time
            fps = processed / elapsed if elapsed > 0 else 0
            
            print(f"Frame {rgb_path.stem} | Points: {len(pcd.points):,} | Progress: {i+1}/{len(rgb_files)} | FPS: {fps:.2f}")
        
        elapsed = time.time() - start_time
        print(f"\nDone! {processed} point clouds in {elapsed:.1f}s")
        print(f"Output: {pc_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate point clouds from depth maps")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    args = parser.parse_args()
    
    processor = BatchProcessor(args.config)
    processor.process()


if __name__ == "__main__":
    main()

