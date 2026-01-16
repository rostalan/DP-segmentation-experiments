#!/usr/bin/env python3
"""
Point Cloud Motion Detection
=============================
Finds moving points between consecutive point cloud frames.
Outputs points that have moved significantly.
"""

import argparse
import numpy as np
import yaml
from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d not installed. Run: pip install open3d")
    exit(1)


def find_moving_points(
    pcd1: o3d.geometry.PointCloud,
    pcd2: o3d.geometry.PointCloud,
    distance_threshold: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find points in pcd2 that don't have a close match in pcd1.
    These are points that "moved" or "appeared" in the second frame.
    
    Args:
        pcd1: First point cloud (reference frame)
        pcd2: Second point cloud (frame to check for motion)
        distance_threshold: Max distance to consider a match (smaller = more sensitive)
    
    Returns:
        Tuple of (moving_points, moving_colors) arrays from pcd2
    """
    points2 = np.asarray(pcd2.points)
    colors2 = np.asarray(pcd2.colors) if pcd2.has_colors() else None
    
    moving_points = []
    moving_colors = []
    
    # Build KD-tree for pcd1 (reference)
    pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
    
    # Find points in pcd2 that don't have a close match in pcd1
    threshold_sq = distance_threshold ** 2
    for i, point in enumerate(points2):
        [k, idx, dist] = pcd1_tree.search_knn_vector_3d(point, 1)
        if k > 0 and dist[0] > threshold_sq:
            moving_points.append(point)
            if colors2 is not None:
                moving_colors.append(colors2[i])
    
    return np.array(moving_points), np.array(moving_colors) if moving_colors else None


def create_motion_pointcloud(
    moving_points: np.ndarray,
    moving_colors: np.ndarray = None,
    highlight_color: list = None
) -> o3d.geometry.PointCloud:
    """
    Create a point cloud from moving points.
    """
    pcd = o3d.geometry.PointCloud()
    
    if len(moving_points) == 0:
        return pcd
    
    pcd.points = o3d.utility.Vector3dVector(moving_points)
    
    if highlight_color is not None:
        colors = np.tile(highlight_color, (len(moving_points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif moving_colors is not None and len(moving_colors) > 0:
        pcd.colors = o3d.utility.Vector3dVector(moving_colors)
    
    return pcd


class MotionDetector:
    """Detect motion between point cloud frames."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        motion_cfg = self.config.get('motion', {})
        
        self.pcd1_path = motion_cfg.get('pcd1')
        self.pcd2_path = motion_cfg.get('pcd2')
        self.distance_threshold = motion_cfg.get('distance_threshold', 0.05)
        self.output_dir = Path(motion_cfg.get('output_dir', 'output/frames/motion'))
        self.highlight_color = motion_cfg.get('highlight_color', [1.0, 0.3, 0.0])
        self.keep_original_colors = motion_cfg.get('keep_original_colors', False)
        
        # Point cloud input directory
        pc_output = Path(self.config['paths'].get('output_dir', 'output/frames'))
        self.pointclouds_dir = pc_output / "pointclouds"
    
    def get_frame_pairs(self) -> list[tuple[Path, Path]]:
        """Get pairs of consecutive frames to compare."""
        if self.pcd1_path and self.pcd2_path:
            # Use specified files
            return [(Path(self.pcd1_path), Path(self.pcd2_path))]
        
        # Auto-detect consecutive frames
        ply_files = sorted(self.pointclouds_dir.glob("*.ply"))
        if len(ply_files) < 2:
            print(f"Need at least 2 PLY files in {self.pointclouds_dir}")
            return []
        
        pairs = []
        for i in range(len(ply_files) - 1):
            pairs.append((ply_files[i], ply_files[i + 1]))
        
        return pairs
    
    def process_pair(self, pcd1_path: Path, pcd2_path: Path) -> int:
        """Process a single pair of point clouds. Returns number of moving points."""
        print(f"\nComparing: {pcd1_path.name} <-> {pcd2_path.name}")
        
        # Load point clouds
        pcd1 = o3d.io.read_point_cloud(str(pcd1_path))
        pcd2 = o3d.io.read_point_cloud(str(pcd2_path))
        
        print(f"  Points: {len(pcd1.points):,} / {len(pcd2.points):,}")
        
        # Find moving points
        moving_points, moving_colors = find_moving_points(
            pcd1, pcd2,
            distance_threshold=self.distance_threshold
        )
        
        if len(moving_points) == 0:
            print(f"  No motion detected")
            return 0
        
        print(f"  Moving points: {len(moving_points):,}")
        
        # Create output
        highlight = None if self.keep_original_colors else self.highlight_color
        motion_pcd = create_motion_pointcloud(moving_points, moving_colors, highlight)
        
        # Save
        output_path = self.output_dir / f"motion_{pcd1_path.stem}_{pcd2_path.stem}.ply"
        o3d.io.write_point_cloud(str(output_path), motion_pcd)
        print(f"  Saved: {output_path.name}")
        
        return len(moving_points)
    
    def process(self):
        """Process all frame pairs."""
        print("=" * 60)
        print("Point Cloud Motion Detection")
        print("=" * 60)
        
        pairs = self.get_frame_pairs()
        if not pairs:
            return
        
        print(f"\nFrame pairs to process: {len(pairs)}")
        print(f"Distance threshold: {self.distance_threshold} meters")
        print(f"Output: {self.output_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        total_motion = 0
        for pcd1_path, pcd2_path in pairs:
            total_motion += self.process_pair(pcd1_path, pcd2_path)
        
        print(f"\nDone! Total moving points: {total_motion:,}")
        print(f"Output: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Detect motion between point clouds")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file")
    args = parser.parse_args()
    
    detector = MotionDetector(args.config)
    detector.process()


if __name__ == "__main__":
    main()
