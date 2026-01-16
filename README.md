# DP-App-2: Video Segmentation Experiments

This repository contains various experiments for video segmentation using state-of-the-art models including YOLO-World, SAM 2, SAM 3, FastSAM, YOLOE, and Depth Anything V2.

## Structure

Each folder represents a different experimental approach with its own virtual environment and requirements.

### Segmentation Pipelines
- **`yolo8x-world_sam2/`**: Zero-shot segmentation using YOLO-World for detection and SAM 2 for segmentation. High quality.
- **`yoloworld-yoloe/`**: Hybrid approach using YOLO-World for open-vocabulary detection and YOLOE for segmentation. Efficient with decent quality.
- **`yoloe-seg/`**: Instance segmentation using YOLOE with text prompts.
- **`yolo8x-world_fastsam/`**: YOLO-World detection + FastSAM segmentation for speed.
- **`sam2-only/`**: Automatic segmentation using SAM 2 (without text prompts).
- **`sam3/`**: Experiments with SAM 3 (requires access).
- **`yolo26x-seg/`**: Experiments with YOLO-26x segmentation.

### Depth & 3D
- **`depth-anything/`**: Depth map generation using Depth Anything V2 and point cloud creation.

## Setup

1. **Models**: Place model weights in the `models/` directory (ignored by git).
   - `yolov8x-worldv2.pt`
   - `sam2.1_t.pt`
   - `yoloe-26x-seg.pt`
   - `depth_anything_v2_vitl.pth`
   - etc.

2. **Resources**: Place input videos in `res/videos/` (ignored by git).

3. **Running an Experiment**:
   Navigate to the desired folder, set up the environment, and run the script.
   
   Example for `yoloworld-yoloe`:
   ```bash
   cd yoloworld-yoloe
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python segment_video.py
   ```

## Requirements

Each folder contains a `requirements.txt` specific to its dependencies.

