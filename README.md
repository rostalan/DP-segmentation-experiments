# DP-App-2

Video segmentation experiments and assembly guidance pipelines.

## Structure

```
global_config.yaml       Default settings inherited by all pipelines
lib/                     Shared code (detection, tracking, visualisation, config, depth, SAM utils)
models/                  Model weights (gitignored)
res/                     Input videos and images (gitignored)
```

### Main pipeline

| Folder | Description |
|--------|-------------|
| `all-in-one/` | Assembly recording & playback -- YOLO detection, MediaPipe gestures, step-based output |

### Segmentation experiments

| Folder | Approach |
|--------|----------|
| `yolo8x-world_sam2/` | YOLO-World + SAM 2 (high quality, built-in tracking) |
| `yoloworld-yoloe/` | YOLO-World + YOLOE (efficient hybrid) |
| `yoloe-seg/` | YOLOE text-prompted segmentation |
| `yolo8x-world_fastsam/` | YOLO-World + FastSAM (fast) |
| `yolo26x-seg/` | YOLO-26x ONNX panoptic segmentation |
| `sam2-only/` | SAM 2 automatic segmentation (no detection model) |
| `sam3/` | SAM 3 interactive video object segmentation |

### Depth & 3D

| Folder | Description |
|--------|-------------|
| `depth-anything/` | Depth Anything V2 depth maps + point cloud generation |
| `segmented_depth/` | SAM 3 + Depth Anything V2 -- per-object depth extraction |

### Utilities

| Folder | Description |
|--------|-------------|
| `hand_detection/` | MediaPipe hand landmarks & gesture recognition |
| `search_img/` | Object search in video (YOLO-World + ResNet similarity) |
| `img2vid/` | Image sequence to video converter |
| `video_sample/` | Save video frames via keystroke |
| `omniglue/` | OmniGlue (CVPR'24) feature matching experiment |

## Configuration

`global_config.yaml` defines project-wide defaults: colour palette, model
paths, device, sampling, tracking, and output settings.  Each folder has a
local `config.yaml` that only contains overrides -- missing keys are filled
from the global file automatically via `lib/config.py`.

To change a default for all pipelines, edit `global_config.yaml`.  To change
it for one pipeline, set the key in that folder's `config.yaml`.

## Quick start

```bash
cd <folder>
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python <script>.py                   # uses config.yaml
python <script>.py --config alt.yaml # custom config
```
