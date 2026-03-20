#!/usr/bin/env python3
"""
Image to Video Converter Utility
Converts a sequence of images from a directory into a video file.
"""

import cv2
import yaml
import argparse
from pathlib import Path
import re
import sys
from datetime import datetime

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def make_video(config_path):
    config = load_config(config_path)
    
    input_dir = Path(config['paths']['input_dir'])
    output_video_config = Path(config['paths']['output_video'])
    
    # Generate timestamp filename
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    extension = output_video_config.suffix if output_video_config.suffix else ".mp4"
    output_filename = f"{timestamp}{extension}"
    
    # Use directory from config, or current dir if none
    output_dir = output_video_config.parent
    output_video = output_dir / output_filename
    
    # Ensure input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
        
    glob_pattern = config['processing'].get('glob_pattern', '*.jpg')
    print(f"Searching for images in '{input_dir}' matching '{glob_pattern}'...")
    
    # Gather image files
    image_files = list(input_dir.glob(glob_pattern))
    
    if not image_files:
        print(f"No images found in '{input_dir}' matching '{glob_pattern}'.")
        sys.exit(1)
        
    # Sort files
    if config['processing'].get('sort_numerical', True):
        image_files.sort(key=lambda p: natural_sort_key(p.name))
    else:
        image_files.sort()
        
    print(f"Found {len(image_files)} images.")
    
    # Read first image to determine properties
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"Error: Could not read first image '{image_files[0]}'.")
        sys.exit(1)
        
    original_height, original_width = first_image.shape[:2]
    
    # Determine output resolution
    target_width = config['video'].get('width')
    target_height = config['video'].get('height')
    do_resize = config['video'].get('resize', False)
    
    if not do_resize or target_width is None or target_height is None:
        width, height = original_width, original_height
    else:
        width, height = target_width, target_height
        
    fps = config['video'].get('fps', 30.0)
    codec_str = config['video'].get('codec', 'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*codec_str)
    
    print(f"Output Video: {output_video}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Codec: {codec_str}")
    
    # Create output directory if it doesn't exist (if output_video has parents)
    if output_video.parent != Path('.'):
        output_video.parent.mkdir(parents=True, exist_ok=True)
        
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not create video writer. Check codec and output path.")
        sys.exit(1)
        
    repeat_frames = config['processing'].get('repeat_frames', 1)
    if repeat_frames < 1:
        repeat_frames = 1
    
    count = 0
    total_frames = len(image_files) * repeat_frames
    
    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        
        if frame is None:
            print(f"Warning: Could not read image '{img_path}'. Skipping.")
            continue
            
        if do_resize and (frame.shape[1] != width or frame.shape[0] != height):
            frame = cv2.resize(frame, (width, height))
        elif frame.shape[1] != width or frame.shape[0] != height:
             # If resize is false but dimensions mismatch first frame, we might have an issue for VideoWriter
             # VideoWriter expects all frames to be same size.
             # We resize to match the writer's dimensions to be safe if they drift.
             frame = cv2.resize(frame, (width, height))

        for _ in range(repeat_frames):
            out.write(frame)
            count += 1
            
            if count % 100 == 0:
                print(f"Processed {count}/{total_frames} frames...", end='\r')
            
    out.release()
    print(f"\nDone! Video saved to {output_video}")

def main():
    parser = argparse.ArgumentParser(description="Convert images to video.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    make_video(args.config)

if __name__ == "__main__":
    main()

