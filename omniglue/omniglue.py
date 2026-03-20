#!/usr/bin/env python3
"""Simple OmniGlue demo - keypoint matching between two images."""

import os
import time
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import omniglue
from omniglue import utils
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
OUTPUT_DIR = SCRIPT_DIR / "output"


def create_sample_images(output_dir: Path) -> tuple[Path, Path]:
    """Create two synthetic images with a shared pattern for demo purposes.

    Generates a base image with geometric shapes, then creates a second
    image as a rotated/translated version to test matching.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    img0 = np.ones((480, 640, 3), dtype=np.uint8) * 240

    cv2.rectangle(img0, (50, 50), (200, 200), (30, 80, 180), -1)
    cv2.circle(img0, (400, 150), 80, (40, 160, 60), -1)
    cv2.polylines(
        img0,
        [np.array([[300, 350], [450, 280], [550, 400], [400, 450]])],
        isClosed=True,
        color=(180, 40, 40),
        thickness=3,
    )
    for i in range(5):
        for j in range(5):
            cx, cy = 80 + i * 120, 60 + j * 90
            cv2.drawMarker(img0, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 15, 2)

    cv2.putText(
        img0, "OmniGlue", (180, 430),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 80, 80), 3,
    )

    rows, cols = img0.shape[:2]
    center = (cols / 2, rows / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 15, 0.9)
    rotation_matrix[0, 2] += 30
    rotation_matrix[1, 2] += -20
    img1 = cv2.warpAffine(img0, rotation_matrix, (cols, rows), borderValue=(240, 240, 240))

    path0 = output_dir / "sample_image0.png"
    path1 = output_dir / "sample_image1.png"
    cv2.imwrite(str(path0), img0)
    cv2.imwrite(str(path1), img1)

    print(f"Created sample images:\n  {path0}\n  {path1}")
    return path0, path1


def run_matching(image0_path: str, image1_path: str, match_threshold: float = 0.02):
    """Run OmniGlue matching between two images and save visualization."""
    for fp in [image0_path, image1_path]:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Image not found: {fp}")

    print("Loading images...")
    image0 = np.array(Image.open(image0_path).convert("RGB"))
    image1 = np.array(Image.open(image1_path).convert("RGB"))

    print("Loading OmniGlue (SuperPoint + DINOv2 + OmniGlue matcher)...")
    t0 = time.time()
    og = omniglue.OmniGlue(
        og_export=str(MODELS_DIR / "og_export"),
        sp_export=str(MODELS_DIR / "sp_v6"),
        dino_export=str(MODELS_DIR / "dinov2_vitb14_pretrain.pth"),
    )
    print(f"  Models loaded in {time.time() - t0:.1f}s")

    print("Finding matches...")
    t0 = time.time()
    match_kp0, match_kp1, match_confidences = og.FindMatches(image0, image1)
    num_matches = match_kp0.shape[0]
    print(f"  Found {num_matches} raw matches in {time.time() - t0:.1f}s")

    keep = match_confidences > match_threshold
    match_kp0 = match_kp0[keep]
    match_kp1 = match_kp1[keep]
    match_confidences = match_confidences[keep]
    num_filtered = match_kp0.shape[0]
    print(f"  {num_filtered}/{num_matches} matches above threshold {match_threshold}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "matches.png"

    viz = utils.visualize_matches(
        image0, image1,
        match_kp0, match_kp1,
        np.eye(num_filtered),
        show_keypoints=True,
        highlight_unmatched=True,
        title=f"{num_filtered} matches",
        line_width=2,
    )

    plt.figure(figsize=(20, 10), dpi=100)
    plt.axis("off")
    plt.imshow(viz)
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print(f"Saved match visualization to {output_path}")
    return num_filtered


def main():
    parser = argparse.ArgumentParser(description="OmniGlue keypoint matching demo")
    parser.add_argument("image0", nargs="?", help="Path to first image")
    parser.add_argument("image1", nargs="?", help="Path to second image")
    parser.add_argument(
        "--threshold", type=float, default=0.02,
        help="Match confidence threshold (default: 0.02)",
    )
    parser.add_argument(
        "--generate-samples", action="store_true",
        help="Generate synthetic sample images and use them for matching",
    )
    args = parser.parse_args()

    if args.generate_samples or (args.image0 is None and args.image1 is None):
        if args.image0 is None and args.image1 is None and not args.generate_samples:
            print("No images provided. Generating synthetic samples for demo...\n")
        img0_path, img1_path = create_sample_images(OUTPUT_DIR)
    elif args.image0 is None or args.image1 is None:
        parser.error("Provide both image paths, or use --generate-samples")
    else:
        img0_path, img1_path = args.image0, args.image1

    print()
    num_matches = run_matching(str(img0_path), str(img1_path), args.threshold)
    print(f"\nDone! {num_matches} matches found.")


if __name__ == "__main__":
    main()
