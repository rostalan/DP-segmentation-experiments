"""Shared Depth Anything V2 wrapper (HuggingFace transformers)."""

from __future__ import annotations
import cv2
import numpy as np
import torch

_MODEL_NAMES = {
    "vits": "depth-anything/Depth-Anything-V2-Small-hf",
    "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
    "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
}


class DepthAnythingV2:
    def __init__(self, encoder: str = "vitl", device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_name = _MODEL_NAMES.get(encoder, _MODEL_NAMES["vitl"])
        print(f"Loading Depth Anything V2 ({encoder}) on {device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(device).eval()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return a normalised (0-1) depth map at the original image resolution."""
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            depth = self.model(**inputs).predicted_depth

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(h, w),
            mode="bicubic", align_corners=False,
        ).squeeze().cpu().numpy()

        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth
