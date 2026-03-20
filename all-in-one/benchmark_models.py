#!/usr/bin/env python3
"""Benchmark: YOLOv11 detection vs RF-DETR Nano on the same video frames."""

import time
import cv2
import numpy as np
import torch
from pathlib import Path

VIDEO_PATH = "../../res/videos/VID20260201102500.mp4"
YOLO_WEIGHTS = "../models/oboe-train-v1-weights.pt"
RFDETR_WEIGHTS = "../models/RF-DETR-nano-weights.pt"
CONFIDENCE = 0.5
NUM_SAMPLE_FRAMES = 50
WARMUP_FRAMES = 3


def load_yolo(weights_path: str):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    return model


def load_rfdetr(weights_path: str):
    from rfdetr import RFDETRNano

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]

    class_names = {i: name for i, name in enumerate(args.class_names)}
    num_classes = args.num_classes

    model = RFDETRNano(
        num_classes=num_classes,
        pretrain_weights=None,
        device="cpu",
    )
    model.model.model.load_state_dict(ckpt["model"])
    model.model.model.eval()
    model.model.args = args
    model._class_names = class_names

    return model, class_names


def sample_frames(video_path: str, n: int) -> list:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n + WARMUP_FRAMES, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def benchmark_yolo(model, frames: list, conf: float):
    for frame in frames[:WARMUP_FRAMES]:
        model(frame, conf=conf, iou=0.5, verbose=False)

    test_frames = frames[WARMUP_FRAMES:]
    times = []
    all_results = []
    for frame in test_frames:
        t0 = time.perf_counter()
        results = model(frame, conf=conf, iou=0.5, verbose=False)
        t1 = time.perf_counter()
        times.append(t1 - t0)

        r = results[0]
        dets = []
        if len(r.boxes) > 0:
            for i in range(len(r.boxes)):
                dets.append({
                    "class": r.names[int(r.boxes.cls[i])],
                    "conf": float(r.boxes.conf[i]),
                    "bbox": r.boxes.xyxy[i].cpu().numpy().tolist(),
                })
        all_results.append(dets)

    return times, all_results


def benchmark_rfdetr(model, class_names, frames: list, conf: float):
    for frame in frames[:WARMUP_FRAMES]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model.predict(rgb, threshold=conf)

    test_frames = frames[WARMUP_FRAMES:]
    times = []
    all_results = []
    for frame in test_frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        detections = model.predict(rgb, threshold=conf)
        t1 = time.perf_counter()
        times.append(t1 - t0)

        dets = []
        if len(detections) > 0:
            for i in range(len(detections)):
                cid = int(detections.class_id[i])
                dets.append({
                    "class": class_names.get(cid, f"class_{cid}"),
                    "conf": float(detections.confidence[i]),
                    "bbox": detections.xyxy[i].tolist(),
                })
        all_results.append(dets)

    return times, all_results


def print_stats(name: str, times: list, all_results: list):
    times_ms = [t * 1000 for t in times]
    avg = np.mean(times_ms)
    med = np.median(times_ms)
    mn = np.min(times_ms)
    mx = np.max(times_ms)
    det_counts = [len(r) for r in all_results]

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Inference ({len(times)} frames):")
    print(f"    Mean:   {avg:.1f} ms")
    print(f"    Median: {med:.1f} ms")
    print(f"    Min:    {mn:.1f} ms")
    print(f"    Max:    {mx:.1f} ms")
    print(f"    FPS:    {1000/avg:.1f}")
    print(f"  Detections per frame:")
    print(f"    Mean:   {np.mean(det_counts):.1f}")
    print(f"    Min:    {np.min(det_counts)}")
    print(f"    Max:    {np.max(det_counts)}")

    class_counts = {}
    total_conf = {}
    for res in all_results:
        for d in res:
            c = d["class"]
            class_counts[c] = class_counts.get(c, 0) + 1
            total_conf[c] = total_conf.get(c, 0.0) + d["conf"]

    print(f"  Class detection frequency (across {len(all_results)} frames):")
    for cls in sorted(class_counts.keys()):
        cnt = class_counts[cls]
        avg_c = total_conf[cls] / cnt
        pct = cnt / len(all_results) * 100
        print(f"    {cls:20s}: {cnt:4d} ({pct:5.1f}%)  avg_conf={avg_c:.3f}")


def compare_frame_results(yolo_results, rfdetr_results):
    sample_idx = np.linspace(0, len(yolo_results) - 1, min(5, len(yolo_results)), dtype=int)
    print(f"\n{'='*60}")
    print("  Side-by-side comparison (sample frames)")
    print(f"{'='*60}")
    for i in sample_idx:
        y_classes = sorted([d["class"] for d in yolo_results[i]])
        r_classes = sorted([d["class"] for d in rfdetr_results[i]])
        match = "MATCH" if y_classes == r_classes else "DIFFER"
        print(f"\n  Frame {i}:  [{match}]")
        print(f"    YOLOv11  ({len(yolo_results[i]):2d}): {', '.join(y_classes)}")
        print(f"    RF-DETR  ({len(rfdetr_results[i]):2d}): {', '.join(r_classes)}")


def main():
    print("Sampling frames from video...")
    frames = sample_frames(VIDEO_PATH, NUM_SAMPLE_FRAMES)
    print(f"  Sampled {len(frames)} frames ({WARMUP_FRAMES} warmup + {len(frames)-WARMUP_FRAMES} test)")
    h, w = frames[0].shape[:2]
    print(f"  Resolution: {w}x{h}")

    print("\nLoading YOLOv11 detection model...")
    yolo_model = load_yolo(YOLO_WEIGHTS)
    print("  Loaded.")

    print("\nBenchmarking YOLOv11...")
    yolo_times, yolo_results = benchmark_yolo(yolo_model, frames, CONFIDENCE)
    print_stats("YOLOv11 Detection (oboe-train-v1)", yolo_times, yolo_results)
    del yolo_model

    print("\n\nLoading RF-DETR Nano model...")
    rfdetr_model, rfdetr_classes = load_rfdetr(RFDETR_WEIGHTS)
    print("  Loaded.")

    print("\nBenchmarking RF-DETR Nano...")
    rfdetr_times, rfdetr_results = benchmark_rfdetr(rfdetr_model, rfdetr_classes, frames, CONFIDENCE)
    print_stats("RF-DETR Nano (fine-tuned)", rfdetr_times, rfdetr_results)

    compare_frame_results(yolo_results, rfdetr_results)

    yolo_avg = np.mean(yolo_times) * 1000
    rfdetr_avg = np.mean(rfdetr_times) * 1000
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(f"  YOLOv11:   {yolo_avg:.1f} ms/frame  ({1000/yolo_avg:.1f} FPS)")
    print(f"  RF-DETR-N: {rfdetr_avg:.1f} ms/frame ({1000/rfdetr_avg:.1f} FPS)")
    ratio = rfdetr_avg / yolo_avg
    faster = "YOLOv11" if yolo_avg < rfdetr_avg else "RF-DETR"
    print(f"  Speed ratio: {ratio:.2f}x  ({faster} is faster)")


if __name__ == "__main__":
    main()
