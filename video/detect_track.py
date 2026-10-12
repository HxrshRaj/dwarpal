"""Run a COCO-pretrained detector over a video, track objects, write Parquet.

Usage:
    python -m video.detect_track path/to/video.avi --out video/results/detections.parquet --stride 3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from video.tracker import IouTracker

PERSON, CAR, BUS, TRUCK, MOTORCYCLE, BICYCLE = 1, 3, 6, 8, 4, 2
KEEP = {PERSON: "person", CAR: "car", BUS: "bus", TRUCK: "truck", MOTORCYCLE: "motorcycle", BICYCLE: "bicycle"}


def load_detector(device: str = "cpu"):
    import torch
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
    )

    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).eval().to(device)
    return torch, model, weights.transforms()


def run(video_path: str, out_path: str, stride: int = 3, score_threshold: float = 0.6) -> dict:
    torch, model, transforms = load_detector()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = IouTracker()
    cols = {k: [] for k in ("frame_idx", "t_sec", "track_id", "label", "score", "x1", "y1", "x2", "y2")}
    latencies = []
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transforms(torch.from_numpy(rgb).permute(2, 0, 1))
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model([tensor])[0]
        latencies.append(time.perf_counter() - t0)
        h, w = rgb.shape[:2]
        th, tw = tensor.shape[1:]
        sx, sy = w / tw, h / th
        boxes, labels, scores = [], [], []
        for b, l, s in zip(out["boxes"].numpy(), out["labels"].numpy(), out["scores"].numpy()):
            if s >= score_threshold and int(l) in KEEP:
                boxes.append([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy])
                labels.append(KEEP[int(l)])
                scores.append(float(s))
        for tid, b, l, s in zip(tracker.update(idx, boxes), boxes, labels, scores):
            for k, v in zip(cols, (idx, idx / fps, tid, l, s, *map(float, b))):
                cols[k].append(v)
    cap.release()
    table = pa.table(cols)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    lat = np.array(latencies)
    return {
        "video": Path(video_path).name, "fps": fps, "frames_in_video": n_frames, "stride": stride,
        "frames_processed": len(lat), "detections": table.num_rows,
        "tracks": len(set(cols["track_id"])),
        "latency_ms_mean": float(lat.mean() * 1000), "latency_ms_p95": float(np.percentile(lat, 95) * 1000),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="video/results/detections.parquet")
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    print(run(a.video, a.out, a.stride))
