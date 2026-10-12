"""Run a pretrained detector over a video, track objects, write Parquet.

Detector: MobileNet-SSD (Caffe, trained on PASCAL VOC) through OpenCV's DNN module.
The torchvision COCO weights used elsewhere in this repo could not be downloaded
in the environment this module was built in, so this uses weights fetched from
the chuanqi305/MobileNet-SSD GitHub repo instead (see video/models/README.md).

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

VOC = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
       "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
KEEP = {"person", "car", "bus", "bicycle", "motorbike"}
MODEL_DIR = Path(__file__).parent / "models"


def load_detector():
    return cv2.dnn.readNetFromCaffe(str(MODEL_DIR / "MobileNetSSD_deploy.prototxt"),
                                    str(MODEL_DIR / "MobileNetSSD_deploy.caffemodel"))


def detect(net, frame_bgr, score_threshold):
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    out = net.forward()[0, 0]
    boxes, labels, scores = [], [], []
    for _, cls, conf, x1, y1, x2, y2 in out:
        name = VOC[int(cls)]
        if conf >= score_threshold and name in KEEP:
            boxes.append([x1 * w, y1 * h, x2 * w, y2 * h])
            labels.append(name)
            scores.append(float(conf))
    return boxes, labels, scores


def run(video_path: str, out_path: str, stride: int = 2, score_threshold: float = 0.3) -> dict:
    net = load_detector()
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
        t0 = time.perf_counter()
        boxes, labels, scores = detect(net, frame, score_threshold)
        latencies.append(time.perf_counter() - t0)
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
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    print(run(a.video, a.out, a.stride))
