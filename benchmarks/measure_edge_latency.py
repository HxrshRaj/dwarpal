"""Phase 4: measure real CPU latency (p50/p95), throughput, and peak memory
for the vehicle detector, both as plain PyTorch and as exported ONNX
(FP32 and INT8 if available). This is the ONLY source for any latency
number in docs/edge-feasibility.md — no number is hand-typed there.

Usage:
    python benchmarks/measure_edge_latency.py --runs 30
Writes:
    benchmarks/results/edge_latency_report.json
"""
import argparse
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

RESULTS_DIR = ROOT / "benchmarks" / "results"


def percentile(values, p):
    return float(np.percentile(values, p))


def measure_pytorch(runs: int, warmup: int = 5):
    import torch
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )

    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
    model.eval()
    dummy = torch.rand(3, 320, 320)

    for _ in range(warmup):
        with torch.no_grad():
            model([dummy])

    tracemalloc.start()
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model([dummy])
        latencies.append((time.perf_counter() - t0) * 1000)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "runtime": "pytorch_fp32_cpu",
        "n_runs": runs,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "mean_ms": float(np.mean(latencies)),
        "throughput_fps": 1000.0 / float(np.mean(latencies)),
        "peak_python_alloc_mb": peak / (1024 * 1024),
    }


def measure_onnx(onnx_path: Path, runs: int, warmup: int = 5, label: str = "onnx"):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = np.random.rand(1, 3, 320, 320).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    tracemalloc.start()
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "runtime": label,
        "n_runs": runs,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "mean_ms": float(np.mean(latencies)),
        "throughput_fps": 1000.0 / float(np.mean(latencies)),
        "peak_python_alloc_mb": peak / (1024 * 1024),
        "model_size_mb": onnx_path.stat().st_size / (1024 * 1024),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "measurements": [],
    }

    print("[measure] pytorch fp32 cpu...")
    report["measurements"].append(measure_pytorch(args.runs))

    fp32_onnx = RESULTS_DIR / "vehicle_detector_fp32.onnx"
    if fp32_onnx.exists():
        print("[measure] onnxruntime fp32...")
        report["measurements"].append(measure_onnx(fp32_onnx, args.runs, label="onnxruntime_fp32_cpu"))

    int8_onnx = RESULTS_DIR / "vehicle_detector_int8.onnx"
    if int8_onnx.exists():
        print("[measure] onnxruntime int8...")
        report["measurements"].append(measure_onnx(int8_onnx, args.runs, label="onnxruntime_int8_cpu"))

    out_path = RESULTS_DIR / "edge_latency_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
