# Edge feasibility (ONNX export, quantization, CPU latency — Jetson estimate)

**No Jetson-class device was available for this project.** Every measured
number below is from the CPU-only machine this project was built on. Any
Jetson figure is explicitly labeled **ESTIMATE**, with the reasoning and any
citable source spelled out — never presented as measured.

## Build machine (where every measured number below comes from)

Captured directly, not typed by hand (`benchmarks/measure_edge_latency.py`'s
`platform.platform()` / `platform.processor()`):

| | |
|---|---|
| OS | Windows-10-10.0.26200-SP0 |
| Processor | Intel64 Family 6 Model 186 Stepping 3, GenuineIntel |
| Logical CPUs | 12 (`os.cpu_count()`) |
| GPU | None available (CPU-only build, see `docs/limitations.md`) |
| Python | 3.11.9 |

No Jetson-class device was available — see section 4.

## 1. ONNX export

```bash
python benchmarks/export_onnx.py
```
Exports the torchvision Faster R-CNN MobileNetV3-320-FPN vehicle detector
(BSD-3-Clause, see `docs/data.md`) to ONNX (FP32), then attempts ONNX
Runtime dynamic INT8 quantization.

| Model | Format | Size on disk |
|---|---|---|
| Vehicle detector (Faster R-CNN MobileNetV3-320-FPN) | FP32 ONNX | 74.3 MB |
| Vehicle detector | INT8 (dynamic quantized) | 19.0 MB (-74.4%) |

Reproduced by `python benchmarks/export_onnx.py`. Export required
`dynamo=False` (the legacy TorchScript-tracing exporter) — the new
torch.export/dynamo-based exporter cannot symbolically trace this model's
data-dependent NMS control flow (`GuardOnDataDependentSymNode` in
`torchvision/ops/boxes.py`). This is a real limitation of the current
PyTorch dynamo exporter for two-stage detectors, not a project bug — see
the export script's inline comment for the exact error.

## 2. CPU latency / throughput / memory

```bash
python benchmarks/measure_edge_latency.py --runs 30
```

Measured with `python benchmarks/measure_edge_latency.py --runs 30` (single
image, batch=1, 320x320 input, 5 warmup runs discarded):

| Runtime | p50 latency | p95 latency | Mean latency | Throughput | Model size |
|---|---|---|---|---|---|
| PyTorch FP32 (CPU) | 59.0 ms | 68.4 ms | 57.9 ms | 17.3 FPS | 74.2 MB (.pth) |
| ONNX Runtime FP32 (CPU) | 9.68 ms | 11.7 ms | 9.74 ms | **102.7 FPS** | 74.3 MB |
| ONNX Runtime INT8 (CPU) | — | — | — | — | 19.0 MB, **fails at inference** |

Two real findings, not estimates:
1. **ONNX Runtime's graph optimizations give a ~6x latency reduction over
   raw PyTorch on CPU** for this model (59ms -> 9.7ms mean), with identical
   FP32 weights. This alone is a strong argument for shipping via ONNX
   Runtime rather than raw PyTorch inference, independent of any Jetson
   question.
2. **Naive ONNX Runtime dynamic INT8 quantization breaks this model at
   inference time** — a `Reshape` failure inside the ROI heads
   (`input shape {0,182,2}, requested shape {0,-1}`). Two-stage detectors
   (Faster R-CNN family) have data-dependent, variable-shape ROI operations
   that are known to be fragile under naive post-training quantization —
   this is architecture-specific, not evidence that quantization in general
   doesn't work for this project. A single-stage detector (SSD/YOLO-class)
   would be a better INT8 quantization candidate and is worth trying before
   concluding quantization isn't viable here.

`peak_python_alloc_mb` (from `tracemalloc`, in the raw JSON at
`benchmarks/results/edge_latency_report.json`) is a lower bound on
Python-side memory, not full process RSS — it does not capture native
library allocations inside PyTorch/ONNX Runtime, so it is omitted from the
table above as more misleading than useful at this precision.

## 3. Accuracy change after quantization

**Not applicable for this pass** — the INT8-quantized model fails at
inference (see finding #2 above), so there is no accuracy number to
compare; there is no working INT8 model to benchmark. This is itself the
honest answer for this specific architecture. Revisit if a single-stage
detector is adopted (see `docs/feasibility.md`'s plate-detector
recommendation).

## 4. Jetson feasibility — attempted estimate, verdict: **mostly unknown**

We do not have a Jetson device. We searched for a citable, directly
comparable CPU-only ONNX Runtime benchmark on a Jetson-class board (same
model family: a small two-stage or single-stage CNN detector, CPU-only, no
TensorRT) to scale our own measured numbers against. **We did not find
one.** Available public benchmarks for Jetson boards (e.g. YOLOv5n at
41.5 FPS, YOLO11n at 37.3 FPS, LAF-YOLOv10 at 24.3 FPS on Jetson Orin Nano)
are all **TensorRT/GPU-optimized results**, not CPU-only ONNX Runtime —
mixing those with our CPU-only x86 number would produce a ratio with no
real basis, which is exactly the kind of invented number this project's
honesty rules forbid.

**What we can say, with the reasoning shown:**

- The Jetson Orin Nano's CPU is a 6-core Arm Cortex-A78AE — a different
  instruction set architecture (ARM vs. our build machine's x86_64) and a
  different microarchitecture. Cross-ISA CPU performance does not scale by
  a simple clock-speed or core-count ratio; without a same-model,
  same-runtime measurement on both, any translated latency number would be
  a guess.
- The Jetson Orin Nano also has a 1024-core Ampere GPU (up to 67 TOPS
  INT8) that we never exercised, because our own benchmark is CPU-only
  (see "Build machine" above — no GPU was available to us at all, not even
  the discrete-GPU comparison point). **If** a real deployment used
  TensorRT on the Jetson's GPU instead of CPU-only ONNX Runtime, the
  publicly benchmarked YOLO-class numbers above (24-41 FPS) suggest gate-
  speed throughput (a few FPS is enough for a gate camera) is plausible in
  principle for a *similar-sized, TensorRT-optimized* single-stage
  detector — but our own detector is a two-stage Faster R-CNN, which our
  own INT8 quantization finding above (section 2) shows is not a clean
  drop-in for aggressive optimization, and TensorRT conversion has its own
  op-support caveats we have not tested.
- **Verdict: unknown**, not estimated. The honest path to an actual number
  is renting or borrowing a Jetson Orin Nano (or similar) for a few hours
  and running `benchmarks/measure_edge_latency.py`'s ONNX Runtime path
  directly on it (CPU provider first, then the TensorRT execution provider
  as a second data point) — the script already produces the exact p50/p95/
  throughput numbers this document needs, it just needs to run on that
  hardware.

**Anywhere else in this repo (README, `docs/feasibility.md`) that mentions
Jetson, treat it as "unknown, pending real hardware" — never a specific
number.**
