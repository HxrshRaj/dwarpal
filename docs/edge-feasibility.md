# Edge feasibility (ONNX export, quantization, CPU latency — Jetson estimate)

**No Jetson-class device was available for this project.** Every measured
number below is from the CPU-only machine this project was built on. Any
Jetson figure is explicitly labeled **ESTIMATE**, with the reasoning and any
citable source spelled out — never presented as measured.

## Build machine (where every measured number below comes from)

Captured directly, not typed by hand:
```bash
python -c "import platform; print(platform.platform())"
python -c "import os; print('cpu_count', os.cpu_count())"
```
Result: _pending first run — see docs/benchmarks.md status note; this file
is updated in the same commit as the benchmark numbers._

## 1. ONNX export

```bash
python benchmarks/export_onnx.py
```
Exports the torchvision Faster R-CNN MobileNetV3-320-FPN vehicle detector
(BSD-3-Clause, see `docs/data.md`) to ONNX (FP32), then attempts ONNX
Runtime dynamic INT8 quantization.

| Model | Format | Size on disk |
|---|---|---|
| Vehicle detector | FP32 ONNX | _pending_ |
| Vehicle detector | INT8 (dynamic quantized) | _pending_ |

## 2. CPU latency / throughput / memory

```bash
python benchmarks/measure_edge_latency.py --runs 30
```

| Runtime | p50 latency | p95 latency | Throughput | Peak Python allocation |
|---|---|---|---|---|
| PyTorch FP32 (CPU) | _pending_ | _pending_ | _pending_ | _pending_ |
| ONNX Runtime FP32 (CPU) | _pending_ | _pending_ | _pending_ | _pending_ |
| ONNX Runtime INT8 (CPU) | _pending_ | _pending_ | _pending_ | _pending_ |

`peak_python_alloc_mb` (from `tracemalloc`) is a lower bound on Python-side
memory, not full process RSS — it does not capture native library
allocations inside PyTorch/ONNX Runtime. Treat it as directionally useful,
not a precise memory ceiling.

## 3. Accuracy change after quantization

INT8 dynamic quantization only touches weights (not activations), so
accuracy impact is typically small for this kind of model, but "typically"
is not a number — `benchmarks/run_benchmarks.py` is re-run against the
ONNX INT8 session in a follow-up pass, and the real mAP delta (not an
assumption) will be filled in here alongside `docs/benchmarks.md`.

| | mAP@0.5 (real, coco_trucks) |
|---|---|
| PyTorch FP32 | _pending_ (see docs/benchmarks.md) |
| ONNX INT8 | _pending_ |

## 4. Jetson feasibility — ESTIMATE, not measured

We do not have a Jetson device. The estimate below combines (a) our own
measured CPU latency/size numbers above, scaled by publicly documented
relative throughput figures for Jetson-class boards where we can cite a
source, and (b) explicit assumptions, listed so they can be checked.

**This section is filled in only after step 2 above produces real numbers**
— an estimate needs a real baseline to scale from; scaling from nothing
would just be a guess wearing a number's clothes. Structure planned:

- Take the measured ONNX Runtime INT8 CPU p50 latency (ms/frame) from
  section 2.
- Apply a documented throughput ratio between the build CPU and a named
  Jetson SKU (e.g. Jetson Orin Nano) **only if a citable NVIDIA benchmark
  for a comparable model class (small ONNX CNN detector) can be found** —
  if no comparable citation exists, this section says "unknown" rather than
  inventing a ratio.
- State the assumption explicitly: single-stream, batch=1, no other
  processes competing for the device, same 320x320 input resolution used
  in our own benchmark.
- Give a range, not a false-precision point estimate, and mark every step
  of the arithmetic so the estimate can be audited or corrected by someone
  who does have the hardware.

**Until that pass runs, treat any Jetson claim anywhere else in this repo
(README, feasibility.md) as "unknown / not yet estimated," not as a
placeholder for a specific number.**
