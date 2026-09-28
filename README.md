# Dwarpal — Yard Gate Detection & OCR Prototype

A computer-vision + OCR prototype for automated capture of truck/trailer
data at yard gates (license plate, USDOT number, trailer ID, seal
presence), built as a feasibility study: what's achievable in-house today,
what needs more data, and what's still unknown without real hardware or
real gate imagery.

**No real yard-gate imagery was used anywhere in this project — none
exists for this project.** Every real image is a public dataset photographed
in some other context (street-level plates, general traffic/CCTV). See
[`docs/limitations.md`](docs/limitations.md) before trusting any number here.

## What is real / synthetic / estimated / pending

| | Real | Synthetic | Estimated | Pending (needs credentials/hardware) |
|---|---|---|---|---|
| Vehicle/truck detection | COCO 2017 images, torchvision Faster R-CNN (pretrained, not fine-tuned) | — | — | GPU fine-tuning (notebook ready) |
| Plate detection + OCR | OpenALPR benchmark (AGPL, local eval only) | Synthetic panel renders | — | A licensed, fine-tuning-scale plate dataset |
| USDOT number | — (no public dataset exists) | Synthetic panel renders, format from 49 CFR 390.21 | — | Real gate imagery |
| Trailer ID | — (no public dataset exists) | Synthetic panel renders, ISO 6346 checksum validated | — | Real gate imagery |
| Seal presence | — | Synthetic procedural icon only (explicitly not representative — see `docs/limitations.md`) | — | Real gate imagery, real seal hardware reference |
| CPU latency / model size | Measured on the build machine | — | — | — |
| Jetson feasibility | — | — | Labeled estimate, or "unknown" where no citable source exists | Real Jetson device |
| Cloud deployment | — | — | — | Cloud account credentials (see `deploy/README.md`) |
| Ruby on Rails admin (Phase 9) | — | — | — | See status note below |

See `docs/feasibility.md` for the full field-by-field verdict, and
`docs/benchmarks.md` / `docs/edge-feasibility.md` for every number's exact
reproduction command.

## Architecture

```
data_pipeline/   real dataset download + synthetic generator + COCO/YOLO conversion + leak-free split
training/        Colab notebook for GPU fine-tuning (pending user run)
benchmarks/      detection/OCR/robustness metrics, ONNX export, CPU latency measurement
backend/         FastAPI gate workflow service (detection -> OCR -> validation -> confidence gating -> Postgres)
frontend/        Next.js UI: upload+overlay, review queue, dashboard, feasibility page
common/          field validators (USDOT/trailer ID/plate) shared by backend + benchmarks
docs/            data.md, benchmarks.md, edge-feasibility.md, feasibility.md, limitations.md
deploy/          ready-to-deploy config, not yet deployed (see deploy/README.md)
```

The backend and the benchmark scripts import the **same**
`backend/app/detection.py` / `backend/app/ocr.py` code — a number in
`docs/benchmarks.md` is what the API actually does, not a separate demo path.

## How to run

### Backend + frontend (Docker Compose)

```bash
docker compose up --build
```
- Backend: http://localhost:8000 (docs at `/docs`)
- Frontend: http://localhost:3000
- MLflow UI: http://localhost:5000

### Data pipeline (local Python, CPU)

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

python data_pipeline/download_coco_trucks.py --limit 250
python data_pipeline/download_openalpr_benchmark.py --region us   # AGPL — local eval only, see docs/data.md
python data_pipeline/synthetic/generate_synthetic.py --count 500 --seed 42
python data_pipeline/convert_to_yolo.py
python data_pipeline/split_by_source.py
```

### Benchmarks

```bash
python benchmarks/run_benchmarks.py --limit-per-source 100
python benchmarks/robustness.py
python benchmarks/tune_thresholds.py
python benchmarks/export_onnx.py
python benchmarks/measure_edge_latency.py --runs 30
python benchmarks/export_for_dashboard.py   # feeds frontend/public/data/benchmark_results.json
```

### Tests

```bash
pytest    # common/validators, split leakage, benchmark metrics, FastAPI contract tests
```

CI (`.github/workflows/ci.yml`) runs lint, the full pytest suite, a frontend
build, and a CPU smoke evaluation on every push.

## Detector license choice

State-of-the-art real-time detectors via `ultralytics` (YOLOv5/v8/v11) are
AGPL-3.0. This prototype uses `torchvision` (BSD-3-Clause) instead, and
reports the resulting accuracy honestly — see
[`docs/data.md`](docs/data.md) "Detector license choice" for the full
reasoning and the tradeoff this implies for a production build.

## Rails integration (Phase 9)

_Status note filled in at the end of the build — see the final report below
for whether this was attempted._

## Final report

_Filled in once the build is complete — three sections: verified and safe
to claim, synthetic/estimated/limited, and pending (needs credentials or
hardware not available in this environment)._
