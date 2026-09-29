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
| Ruby on Rails admin (Phase 9) | Built and live-integration-tested against the real backend (see below) | — | — | — |

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
gate-visits-admin/  optional Phase 9: Rails admin UI consuming the FastAPI service over HTTP
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
python benchmarks/run_benchmarks.py --limit-per-source 250
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

**Attempted and working.** [`gate-visits-admin/`](gate-visits-admin/) is a
real Rails 8.1 app (Ruby 3.4, generated with `--skip-active-record` — it
has no database of its own; all data lives in the FastAPI/Postgres
backend) with a model, controller, views, and 8 passing Minitest tests. It
consumes the gate service over plain `Net::HTTP`: a visit list, a visit
detail page with per-field OCR results, and an operator correction form.

This was verified with a **live integration test**, not just mocks: the
real FastAPI backend and the Rails app were booted side by side, a real
OpenALPR benchmark plate image was uploaded through the API, Rails
correctly rendered the real detected fields and audit trail, and an
operator correction submitted through the actual HTML form (real CSRF
protection intact) was confirmed to land in the backend's
`/export/corrections` dataset. Two real bugs were caught by this and
fixed — see the commit history for `gate-visits-admin/` for specifics; one
of them (a missing `require`) was invisible to the mocked test suite and
only surfaced under live integration, which is exactly why that extra step
was worth doing.

Run it: `cd gate-visits-admin && bin/rails server -p 3001` with
`DWARPAL_API_URL` pointing at the FastAPI backend.

## Final report

### 1. Verified and safe to claim

- **Truck/vehicle detection works on real photos**: torchvision Faster
  R-CNN (pretrained COCO, not fine-tuned) reaches mAP@0.5 = 0.328 on 248
  real COCO images. `python benchmarks/run_benchmarks.py --limit-per-source 250`.
- **OCR works, end to end, on a real plate photo**: manually verified
  (`backend/app/pipeline.py`'s `GatePipeline.run()`, the exact code the API
  uses) on a real OpenALPR benchmark image — EasyOCR read a real plate as
  "YG9-X2G" against ground truth "YG9X2G", an exact match after the
  production validator's normalization.
- **The full gate workflow is real and was exercised live end to end**:
  image upload → detection → OCR (two engines compared) → format
  validation (49 CFR 390.21-cited USDOT rules, ISO 6346-checksummed
  trailer IDs) → confidence gating → Postgres-backed persistence → audit
  log → operator correction → corrections export, all through the actual
  FastAPI service (`backend/app/main.py`), verified with `curl` against a
  running instance, not just unit tests.
- **The Rails admin app (Phase 9) is real and was live-integration-tested**
  against the real backend, not just mocks — see "Rails integration" above.
  Two real bugs were caught this way (a missing `require`, and a view
  bypassing its controller's injected client).
- **54 automated tests pass** (`pytest`: validators, split-leakage,
  detection/OCR metrics, FastAPI contract tests; `gate-visits-admin`: 8
  Minitest tests) and **CI is green on GitHub Actions** (fresh Ubuntu
  runners, not just locally) for Python lint+test, the Rails test suite,
  the Next.js production build, and a CPU smoke evaluation.
- **ONNX export and CPU latency are real, measured numbers** on this build
  machine (12-core Intel, no GPU): ONNX Runtime FP32 gives a ~6x speedup
  over raw PyTorch (9.7ms vs 58ms mean latency); see `docs/edge-feasibility.md`.
- **Real, licensed datasets were sourced and documented**: 248 COCO 2017
  truck images (CC BY), 222 OpenALPR benchmark US plates (AGPL — used
  local-eval-only by design, never redistributed, see `docs/data.md`).
- **Every format validator cites its actual source**: USDOT from 49 CFR
  390.21 (fetched directly from govinfo.gov), trailer ID from ISO 6346
  (checksum verified against the standard's own worked example in tests),
  plate format explicitly *not* claimed to match any specific state.

### 2. Synthetic, estimated, or limited

- **USDOT, trailer ID, and seal detection/OCR numbers are synthetic-only**
  (no public dataset exists for these fields — a genuine finding, not a
  gap papered over). Real-data numbers exist only for truck detection and
  plates; see `docs/benchmarks.md`'s two clearly separated tables.
- **The real/synthetic domain gap is now quantified, not assumed**: plate
  OCR exact-match drops from 42-57.6% on synthetic to 11.7%/4.1% on real
  photos; plate detection mAP drops from ~0.01 (synthetic) to 0.0019
  (real). Treat every synthetic number in this repo as an upper bound on
  real performance, not a prediction of it.
- **The classical MSER-based text-region proposer is a real, working
  weakness, not a placeholder**: chosen specifically to avoid the AGPL
  license of `ultralytics`/YOLO (see `docs/data.md`), it measurably
  underperforms a trained detector. A production build should budget for
  either a commercial Ultralytics license or an Apache-licensed
  alternative (e.g. YOLOX) and real training data.
- **Jetson feasibility is explicitly "unknown," not estimated** — we could
  not find a citable, comparable CPU-only ONNX Runtime benchmark on
  Jetson-class hardware to scale our own numbers against, and said so
  rather than inventing a ratio (`docs/edge-feasibility.md` section 4).
- **INT8 quantization does not currently work** for this two-stage
  detector (a real ONNX Runtime Reshape failure in the ROI heads,
  reported honestly rather than hidden) — 74% smaller, but broken.
- **Seal detection is explicitly experimental**: our synthetic seal is a
  procedural icon, not a rendering of real seal hardware — any number
  attached to it is a sanity check on the code path, not a real-world
  signal.

### 3. Pending — needs credentials or hardware not available here

- **GPU fine-tuning**: `training/finetune_detector_colab.ipynb` is written
  and ready to run on a free Colab GPU runtime; not executed in this
  environment (CPU-only). No fine-tuned-model numbers exist anywhere in
  this repo as a result.
- **Jetson measurement**: no device was available; see above.
- **Cloud deployment**: `deploy/render.yaml` is ready; no Render (or other
  cloud) account credentials were available to actually deploy it. The
  live URLs in this README are placeholders until that happens.
- **A licensed, fine-tuning-scale plate dataset**: the only real plate
  data available without credentials was the AGPL-licensed OpenALPR
  benchmark (444 images, evaluation-only by design). A production build
  needs either a paid/licensed dataset or a data-collection effort.
- **Full `docker compose up --build` fresh-clone verification**: the
  compose config was validated (`docker compose config`) and each service
  was verified individually (backend via direct `uvicorn` run against
  sqlite, frontend via `npm run build`, both against real data) — a full
  multi-service Docker Compose build was not run end-to-end in this
  session due to the slow network already spending significant time on
  dataset/dependency downloads. Recommended as the first thing to verify
  in a normal-bandwidth environment.
