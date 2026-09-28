# Benchmarks

All numbers below are produced by `benchmarks/run_benchmarks.py`, which uses
the exact same detection/OCR code as the FastAPI backend
(`backend/app/detection.py`, `backend/app/ocr.py`) — this file reports what
the deployed API actually does, not a separate "demo" implementation.

**Status: pending first full run in this environment.** This section is
filled in as soon as `python data_pipeline/download_coco_trucks.py`,
`python data_pipeline/download_openalpr_benchmark.py`, and
`python benchmarks/run_benchmarks.py` complete — see the commit history for
the run that populated the tables below. Until that commit, treat every
number here as **not yet measured**.

## Hardware

- Machine: see `docs/edge-feasibility.md` for the exact CPU/OS this was run
  on (same machine — no GPU was available for this project, see
  `docs/limitations.md`).
- All numbers below are **CPU inference**, off-the-shelf pretrained models
  (no fine-tuning — see `training/finetune_detector_colab.ipynb` for the
  fine-tuning notebook, marked pending a GPU run).

## Real data

Reproduce with:
```bash
python data_pipeline/download_coco_trucks.py --limit 300
python data_pipeline/download_openalpr_benchmark.py --region us
python benchmarks/run_benchmarks.py --limit-per-source 100
```

| Field | Source | N | Detection mAP@0.5 | OCR exact-match (EasyOCR) | OCR exact-match (Tesseract) | CER (EasyOCR) | CER (Tesseract) |
|---|---|---|---|---|---|---|---|
| _pending_ | | | | | | | |

## Synthetic data

**Never compared directly to the real-data table above** — see
`docs/limitations.md` for why a 2D text-composite generator cannot stand in
for real camera/material conditions.

Reproduce with:
```bash
python data_pipeline/synthetic/generate_synthetic.py --count 500
python benchmarks/run_benchmarks.py --limit-per-source 100
```

| Field | Source | N | Detection mAP@0.5 | OCR exact-match (EasyOCR) | OCR exact-match (Tesseract) | CER (EasyOCR) | CER (Tesseract) |
|---|---|---|---|---|---|---|---|
| _pending_ | | | | | | | |

## Robustness analysis

`benchmarks/robustness.py` (see script for the exact brightness/blur
computation) breaks OCR accuracy down by image brightness bin and blur
score. Filled in alongside the tables above; example failure images are
saved to `benchmarks/results/failures/` when this script runs.

## Confidence threshold selection

`benchmarks/tune_thresholds.py` sweeps the OCR confidence threshold against
the real-data precision/coverage tradeoff and writes the chosen operating
point to `common/confidence_thresholds.json`, which overrides the
placeholder defaults in `common/validators.py`. Pending the same first run.

## Before/after fine-tuning comparison

Not run in this environment (no GPU). See
`training/finetune_detector_colab.ipynb` — run it on Colab, then copy
`benchmarks/results/finetune_before_after.json` back into this repo and this
section will be updated with real before/after numbers on the same held-out
synthetic test split.
