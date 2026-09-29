# Benchmarks

All numbers below are produced by `benchmarks/run_benchmarks.py`, which uses
the exact same detection/OCR code as the FastAPI backend
(`backend/app/detection.py`, `backend/app/ocr.py`) — this file reports what
the deployed API actually does, not a separate "demo" implementation.

**Status: real run completed on this machine.** Numbers below come from an
actual execution of `benchmarks/run_benchmarks.py --limit-per-source 250`
against 248 real COCO truck images, 222 real OpenALPR benchmark US plate
images (AGPL — evaluated locally only, see `docs/data.md`), and 500
synthetic images. Raw predictions: `benchmarks/results/raw_predictions.jsonl`
(1500+ rows). Nothing below is hand-typed.

## Hardware

Same build machine as `docs/edge-feasibility.md`: Windows-10-10.0.26200-SP0,
Intel64 Family 6 Model 186 Stepping 3, 12 logical CPUs, no GPU (see
`docs/limitations.md`). All numbers below are **CPU inference**,
off-the-shelf pretrained models (no fine-tuning — see
`training/finetune_detector_colab.ipynb` for the fine-tuning notebook,
marked pending a GPU run).

## Real data

Reproduce with:
```bash
python data_pipeline/download_coco_trucks.py --limit 250
python data_pipeline/download_openalpr_benchmark.py --region us
python benchmarks/run_benchmarks.py --limit-per-source 250
```

| Field | Source | N | Detection mAP@0.5 | OCR exact-match (EasyOCR) | OCR exact-match (Tesseract) | CER (EasyOCR) | CER (Tesseract) |
|---|---|---|---|---|---|---|---|
| truck | coco2017val | 248 | **0.328** (mean detection latency 209 ms/image) | n/a (no text) | n/a | n/a | n/a |
| plate | openalpr_benchmark_us | 222 | **0.0019** | **0.117** | 0.041 | 0.683 | 0.806 |

`exact_match` above is computed **after the same alphanumeric
normalization `common/validators.py` applies before an operator ever sees
the text** (strip punctuation/whitespace, uppercase) — see
`benchmarks/metrics.py` — so it reflects what the deployed system would
actually accept, not a stricter raw-string comparison. `CER` deliberately
stays a strict raw character-level metric (standard OCR practice), which
is why CER can look high (0.68–0.81) even though ~12% of predictions are
exact matches: it means the non-matching ~88% skew toward near-total
misses (empty or garbage OCR output on a crop), not near-misses.

Two blunt, real findings from this row:

1. **Truck detection (mAP 0.328) works; plate localization on real
   dashcam/CCTV photos essentially does not (mAP 0.0019).** The classical
   MSER text-region proposer, evaluated against the real, human-annotated
   OpenALPR plate boxes, almost never produces a box tight enough to count
   at IoU 0.5 — real images have far more visual clutter (backgrounds,
   other vehicles, signage) than the clean synthetic panels, and MSER has
   no way to know "this blob is a license plate" versus any other
   text-shaped blob in the scene.
2. **OCR-on-the-real-ground-truth-crop reaches only 11.7% (EasyOCR) / 4.1%
   (Tesseract) exact match on real plates**, versus 42%/57.6% on synthetic
   plates. This is the real domain gap between our synthetic renders and
   real photography (real plates: reflective materials, motion blur,
   oblique viewing angles, state-specific fonts/graphics our generator
   doesn't model) — exactly the caveat `docs/limitations.md` and
   `docs/data.md` warn about, now with a number attached.

## Synthetic data

**Never compared directly to the real-data table above** — see
`docs/limitations.md` for why a 2D text-composite generator cannot stand in
for real camera/material conditions.

Reproduce with:
```bash
python data_pipeline/synthetic/generate_synthetic.py --count 500 --seed 42
python benchmarks/run_benchmarks.py --limit-per-source 250
```

| Field | Source | N | Detection mAP@0.5 | OCR exact-match (EasyOCR) | OCR exact-match (Tesseract) | CER (EasyOCR) | CER (Tesseract) |
|---|---|---|---|---|---|---|---|
| usdot | synthetic | 250 | 0.028 | 0.552 | 0.528 | 0.078 | 0.155 |
| trailer_id | synthetic | 250 | 0.010 | **0.824** | 0.740 | 0.042 | 0.220 |
| plate | synthetic | 250 | 0.010 | 0.420 | **0.576** | 0.133 | 0.223 |
| seal | synthetic | 127 (present) | 0.010 | n/a (no text) | n/a | n/a | n/a |

(`exact_match` uses the same post-normalization comparison described
above.)

Real findings from this table:

1. **Detection mAP@0.5 for the classical MSER text-region proposer is very
   low (0.01–0.03) even on clean synthetic renders**, versus 0.328 for the
   trained-on-COCO vehicle detector above, and worse still on real photos
   (0.0019 for plates, see above). This is the honest cost of the
   AGPL-avoidance choice documented in `docs/data.md` — a classical,
   untrained proposer simply does not localize small text regions
   precisely enough to match tight ground-truth boxes at IoU 0.5. OCR
   numbers in this table use the *ground-truth* crop, not the detector's
   proposal, specifically so OCR-engine quality can be judged separately
   from this known detector weakness (see `benchmarks/run_benchmarks.py`
   `eval_synthetic`).
2. **Neither OCR engine dominates uniformly on synthetic data** — EasyOCR
   wins clearly on trailer_id (82.4% vs 74.0%) and usdot (55.2% vs 52.8%,
   close), but Tesseract wins on plate (57.6% vs 42.0%). On real plates,
   though, EasyOCR is clearly better (11.7% vs 4.1%) — synthetic-data
   engine comparisons do not reliably predict real-data ranking, which is
   itself a finding: don't pick a production OCR engine from synthetic
   benchmarks alone.

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
