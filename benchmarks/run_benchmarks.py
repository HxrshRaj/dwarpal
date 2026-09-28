"""Phase 3 benchmark script — produces the numbers reported in
docs/benchmarks.md. Uses the exact same detection/OCR code as the FastAPI
backend (backend/app/detection.py, backend/app/ocr.py) so a reported number
reflects what the deployed API actually does.

Real and synthetic results are computed and written SEPARATELY and never
averaged together (see docs/data.md, docs/limitations.md).

Usage:
    python benchmarks/run_benchmarks.py --limit-per-source 100
Writes:
    benchmarks/results/results.json   (full detail, all predictions)
    benchmarks/results/summary.json   (per field/source table, matches the
                                        schema frontend/public/data/benchmark_results.json expects)
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import mlflow

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.detection import TextRegionProposer, VehicleDetector  # noqa: E402
from app.ocr import EasyOcrEngine, TesseractEngine  # noqa: E402
from benchmarks.image_quality import blur_score, brightness  # noqa: E402
from benchmarks.metrics import average_precision_at_iou, character_error_rate, exact_match, mean_average_precision  # noqa: E402

RESULTS_DIR = ROOT / "benchmarks" / "results"
RAW_DIR = ROOT / "data_pipeline" / "raw"

RAW_PREDICTIONS = []  # populated by eval_* functions, dumped to raw_predictions.jsonl


def _record_prediction(source, field_class, is_synthetic, engine, pred_text, gt_text, confidence, image_bgr):
    RAW_PREDICTIONS.append(
        {
            "source": source,
            "field_class": field_class,
            "is_synthetic": is_synthetic,
            "engine": engine,
            "pred_text": pred_text,
            "gt_text": gt_text,
            "confidence": confidence,
            "exact_match": exact_match(pred_text, gt_text),
            "cer": character_error_rate(pred_text, gt_text),
            "brightness": brightness(image_bgr),
            "blur_score": blur_score(image_bgr),
        }
    )


def eval_coco_trucks(vehicle_detector, limit):
    """Real data: vehicle/truck detection mAP@0.5 against COCO ground truth."""
    manifest_path = RAW_DIR / "coco_trucks" / "manifest.jsonl"
    if not manifest_path.exists():
        print("[skip] coco_trucks not downloaded")
        return None

    records = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()][:limit]
    per_image_ap = []
    latencies = []
    for rec in records:
        img_path = RAW_DIR / "coco_trucks" / rec["file_name"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        dets = vehicle_detector.detect(img_rgb)
        latencies.append(time.perf_counter() - t0)
        preds = [(d.bbox_xywh, d.score) for d in dets if d.label == "truck"]
        ap = average_precision_at_iou(preds, rec["boxes_xywh"])
        per_image_ap.append(ap)

    return {
        "field": "truck",
        "source": "coco2017val",
        "n": len(per_image_ap),
        "detection_map50": mean_average_precision(per_image_ap),
        "mean_latency_ms": (sum(latencies) / len(latencies) * 1000) if latencies else None,
    }


def eval_openalpr_plates(text_proposer, easy_ocr, tesseract, limit, region="us"):
    """Real data: plate detection recall (via text-region proposer) +
    OCR exact-match/CER on the ground-truth crop, for both engines."""
    manifest_path = RAW_DIR / "openalpr_benchmark" / f"manifest_{region}.jsonl"
    img_dir = RAW_DIR / "openalpr_benchmark"
    if not manifest_path.exists():
        print(f"[skip] openalpr_benchmark ({region}) not downloaded")
        return None

    records = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()][:limit]

    per_image_ap = []
    engine_results = {"easyocr": {"exact": 0, "cer": []}, "tesseract": {"exact": 0, "cer": []}}
    n_with_text = 0

    for rec in records:
        img_path = img_dir / rec["file_name"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        gt = rec["gt_raw"]
        # OpenALPR benchmark txt format: "position_plate x y w h" (pixels), "plate <TEXT>"
        gt_box = None
        if "position_plate" in gt and len(gt["position_plate"]) == 4:
            gt_box = [float(v) for v in gt["position_plate"]]
        gt_text = gt.get("plate", [None])[0]

        if gt_box:
            proposals = text_proposer.propose(img_bgr)
            preds = [(p.bbox_xywh, p.score) for p in proposals]
            ap = average_precision_at_iou(preds, [gt_box])
            per_image_ap.append(ap)

        if gt_box and gt_text:
            x, y, w, h = [int(v) for v in gt_box]
            crop = img_bgr[max(0, y) : y + h, max(0, x) : x + w]
            if crop.size == 0:
                continue
            n_with_text += 1
            for name, engine in [("easyocr", easy_ocr), ("tesseract", tesseract)]:
                res = engine.read(crop)
                if exact_match(res.text, gt_text):
                    engine_results[name]["exact"] += 1
                engine_results[name]["cer"].append(character_error_rate(res.text, gt_text))
                _record_prediction(f"openalpr_benchmark_{region}", "plate", False, name, res.text, gt_text, res.confidence, crop)

    out = {
        "field": "plate",
        "source": f"openalpr_benchmark_{region}",
        "n": len(per_image_ap),
        "detection_map50": mean_average_precision(per_image_ap) if per_image_ap else None,
    }
    for name in ("easyocr", "tesseract"):
        er = engine_results[name]
        out[f"{name}_ocr_exact_match"] = (er["exact"] / n_with_text) if n_with_text else None
        out[f"{name}_cer"] = (sum(er["cer"]) / len(er["cer"])) if er["cer"] else None
    out["n_ocr"] = n_with_text
    return out


def eval_synthetic(text_proposer, easy_ocr, tesseract, limit):
    """Synthetic data: same metrics as above, per field class, using our own
    generator's ground truth. NEVER merged with the real-data numbers above."""
    manifest_path = RAW_DIR / "synthetic" / "manifest.jsonl"
    if not manifest_path.exists():
        print("[skip] synthetic data not generated")
        return []

    records = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()][:limit]
    by_class = {}
    for rec in records:
        img_bgr = cv2.imread(str(RAW_DIR / "synthetic" / rec["file_name"]))
        if img_bgr is None:
            continue
        proposals = text_proposer.propose(img_bgr)
        pred_boxes = [(p.bbox_xywh, p.score) for p in proposals]

        for field in rec["fields"]:
            cls = field["class"]
            by_class.setdefault(cls, {"ap": [], "easyocr": {"exact": 0, "cer": []}, "tesseract": {"exact": 0, "cer": []}, "n_ocr": 0})
            gt_box = field["bbox_xywh"]
            ap = average_precision_at_iou(pred_boxes, [gt_box])
            by_class[cls]["ap"].append(ap)

            if cls == "seal":
                continue
            x, y, w, h = [int(v) for v in gt_box]
            crop = img_bgr[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            by_class[cls]["n_ocr"] += 1
            for name, engine in [("easyocr", easy_ocr), ("tesseract", tesseract)]:
                res = engine.read(crop)
                if exact_match(res.text, field["text"]):
                    by_class[cls][name]["exact"] += 1
                by_class[cls][name]["cer"].append(character_error_rate(res.text, field["text"]))
                _record_prediction("synthetic", cls, True, name, res.text, field["text"], res.confidence, crop)

    results = []
    for cls, data in by_class.items():
        row = {
            "field": cls,
            "source": "synthetic",
            "n": len(data["ap"]),
            "detection_map50": mean_average_precision(data["ap"]),
        }
        for name in ("easyocr", "tesseract"):
            er = data[name]
            row[f"{name}_ocr_exact_match"] = (er["exact"] / data["n_ocr"]) if data["n_ocr"] else None
            row[f"{name}_cer"] = (sum(er["cer"]) / len(er["cer"])) if er["cer"] else None
        row["n_ocr"] = data["n_ocr"]
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-source", type=int, default=100)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"file:{ROOT / 'mlruns'}"))
    mlflow.set_experiment("dwarpal-gate-benchmarks")

    print("[init] loading models (torchvision Faster R-CNN, EasyOCR, Tesseract)...")
    vehicle_detector = VehicleDetector()
    text_proposer = TextRegionProposer()
    easy_ocr = EasyOcrEngine()
    tesseract = TesseractEngine()

    real_results = []
    r1 = eval_coco_trucks(vehicle_detector, args.limit_per_source)
    if r1:
        real_results.append(r1)
    r2 = eval_openalpr_plates(text_proposer, easy_ocr, tesseract, args.limit_per_source)
    if r2:
        real_results.append(r2)

    synthetic_results = eval_synthetic(text_proposer, easy_ocr, tesseract, args.limit_per_source)

    full = {"real": real_results, "synthetic": synthetic_results}
    (RESULTS_DIR / "results.json").write_text(json.dumps(full, indent=2))

    with mlflow.start_run(run_name="run_benchmarks"):
        mlflow.log_param("limit_per_source", args.limit_per_source)
        for row in real_results + synthetic_results:
            prefix = f"{row['source']}_{row['field']}".replace(" ", "_")
            for key in ("detection_map50", "easyocr_ocr_exact_match", "easyocr_cer", "tesseract_ocr_exact_match", "tesseract_cer"):
                val = row.get(key)
                if val is not None:
                    mlflow.log_metric(f"{prefix}_{key}", val)
        mlflow.log_artifact(str(RESULTS_DIR / "results.json"))

    raw_path = RESULTS_DIR / "raw_predictions.jsonl"
    with open(raw_path, "w") as f:
        for row in RAW_PREDICTIONS:
            f.write(json.dumps(row) + "\n")

    print(json.dumps(full, indent=2))
    print(f"[done] wrote {RESULTS_DIR / 'results.json'} and {raw_path} ({len(RAW_PREDICTIONS)} raw predictions)")


if __name__ == "__main__":
    main()
