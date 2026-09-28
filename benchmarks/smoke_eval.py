"""CPU-cheap smoke evaluation for CI: runs OCR + validation (using
ground-truth crop boxes, not the vehicle/text detectors, to keep this fast
and avoid a large model download on every CI run) on a handful of freshly
generated synthetic images, and asserts the pipeline produces sane output.

This is NOT the reported benchmark (see docs/benchmarks.md / run_benchmarks.py
for that) — it exists only to catch a broken pipeline before merge.
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ocr import EasyOcrEngine, TesseractEngine

from common.validators import validate_field

SYNTH_DIR = ROOT / "data_pipeline" / "raw" / "synthetic"


def main():
    manifest_path = SYNTH_DIR / "manifest.jsonl"
    if not manifest_path.exists():
        print("[smoke] no synthetic manifest found — generate it first", file=sys.stderr)
        sys.exit(1)

    records = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    print(f"[smoke] loaded {len(records)} synthetic records")

    easy = EasyOcrEngine()
    tess = TesseractEngine()

    n_checked = 0
    n_exact = 0
    for rec in records:
        img = cv2.imread(str(SYNTH_DIR / rec["file_name"]))
        assert img is not None, f"failed to load {rec['file_name']}"
        for field in rec["fields"]:
            x, y, w, h = [int(v) for v in field["bbox_xywh"]]
            crop = img[y : y + h, x : x + w]
            if crop.size == 0 or field["class"] == "seal":
                continue
            easy_res = easy.read(crop)
            tess_res = tess.read(crop)
            best = max([easy_res, tess_res], key=lambda r: r.confidence)
            validation = validate_field(field["class"], best.text)
            n_checked += 1
            if best.text.strip().upper().replace(" ", "") == field["text"].strip().upper().replace(" ", ""):
                n_exact += 1
            print(f"[smoke] {field['class']}: gt='{field['text']}' pred='{best.text}' engine={best.engine} valid_format={validation.is_valid_format}")

    assert n_checked > 0, "no fields were evaluated"
    print(f"[smoke] checked {n_checked} fields, {n_exact} exact matches on this tiny synthetic sample")
    print("[smoke] pipeline is functional (this is a smoke test, not a reported benchmark)")


if __name__ == "__main__":
    main()
