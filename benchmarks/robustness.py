"""Phase 3.2: break OCR accuracy down by brightness bin and blur bin,
computed directly from the crop pixels (not guessed), and save example
failure images per engine/bin. Reads benchmarks/results/raw_predictions.jsonl
(written by run_benchmarks.py) — run that first.

Usage:
    python benchmarks/robustness.py
Writes:
    benchmarks/results/robustness_report.json
    benchmarks/results/failures/<engine>_<field>_<bin>_<i>.txt (metadata;
        see script for why we don't re-save the crop image bytes here)
"""
import json
from collections import defaultdict
from pathlib import Path

from benchmarks.image_quality import blur_bin, brightness_bin

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"


def main():
    raw_path = RESULTS_DIR / "raw_predictions.jsonl"
    if not raw_path.exists():
        raise SystemExit("run benchmarks/run_benchmarks.py first")

    rows = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]

    by_brightness = defaultdict(lambda: {"n": 0, "correct": 0})
    by_blur = defaultdict(lambda: {"n": 0, "correct": 0})
    failures = []

    for r in rows:
        b_bin = brightness_bin(r["brightness"])
        bl_bin = blur_bin(r["blur_score"])
        key_b = (r["source"], r["field_class"], r["engine"], b_bin)
        key_bl = (r["source"], r["field_class"], r["engine"], bl_bin)
        by_brightness[key_b]["n"] += 1
        by_brightness[key_b]["correct"] += int(r["exact_match"])
        by_blur[key_bl]["n"] += 1
        by_blur[key_bl]["correct"] += int(r["exact_match"])

        if not r["exact_match"] and len(failures) < 200:
            failures.append(r)

    def fmt(d, dims):
        out = []
        for key, v in d.items():
            row = dict(zip(dims, key))
            row["n"] = v["n"]
            row["accuracy"] = v["correct"] / v["n"] if v["n"] else None
            out.append(row)
        return out

    report = {
        "by_brightness": fmt(by_brightness, ["source", "field_class", "engine", "brightness_bin"]),
        "by_blur": fmt(by_blur, ["source", "field_class", "engine", "blur_bin"]),
        "n_failures_sampled": len(failures),
    }

    (RESULTS_DIR / "robustness_report.json").write_text(json.dumps(report, indent=2))

    failures_dir = RESULTS_DIR / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    # We only persist failure METADATA (predicted vs ground truth text,
    # brightness/blur, source), not re-saved crop bytes, so this repo never
    # accidentally bundles AGPL-restricted OpenALPR images (see docs/data.md).
    (failures_dir / "failure_examples.jsonl").write_text("\n".join(json.dumps(f) for f in failures))

    print(json.dumps(report, indent=2))
    print(f"[done] wrote {RESULTS_DIR / 'robustness_report.json'} and {len(failures)} failure examples")


if __name__ == "__main__":
    main()
