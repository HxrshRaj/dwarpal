"""Phase 2.4: choose the confidence-gating threshold per field FROM DATA
(precision/coverage tradeoff on real-data predictions), not by guessing.
Reads benchmarks/results/raw_predictions.jsonl (run_benchmarks.py first).

"Precision" here = among predictions kept (confidence >= threshold), the
fraction that were an exact match. "Coverage" = fraction of all predictions
kept. We pick, per field, the lowest threshold that keeps precision >= 0.90
on REAL data only (synthetic predictions are excluded from this selection,
consistent with docs/data.md never mixing the two) — falling back to the
highest-precision threshold available if 0.90 is never reached, and noting
that explicitly.

Usage:
    python benchmarks/tune_thresholds.py
Writes:
    common/confidence_thresholds.json  (loaded by common/validators.py if present)
    docs/benchmarks.md's threshold table is filled in by hand from this
    output — see the printed table.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"
TARGET_PRECISION = 0.90
CANDIDATE_THRESHOLDS = [0.1 * i for i in range(10)]


def main():
    raw_path = RESULTS_DIR / "raw_predictions.jsonl"
    if not raw_path.exists():
        raise SystemExit("run benchmarks/run_benchmarks.py first")

    rows = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
    real_rows = [r for r in rows if not r["is_synthetic"]]

    by_field = defaultdict(list)
    for r in real_rows:
        by_field[r["field_class"]].append(r)

    chosen = {}
    curves = {}
    for field, field_rows in by_field.items():
        curve = []
        best_threshold = None
        for t in CANDIDATE_THRESHOLDS:
            kept = [r for r in field_rows if r["confidence"] >= t]
            if not kept:
                curve.append({"threshold": t, "precision": None, "coverage": 0.0, "n_kept": 0})
                continue
            precision = sum(r["exact_match"] for r in kept) / len(kept)
            coverage = len(kept) / len(field_rows)
            curve.append({"threshold": t, "precision": precision, "coverage": coverage, "n_kept": len(kept)})
            if precision >= TARGET_PRECISION and best_threshold is None:
                best_threshold = t
        curves[field] = curve
        if best_threshold is None:
            # never reached target precision — pick the threshold with the
            # highest observed precision instead, and say so honestly
            valid = [c for c in curve if c["precision"] is not None]
            best_threshold = max(valid, key=lambda c: c["precision"])["threshold"] if valid else 0.5
            print(f"[warn] field '{field}': never reached {TARGET_PRECISION} precision on real data; "
                  f"using best-available threshold {best_threshold} instead")
        chosen[field] = best_threshold

    out_path = ROOT / "common" / "confidence_thresholds.json"
    out_path.write_text(json.dumps(chosen, indent=2))

    report_path = RESULTS_DIR / "threshold_tuning_report.json"
    report_path.write_text(json.dumps({"chosen": chosen, "curves": curves, "target_precision": TARGET_PRECISION}, indent=2))

    print(json.dumps({"chosen_thresholds": chosen}, indent=2))
    print(f"[done] wrote {out_path} and {report_path}")


if __name__ == "__main__":
    main()
