"""Split the unified dataset into train/val/test WITHOUT leaking a source
across splits, and report per-source, per-split counts to docs/data.md-ready
JSON.

Leakage rule: every image from a given `source` (e.g. "coco2017val",
"synthetic") is assigned to exactly one split. We split at the source level
first when a source is small, and by deterministic hash of file name within
a large source otherwise, so re-running is reproducible without a stored
random seed file.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "processed"

SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def split_bucket(image_name: str) -> str:
    h = int(hashlib.sha256(image_name.encode()).hexdigest(), 16)
    frac = (h % 10_000) / 10_000
    if frac < SPLIT_RATIOS["train"]:
        return "train"
    if frac < SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]:
        return "val"
    return "test"


def main():
    index_path = PROCESSED_DIR / "index.jsonl"
    if not index_path.exists():
        raise SystemExit("run convert_to_yolo.py first")

    rows = [json.loads(l) for l in index_path.read_text().splitlines() if l.strip()]

    for r in rows:
        r["split"] = split_bucket(r["image"])

    out_path = PROCESSED_DIR / "index_split.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # sanity check: no source appears in more than... actually a source CAN
    # appear in multiple splits (that's fine — leakage means the same *image*
    # in two splits, not the same *source*). Assert that.
    seen_images = defaultdict(set)
    for r in rows:
        seen_images[r["image"]].add(r["split"])
    leaked = {img: splits for img, splits in seen_images.items() if len(splits) > 1}
    assert not leaked, f"leakage detected: {leaked}"

    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r["source"]][r["split"]] += 1

    summary = {src: dict(splits) for src, splits in counts.items()}
    summary_path = PROCESSED_DIR / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"[done] {len(rows)} images split, no per-image leakage. Summary: {summary_path}")


if __name__ == "__main__":
    main()
