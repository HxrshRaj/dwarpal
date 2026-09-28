"""Download the OpenALPR benchmark dataset (real, human-annotated plate photos).

Source: https://github.com/openalpr/benchmarks (444 real CCTV/dashcam photos
across US/EU/Brazil plates, each with a ground-truth bounding box + plate
string).

LICENSE WARNING — read before using this data beyond local evaluation:
The openalpr/benchmarks repository is licensed AGPL-3.0 in full (no per-file
override was found in the repo). AGPL is a copyleft license with network-use
disclosure obligations. We therefore:
  - use this dataset ONLY for local benchmarking/evaluation (never for
    training weights we redistribute, and never bundled into this repo's
    git history — this script downloads to data_pipeline/raw/, which is
    gitignored);
  - do NOT redistribute these images or derived crops anywhere in this
    project's committed files or Docker images;
  - report this choice and its implication in docs/data.md.

Usage:
    python data_pipeline/download_openalpr_benchmark.py --region us
"""
import argparse
import json
from pathlib import Path

import requests
from tqdm import tqdm

API_BASE = "https://api.github.com/repos/openalpr/benchmarks/contents/endtoend"
RAW_BASE = "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend"

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw" / "openalpr_benchmark"


def list_dir(region: str) -> list:
    resp = requests.get(f"{API_BASE}/{region}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_gt_txt(text: str) -> dict:
    """OpenALPR benchmark txt format: one line 'position_plate <x> <y> <w> <h>'
    followed by 'plate <TEXT>' or similar; be liberal parsing key: value pairs."""
    result = {}
    for line in text.strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        key = parts[0]
        result[key] = parts[1:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us", choices=["us", "eu", "br"])
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    entries = list_dir(args.region)
    jpgs = [e for e in entries if e["name"].endswith(".jpg")]
    print(f"[info] {len(jpgs)} real labeled plate images found in endtoend/{args.region}")

    manifest = []
    for e in tqdm(jpgs, desc=f"downloading {args.region} plates"):
        stem = e["name"][:-4]
        img_dest = RAW_DIR / e["name"]
        txt_dest = RAW_DIR / f"{stem}.txt"
        if not img_dest.exists():
            r = requests.get(f"{RAW_BASE}/{args.region}/{e['name']}", timeout=30)
            r.raise_for_status()
            img_dest.write_bytes(r.content)
        if not txt_dest.exists():
            r = requests.get(f"{RAW_BASE}/{args.region}/{stem}.txt", timeout=30)
            r.raise_for_status()
            txt_dest.write_bytes(r.content)
        gt = parse_gt_txt(txt_dest.read_text())
        manifest.append({"file_name": e["name"], "source": f"openalpr_benchmark_{args.region}", "region": args.region, "gt_raw": gt})

    manifest_path = RAW_DIR / f"manifest_{args.region}.jsonl"
    with open(manifest_path, "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    print(f"[done] {len(manifest)} real labeled plate images saved to {RAW_DIR}")
    print("[license] AGPL-3.0 — local evaluation only, not redistributed. See docs/data.md")


if __name__ == "__main__":
    main()
