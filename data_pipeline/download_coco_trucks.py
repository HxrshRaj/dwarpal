"""Download a real, permissively-licensed subset of COCO images containing trucks.

Source: COCO 2017 val split (https://cocodataset.org), images licensed CC-BY 4.0
(and other Flickr Creative Commons variants) per-image — see the `licenses`
field in the annotation file for each image's exact license and owner. We use
COCO here as the "general vehicle/truck imagery" real dataset, because it is
large, well-annotated, and downloadable without an account.

This does NOT contain plates, USDOT numbers, cab numbers, trailer IDs, or seals
— COCO has no such categories. It is used for general truck/vehicle detection
and for computing image-quality metrics (brightness/blur) on real photos.

Usage:
    python data_pipeline/download_coco_trucks.py --limit 300
"""
import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_BASE_URL = "http://images.cocodataset.org/val2017/"

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw" / "coco_trucks"
ANN_CACHE = ROOT / "raw" / "_coco_annotations"


def download_file(url: str, dest: Path, desc: str) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            bar.update(len(chunk))
    tmp.rename(dest)


def ensure_annotations() -> Path:
    ANN_CACHE.mkdir(parents=True, exist_ok=True)
    ann_json = ANN_CACHE / "instances_val2017.json"
    if ann_json.exists():
        return ann_json
    zip_path = ANN_CACHE / "annotations_trainval2017.zip"
    download_file(ANNOTATIONS_URL, zip_path, "annotations zip")
    with zipfile.ZipFile(zip_path) as zf:
        member = "annotations/instances_val2017.json"
        zf.extract(member, ANN_CACHE)
    (ANN_CACHE / "annotations" / "instances_val2017.json").rename(ann_json)
    zip_path.unlink()
    return ann_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300, help="max number of truck images to download")
    parser.add_argument("--category", default="truck", choices=["truck", "bus", "car"])
    args = parser.parse_args()

    ann_json = ensure_annotations()
    print(f"[info] loading {ann_json}")
    with open(ann_json) as f:
        coco = json.load(f)

    cat_id = next(c["id"] for c in coco["categories"] if c["name"] == args.category)
    images_by_id = {im["id"]: im for im in coco["images"]}
    licenses_by_id = {lic["id"]: lic for lic in coco["licenses"]}

    image_ids_with_cat = sorted({ann["image_id"] for ann in coco["annotations"] if ann["category_id"] == cat_id})
    image_ids_with_cat = image_ids_with_cat[: args.limit]
    print(f"[info] {len(image_ids_with_cat)} images contain category '{args.category}' (capped at --limit)")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for img_id in tqdm(image_ids_with_cat, desc="downloading images"):
        im = images_by_id[img_id]
        fname = im["file_name"]
        dest = RAW_DIR / fname
        if not dest.exists():
            url = IMAGE_BASE_URL + fname
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                dest.write_bytes(r.content)
            except Exception as e:
                print(f"[warn] failed {fname}: {e}", file=sys.stderr)
                continue
        lic = licenses_by_id.get(im.get("license"), {})
        anns = [a for a in coco["annotations"] if a["image_id"] == img_id and a["category_id"] == cat_id]
        manifest.append(
            {
                "file_name": fname,
                "source": "coco2017val",
                "coco_image_id": img_id,
                "width": im["width"],
                "height": im["height"],
                "license_name": lic.get("name"),
                "license_url": lic.get("url"),
                "flickr_url": im.get("flickr_url"),
                "boxes_xywh": [a["bbox"] for a in anns],
                "category": args.category,
            }
        )

    manifest_path = RAW_DIR / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    print(f"[done] {len(manifest)} real images saved to {RAW_DIR}")
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
