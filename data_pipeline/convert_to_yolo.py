"""Convert per-source manifests into a single unified YOLO-format dataset.

YOLO format: one .txt per image, each line `class_id x_center y_center w h`
(all normalized 0-1). Classes are defined in classes.yaml.

Sources handled:
  - coco_trucks manifest.jsonl (real, class: truck)
  - openalpr_benchmark manifest_<region>.jsonl (real, class: plate)
  - synthetic manifest.jsonl (synthetic, classes: plate/usdot/trailer_id/seal)

Each output row also records `source` and `is_synthetic` in a side-car
`index.jsonl` so downstream splitting can guarantee no source leakage between
train/val/test and so real vs synthetic are never silently merged when
reporting metrics.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
PROCESSED_DIR = ROOT / "processed"

CLASSES = ["truck", "plate", "usdot", "trailer_id", "seal"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def xywh_to_yolo(box, img_w, img_h):
    x, y, w, h = box
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    return xc, yc, w / img_w, h / img_h


def convert_coco_trucks():
    manifest_path = RAW_DIR / "coco_trucks" / "manifest.jsonl"
    if not manifest_path.exists():
        print("[skip] coco_trucks manifest not found — run download_coco_trucks.py first")
        return []
    rows = []
    img_out = PROCESSED_DIR / "images"
    lbl_out = PROCESSED_DIR / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for line in manifest_path.read_text().splitlines():
        rec = json.loads(line)
        src_img = RAW_DIR / "coco_trucks" / rec["file_name"]
        if not src_img.exists():
            continue
        out_name = f"coco_{rec['coco_image_id']}.jpg"
        shutil.copy(src_img, img_out / out_name)
        lines = []
        for box in rec["boxes_xywh"]:
            xc, yc, w, h = xywh_to_yolo(box, rec["width"], rec["height"])
            lines.append(f"{CLASS_TO_ID['truck']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        (lbl_out / out_name.replace(".jpg", ".txt")).write_text("\n".join(lines))
        rows.append({"image": out_name, "source": "coco2017val", "is_synthetic": False, "classes": ["truck"]})
    print(f"[ok] converted {len(rows)} coco_trucks images")
    return rows


def convert_synthetic():
    manifest_path = RAW_DIR / "synthetic" / "manifest.jsonl"
    if not manifest_path.exists():
        print("[skip] synthetic manifest not found — run synthetic/generate_synthetic.py first")
        return []
    rows = []
    img_out = PROCESSED_DIR / "images"
    lbl_out = PROCESSED_DIR / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for line in manifest_path.read_text().splitlines():
        rec = json.loads(line)
        src_img = RAW_DIR / "synthetic" / rec["file_name"]
        if not src_img.exists():
            continue
        out_name = f"synth_{rec['file_name']}"
        shutil.copy(src_img, img_out / out_name)
        lines = []
        classes_present = []
        for field in rec["fields"]:
            cls = field["class"]
            classes_present.append(cls)
            xc, yc, w, h = xywh_to_yolo(field["bbox_xywh"], rec["width"], rec["height"])
            lines.append(f"{CLASS_TO_ID[cls]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        (lbl_out / out_name.replace(".png", ".txt")).write_text("\n".join(lines))
        rows.append({"image": out_name, "source": "synthetic", "is_synthetic": True, "classes": classes_present})
    print(f"[ok] converted {len(rows)} synthetic images")
    return rows


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    rows += convert_coco_trucks()
    rows += convert_synthetic()

    index_path = PROCESSED_DIR / "index.jsonl"
    with open(index_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    classes_yaml = PROCESSED_DIR / "classes.yaml"
    classes_yaml.write_text("names:\n" + "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES)) + "\n")

    print(f"[done] {len(rows)} total images indexed at {index_path}")


if __name__ == "__main__":
    main()
