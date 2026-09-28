"""Synthetic data generator for the fields that have no (or almost no) public
labeled data: USDOT numbers, trailer IDs, and license plates rendered onto
procedural panel backgrounds, plus a seal presence/absence overlay.

EVERYTHING PRODUCED HERE IS SYNTHETIC. It is written to
data_pipeline/raw/synthetic/ and every manifest row is tagged
`"is_synthetic": true`. Never merge these images or their metrics with real
data — see docs/data.md and docs/benchmarks.md, which report real and
synthetic numbers in separate tables.

Format references used to make the rendered text plausible (not just random
strings):
  - USDOT: "USDOT" + FMCSA-issued numeric ID, per 49 CFR 390.21(b)(2)
    (govinfo.gov CFR-2024-title49-vol5, sec 390.21). The regulation does not
    mandate a digit count; we render 6-8 digits, the commonly observed range
    in FMCSA's public SAFER registration system.
  - Trailer ID (intermodal chassis convention): ISO 6346 — 4-letter prefix
    (3-letter owner code + equipment category letter, commonly 'U') + 6-digit
    serial + 1 check digit, mod-11 algorithm (see iso6346_check_digit below).
    Domestic dry-van trailers often use company-specific IDs instead; we
    render both styles.
  - License plate: no single national format exists (plates are state-issued
    in the US). We render generic alphanumeric strings of plausible length
    (5-7 characters) and do NOT claim this matches any specific state's
    format. See docs/data.md.

Usage:
    python data_pipeline/synthetic/generate_synthetic.py --count 500
"""
import argparse
import json
import random
import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "raw" / "synthetic"

ISO6346_LETTER_VALUES = {c: v for c, v in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", list(range(10, 34)))}
# ISO 6346 skips multiples of 11 in the letter value table
_vals = []
v = 10
for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    while v % 11 == 0:
        v += 1
    ISO6346_LETTER_VALUES[c] = v
    v += 1


def iso6346_check_digit(owner_and_serial: str) -> int:
    """owner_and_serial: 4 letters + 6 digits (10 chars). Returns check digit 0-9."""
    total = 0
    for i, ch in enumerate(owner_and_serial):
        val = ISO6346_LETTER_VALUES[ch] if ch.isalpha() else int(ch)
        total += val * (2 ** i)
    return total % 11 % 10


def random_usdot() -> str:
    digits = random.randint(6, 8)
    return "USDOT " + "".join(random.choices(string.digits, k=digits))


def random_trailer_id(style: str) -> str:
    if style == "iso6346":
        owner = "".join(random.choices(string.ascii_uppercase, k=3)) + "U"
        serial = "".join(random.choices(string.digits, k=6))
        chk = iso6346_check_digit(owner + serial)
        return f"{owner}{serial}{chk}"
    prefix = "".join(random.choices(string.ascii_uppercase, k=random.choice([2, 3])))
    digits = "".join(random.choices(string.digits, k=random.choice([5, 6])))
    return f"{prefix}{digits}"


def random_plate() -> str:
    length = random.randint(5, 7)
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))


def load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consolab.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def draw_panel(w: int, h: int, base_color, text: str, font_size: int, text_color=(20, 20, 20)):
    img = Image.new("RGB", (w, h), base_color)
    draw = ImageDraw.Draw(img)
    # subtle panel noise/texture
    noise = (np.random.randn(h, w, 3) * 6).astype(np.int16)
    arr = np.array(img).astype(np.int16) + noise
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    font = load_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = ((w - tw) / 2 - bbox[0], (h - th) / 2 - bbox[1])
    draw.text(pos, text, font=font, fill=text_color)
    return img


def draw_seal(present: bool, size: int = 48):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    if present:
        draw = ImageDraw.Draw(img)
        draw.ellipse([2, 2, size - 2, size - 2], fill=(200, 40, 40, 255), outline=(120, 20, 20, 255), width=2)
        draw.rectangle([size * 0.4, 0, size * 0.6, size * 0.3], fill=(120, 120, 120, 255))
    return img


def apply_augmentation(img: Image.Image, rng: random.Random) -> Image.Image:
    # brightness / low light / glare
    arr = np.array(img).astype(np.float32)
    brightness = rng.uniform(0.45, 1.6)
    arr = arr * brightness
    if rng.random() < 0.25:
        # glare: bright elliptical patch
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        rad = rng.uniform(w * 0.15, w * 0.4)
        mask = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad ** 2)))
        arr = arr + mask[..., None] * rng.uniform(80, 160)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # blur
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 2.5)))

    # dirt: random dark specks
    if rng.random() < 0.4:
        arr = np.array(img)
        n_specks = rng.randint(20, 120)
        for _ in range(n_specks):
            x, y = rng.randrange(arr.shape[1]), rng.randrange(arr.shape[0])
            r = rng.randint(1, 3)
            arr[max(0, y - r):y + r, max(0, x - r):x + r] = rng.randint(20, 60)
        img = Image.fromarray(arr)

    # skew (perspective-ish via affine shear)
    if rng.random() < 0.4:
        w, h = img.size
        shear = rng.uniform(-0.15, 0.15)
        img = img.transform(
            (w, h), Image.AFFINE, (1, shear, -shear * h / 2, 0, 1, 0), resample=Image.BICUBIC, fillcolor=(30, 30, 30)
        )

    return img


def generate_one(idx: int, rng: random.Random) -> dict:
    canvas_w, canvas_h = 640, 480
    canvas = Image.new("RGB", (canvas_w, canvas_h), (60, 60, 65))
    # background texture (procedural truck-side panel)
    bg_noise = (np.random.randn(canvas_h, canvas_w, 3) * 10 + 70).astype(np.uint8)
    canvas = Image.fromarray(bg_noise)

    fields = []

    # USDOT text field
    usdot_text = random_usdot()
    usdot_w, usdot_h = rng.randint(200, 280), rng.randint(40, 60)
    usdot_panel = draw_panel(usdot_w, usdot_h, (235, 235, 235), usdot_text, font_size=int(usdot_h * 0.6))
    ux, uy = rng.randint(20, canvas_w - usdot_w - 20), rng.randint(20, 120)
    canvas.paste(usdot_panel, (ux, uy))
    fields.append({"class": "usdot", "text": usdot_text, "bbox_xywh": [ux, uy, usdot_w, usdot_h]})

    # trailer id field
    style = rng.choice(["iso6346", "domestic"])
    trailer_text = random_trailer_id("iso6346" if style == "iso6346" else "domestic")
    tw, th = rng.randint(220, 300), rng.randint(45, 65)
    trailer_panel = draw_panel(tw, th, (250, 250, 240), trailer_text, font_size=int(th * 0.55))
    tx, ty = rng.randint(20, canvas_w - tw - 20), rng.randint(140, 260)
    canvas.paste(trailer_panel, (tx, ty))
    fields.append({"class": "trailer_id", "text": trailer_text, "bbox_xywh": [tx, ty, tw, th]})

    # plate field
    plate_text = random_plate()
    pw, ph = rng.randint(140, 180), rng.randint(50, 70)
    plate_color = rng.choice([(255, 255, 255), (230, 230, 150), (200, 220, 255)])
    plate_panel = draw_panel(pw, ph, plate_color, plate_text, font_size=int(ph * 0.55))
    px, py = rng.randint(20, canvas_w - pw - 20), rng.randint(300, 380)
    canvas.paste(plate_panel, (px, py))
    fields.append({"class": "plate", "text": plate_text, "bbox_xywh": [px, py, pw, ph]})

    # seal (present ~50% of the time)
    seal_present = rng.random() < 0.5
    seal_size = 44
    seal_img = draw_seal(seal_present, size=seal_size)
    sx, sy = rng.randint(20, canvas_w - seal_size - 20), rng.randint(400, 440)
    if seal_present:
        canvas.paste(seal_img, (sx, sy), seal_img)
        fields.append({"class": "seal", "text": "present", "bbox_xywh": [sx, sy, seal_size, seal_size]})

    canvas = apply_augmentation(canvas, rng)

    fname = f"synth_{idx:05d}.png"
    canvas.save(OUT_DIR / fname)

    return {
        "file_name": fname,
        "source": "synthetic",
        "is_synthetic": True,
        "width": canvas_w,
        "height": canvas_h,
        "fields": fields,
        "seal_present": seal_present,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    manifest = []
    for i in range(args.count):
        manifest.append(generate_one(i, rng))

    manifest_path = OUT_DIR / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    print(f"[done] {len(manifest)} SYNTHETIC images written to {OUT_DIR}")
    print("[reminder] synthetic data — never merge with real-data metrics")


if __name__ == "__main__":
    main()
