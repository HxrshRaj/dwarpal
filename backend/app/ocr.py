"""OCR stage: run two engines on a cropped region and return both results
for comparison (per project brief: "run at least two OCR engines ... and
compare them").

Engines:
  - EasyOCR (Apache-2.0 license) — deep-learning based, CPU-capable.
  - Tesseract 5 (Apache-2.0 license) via pytesseract — classical LSTM OCR
    engine, installed locally (winget install tesseract-ocr.tesseract).

Both return (text, confidence in [0,1]).
"""
import shutil
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class OcrResult:
    engine: str
    text: str
    confidence: float


class EasyOcrEngine:
    def __init__(self, languages=("en",)):
        import easyocr

        self.reader = easyocr.Reader(list(languages), gpu=False)

    def read(self, image_bgr: np.ndarray) -> OcrResult:
        results = self.reader.readtext(image_bgr)
        if not results:
            return OcrResult("easyocr", "", 0.0)
        # concatenate all detected text in the crop, weight confidence by text length
        texts, confs, lens = [], [], []
        for _, text, conf in results:
            texts.append(text)
            confs.append(conf)
            lens.append(max(len(text), 1))
        joined = " ".join(texts)
        weighted_conf = sum(c * l for c, l in zip(confs, lens)) / sum(lens)
        return OcrResult("easyocr", joined, float(weighted_conf))


class TesseractEngine:
    def __init__(self, tesseract_cmd: Optional[str] = None):
        import pytesseract

        self.pytesseract = pytesseract
        cmd = tesseract_cmd or shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        pytesseract.pytesseract.tesseract_cmd = cmd

    def read(self, image_bgr: np.ndarray) -> OcrResult:
        import cv2

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        data = self.pytesseract.image_to_data(
            gray, config="--psm 7", output_type=self.pytesseract.Output.DICT
        )
        words = [w for w in data["text"] if w.strip()]
        confs = [float(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and float(c) >= 0]
        if not words:
            return OcrResult("tesseract", "", 0.0)
        joined = " ".join(words)
        avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
        return OcrResult("tesseract", joined, avg_conf)


def run_both_engines(image_bgr: np.ndarray, easy: EasyOcrEngine, tess: TesseractEngine):
    return [easy.read(image_bgr), tess.read(image_bgr)]


def pick_best(results) -> OcrResult:
    """Pick the higher-confidence non-empty result; prefer non-empty over empty."""
    non_empty = [r for r in results if r.text.strip()]
    pool = non_empty or results
    return max(pool, key=lambda r: r.confidence)
