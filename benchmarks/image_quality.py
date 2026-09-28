"""Cheap, real image-quality metrics computed directly from pixels — used to
bucket OCR accuracy by condition in docs/benchmarks.md's robustness section.
No external model required."""
import cv2
import numpy as np


def brightness(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def blur_score(image_bgr: np.ndarray) -> float:
    """Variance of the Laplacian — a standard, real focus-measure. Lower
    means blurrier."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_bin(value: float) -> str:
    if value < 60:
        return "dark (<60)"
    if value < 120:
        return "low (60-120)"
    if value < 180:
        return "normal (120-180)"
    return "bright (>=180)"


def blur_bin(value: float) -> str:
    if value < 50:
        return "very_blurry (<50)"
    if value < 150:
        return "blurry (50-150)"
    if value < 400:
        return "moderate (150-400)"
    return "sharp (>=400)"
