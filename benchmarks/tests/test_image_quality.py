import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.image_quality import blur_bin, blur_score, brightness, brightness_bin


def test_brightness_black_image_is_zero():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    assert brightness(img) == 0.0


def test_brightness_white_image_is_255():
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    assert brightness(img) == 255.0


def test_brightness_mid_gray():
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    assert abs(brightness(img) - 128.0) < 1.0


def test_blur_score_flat_image_is_near_zero():
    img = np.full((50, 50, 3), 100, dtype=np.uint8)
    assert blur_score(img) < 1.0


def test_blur_score_high_contrast_checkerboard_is_high():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255
    assert blur_score(img) > 100.0


def test_brightness_bin_boundaries():
    assert brightness_bin(30) == "dark (<60)"
    assert brightness_bin(90) == "low (60-120)"
    assert brightness_bin(150) == "normal (120-180)"
    assert brightness_bin(200) == "bright (>=180)"


def test_blur_bin_boundaries():
    assert blur_bin(10) == "very_blurry (<50)"
    assert blur_bin(100) == "blurry (50-150)"
    assert blur_bin(300) == "moderate (150-400)"
    assert blur_bin(500) == "sharp (>=400)"
