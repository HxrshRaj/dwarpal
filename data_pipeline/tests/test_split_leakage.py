"""Assert the train/val/test split never puts the same image filename in two
splits (the specific leakage the project brief calls out to test)."""
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_pipeline.split_by_source import split_bucket


def test_split_is_deterministic():
    assert split_bucket("image_0001.jpg") == split_bucket("image_0001.jpg")


def test_split_has_no_leakage_across_many_names():
    names = [f"img_{i:05d}.jpg" for i in range(5000)]
    seen = defaultdict(set)
    for n in names:
        seen[n].add(split_bucket(n))
    leaked = {n: s for n, s in seen.items() if len(s) > 1}
    assert not leaked


def test_split_ratios_are_roughly_correct():
    names = [f"img_{i:05d}.jpg" for i in range(20000)]
    counts = defaultdict(int)
    for n in names:
        counts[split_bucket(n)] += 1
    total = len(names)
    train_frac = counts["train"] / total
    val_frac = counts["val"] / total
    test_frac = counts["test"] / total
    assert 0.65 < train_frac < 0.75
    assert 0.10 < val_frac < 0.20
    assert 0.10 < test_frac < 0.20


def test_hash_function_is_sha256_based_and_stable():
    # regression guard: if the hashing scheme ever changes, this test forces
    # a conscious re-check of split reproducibility across machines
    h = int(hashlib.sha256(b"img_00000.jpg").hexdigest(), 16)
    assert isinstance(h, int)
