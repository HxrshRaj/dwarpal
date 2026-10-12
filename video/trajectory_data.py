"""Build (last K steps -> next step) trajectory windows from the detections Parquet."""
from __future__ import annotations

import numpy as np
import polars as pl

K = 4  # history length (sampled frames)


def make_windows(path: str, k: int = K, min_len: int = 8, only_label: str | None = "person"):
    """Returns (X, y, track_ids). X: (n, k*2) past centroid displacements; y: (n, 2) next displacement.
    Displacements are in pixels per sampled step, and windows keep their track id so
    splits can be made by track (never by window) to avoid leakage."""
    df = pl.read_parquet(path)
    if only_label:
        df = df.filter(pl.col("label") == only_label)
    df = df.with_columns(((pl.col("x1") + pl.col("x2")) / 2).alias("cx"), ((pl.col("y1") + pl.col("y2")) / 2).alias("cy"))
    X, y, tids = [], [], []
    for tid, g in df.sort(["track_id", "frame_idx"]).group_by("track_id", maintain_order=True):
        if g.height < min_len:
            continue
        c = np.stack([g["cx"].to_numpy(), g["cy"].to_numpy()], 1)
        d = np.diff(c, axis=0)
        for i in range(len(d) - k):
            X.append(d[i : i + k].reshape(-1))
            y.append(d[i + k])
            tids.append(tid[0] if isinstance(tid, tuple) else tid)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(tids)


def split_by_track(X, y, tids, test_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(tids)
    rng.shuffle(uniq)
    test_ids = set(uniq[: max(1, int(len(uniq) * test_frac))].tolist())
    m = np.array([t in test_ids for t in tids])
    return (X[~m], y[~m]), (X[m], y[m])
