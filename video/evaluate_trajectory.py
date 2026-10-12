"""Train JAX and TF forecasters; compare against a constant-velocity baseline on held-out TRACKS."""
from __future__ import annotations

import json
import sys

import numpy as np

from video import trajectory_data as td


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def main(path="video/results/detections.parquet", seeds=(0, 1, 2, 3, 4)):
    from video import trajectory_jax, trajectory_tf

    X, y, tids = td.make_windows(path)
    res = {"windows": int(len(X)), "tracks": int(len(np.unique(tids))), "seeds": list(seeds), "rmse_px": {}}
    rows = {"zero_motion": [], "constant_velocity": [], "jax_mlp": [], "tf_mlp": []}
    for s in seeds:
        (Xtr, ytr), (Xte, yte) = td.split_by_track(X, y, tids, seed=s)
        last = Xte[:, -2:]  # last observed displacement
        rows["zero_motion"].append(rmse(np.zeros_like(yte), yte))
        rows["constant_velocity"].append(rmse(last, yte))
        rows["jax_mlp"].append(rmse(trajectory_jax.train(Xtr, ytr, seed=s)(Xte), yte))
        rows["tf_mlp"].append(rmse(trajectory_tf.train(Xtr, ytr, seed=s)(Xte), yte))
    res["test_windows_last_split"] = int(len(Xte))
    for k, v in rows.items():
        res["rmse_px"][k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:2])
