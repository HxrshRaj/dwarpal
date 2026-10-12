import pyarrow as pa
import pyarrow.parquet as pq

from video import analytics


def _write(tmp_path):
    rows = {"frame_idx": [0, 3, 6, 0, 3], "t_sec": [0.0, 0.12, 0.24, 0.0, 0.12],
            "track_id": [1, 1, 1, 2, 2], "label": ["person"] * 3 + ["car"] * 2,
            "score": [0.9] * 5, "x1": [0.0, 3, 6, 10, 10], "y1": [0.0] * 5,
            "x2": [10.0, 13, 16, 20, 20], "y2": [10.0] * 5}
    p = tmp_path / "d.parquet"
    pq.write_table(pa.table(rows), p)
    return str(p)


def test_polars_track_summary_values(tmp_path):
    df = analytics.track_summary_polars(_write(tmp_path))
    r = df.filter(df["track_id"] == 1).row(0, named=True)
    assert r["n_obs"] == 3 and abs(r["path_px"] - 6.0) < 1e-9 and abs(r["dwell_s"] - 0.24) < 1e-9
    assert df.filter(df["track_id"] == 2).row(0, named=True)["path_px"] == 0.0


def test_spark_matches_polars(tmp_path):
    res = analytics.compare(_write(tmp_path))
    assert res["agree"], res
