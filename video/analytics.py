"""Per-track and per-second analytics over the detections Parquet.

The same three aggregations are implemented twice -- Polars (single machine,
lazy) and PySpark (distributed engine, run in local mode here) -- and
`compare()` asserts they agree, so neither implementation is trusted blindly.
"""
from __future__ import annotations

import polars as pl

TRACK_COLS = ["track_id", "label", "n_obs", "first_t", "last_t", "dwell_s", "path_px"]


def track_summary_polars(path: str) -> pl.DataFrame:
    lf = pl.scan_parquet(path).with_columns(
        ((pl.col("x1") + pl.col("x2")) / 2).alias("cx"),
        ((pl.col("y1") + pl.col("y2")) / 2).alias("cy"),
    ).sort(["track_id", "frame_idx"])
    step = (
        (pl.col("cx").diff().over("track_id") ** 2 + pl.col("cy").diff().over("track_id") ** 2).sqrt()
    )
    return (
        lf.with_columns(step.alias("step"))
        .group_by("track_id")
        .agg(
            pl.col("label").first(),
            pl.len().alias("n_obs"),
            pl.col("t_sec").min().alias("first_t"),
            pl.col("t_sec").max().alias("last_t"),
            pl.col("step").sum().alias("path_px"),
        )
        .with_columns((pl.col("last_t") - pl.col("first_t")).alias("dwell_s"))
        .select(TRACK_COLS)
        .sort("track_id")
        .collect()
    )


def per_second_counts_polars(path: str) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .with_columns(pl.col("t_sec").floor().cast(pl.Int64).alias("second"))
        .group_by("second")
        .agg(pl.col("track_id").n_unique().alias("unique_tracks"), pl.len().alias("detections"))
        .sort("second")
        .collect()
    )


def track_summary_spark(spark, path: str):
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    df = spark.read.parquet(path).withColumn("cx", (F.col("x1") + F.col("x2")) / 2).withColumn(
        "cy", (F.col("y1") + F.col("y2")) / 2
    )
    w = Window.partitionBy("track_id").orderBy("frame_idx")
    df = df.withColumn(
        "step",
        F.sqrt((F.col("cx") - F.lag("cx").over(w)) ** 2 + (F.col("cy") - F.lag("cy").over(w)) ** 2),
    )
    return (
        df.groupBy("track_id")
        .agg(
            F.first("label").alias("label"),
            F.count("*").alias("n_obs"),
            F.min("t_sec").alias("first_t"),
            F.max("t_sec").alias("last_t"),
            F.sum("step").alias("path_px"),
        )
        .withColumn("dwell_s", F.col("last_t") - F.col("first_t"))
        .select(*TRACK_COLS)
        .orderBy("track_id")
    )


def per_second_counts_spark(spark, path: str):
    from pyspark.sql import functions as F

    return (
        spark.read.parquet(path)
        .withColumn("second", F.floor("t_sec").cast("long"))
        .groupBy("second")
        .agg(F.countDistinct("track_id").alias("unique_tracks"), F.count("*").alias("detections"))
        .orderBy("second")
    )


def compare(path: str, tol: float = 1e-6) -> dict:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[2]").appName("dwarpal-video").config(
        "spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "4").getOrCreate()
    try:
        pl_t = track_summary_polars(path)
        sp_t = pl.from_pandas(track_summary_spark(spark, path).toPandas())
        pl_s = per_second_counts_polars(path)
        sp_s = pl.from_pandas(per_second_counts_spark(spark, path).toPandas())
        sp_t = sp_t.with_columns(pl.col("n_obs").cast(pl.UInt32))
        sp_s = sp_s.with_columns(pl.col("unique_tracks").cast(pl.UInt32), pl.col("detections").cast(pl.UInt32))
        ok_t = pl_t.select(["track_id", "label", "n_obs"]).equals(sp_t.select(["track_id", "label", "n_obs"]))
        num_diff = max(
            (pl_t[c] - sp_t[c]).abs().max() or 0.0 for c in ("first_t", "last_t", "dwell_s", "path_px")
        )
        ok_s = pl_s.equals(sp_s.select(pl_s.columns))
        return {
            "tracks": pl_t.height, "seconds": pl_s.height,
            "track_ids_labels_counts_equal": bool(ok_t),
            "max_numeric_abs_diff": float(num_diff),
            "per_second_equal": bool(ok_s),
            "agree": bool(ok_t and ok_s and num_diff <= tol),
        }
    finally:
        spark.stop()


if __name__ == "__main__":
    import json
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else "video/results/detections.parquet"
    print(track_summary_polars(p).describe())
    print(json.dumps(compare(p), indent=2))
