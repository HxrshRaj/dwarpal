# video/ — video analysis, columnar analytics, and JAX/TensorFlow forecasting

Adds a real video-analysis path to Dwarpal. Everything below was run end to end; the numbers come from
`video/results/`.

```
bash video/models/download.sh                         # detector weights + sample video (not committed)
python -m video.detect_track video/vtest.avi          # detect + track -> video/results/detections.parquet
python -m video.analytics                             # Polars vs PySpark aggregations, asserted equal
python -m video.evaluate_trajectory                   # JAX vs TensorFlow vs baselines
pytest video/tests
```

## What it does

1. **Detect + track** (`detect_track.py`, `tracker.py`): samples every 2nd frame, runs MobileNet-SSD (OpenCV DNN,
   Caffe, PASCAL-VOC classes; ~51 ms mean / 72 ms p95 per frame on 2 CPU cores), and links boxes into tracks with a
   greedy IoU tracker written from scratch. Output is an Apache Arrow/Parquet table.
2. **Columnar analytics** (`analytics.py`): per-track dwell time and path length, plus per-second counts, written in
   **Polars** (lazy) and **PySpark** (local mode, window functions). `compare()` asserts the two agree:
   315 tracks, 80 seconds, identical counts and max numeric difference 0.0.
3. **Trajectory forecasting** (`trajectory_*.py`, `evaluate_trajectory.py`): the same small MLP (4 past steps → next
   step, Huber loss, Adam) in **JAX** (hand-written Adam, jit/grad) and **TensorFlow/Keras**, evaluated on held-out
   *tracks* (never held-out windows, to avoid leakage) over 5 seeds against zero-motion and constant-velocity baselines.

## Results (OpenCV `vtest.avi`: 795 frames, 10 fps, 768x576, pedestrians)

| Method | Test RMSE (px/step), mean ± std over 5 seeds |
|---|---|
| zero motion | 5.52 ± 1.37 |
| constant velocity | **5.30 ± 1.01** |
| JAX MLP | 5.51 ± 0.85 |
| TensorFlow MLP | 5.83 ± 1.56 |

**Neither learned model beats the constant-velocity baseline.** With only 299 windows from 31 tracks and jittery
SSD boxes, there is not enough signal for an MLP to learn more than "keep moving the way you were". This is a
negative result and is reported as one; more data, box smoothing (e.g. a Kalman filter) or a sequence model would be
the next steps.

## Limits

- Detector is MobileNet-SSD at 300x300: it misses small/occluded people (398 frames, 1,207 detections, 315 tracks,
  many short fragments from missed detections). Track fragmentation is the main error source.
- The torchvision COCO weights used elsewhere in this repo could not be downloaded in the environment where this was
  built, so a different pretrained detector is used here.
- `opencv-python` must be 4.x (the Caffe importer was removed in 5.x).
- The video is a public sample clip, not gate footage; nothing here claims gate-camera accuracy.
- Sample video: OpenCV (Apache-2.0). Weights: chuanqi305/MobileNet-SSD (MIT).
