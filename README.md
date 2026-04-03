# Point Cloud Similarity Score

A perception-gating framework for autonomous driving that decides, frame by frame,
whether to **reuse** the previous detection result or run a **full 3D object detector**.
The system learns scene stability from LiDAR point clouds using a separate
**Online Bayesian Linear Regression** model per dataset (nuScenes, KITTI, Waymo).

---

## Table of Contents

1. [How it works](#1-how-it-works)
2. [Repository layout](#2-repository-layout)
3. [Prerequisites](#3-prerequisites)
4. [Data preparation](#4-data-preparation)
5. [Step-by-step pipeline](#5-step-by-step-pipeline)
   - 5.1 [nuScenes (keyframes, 2 Hz)](#51-nuscenes-keyframes-2-hz)
   - 5.2 [nuScenes Full (all sweeps, ~20 Hz)](#52-nuscenes-full-all-sweeps-20-hz)
   - 5.3 [KITTI](#53-kitti)
   - 5.4 [Waymo Open Dataset](#54-waymo-open-dataset)
6. [Running the Online Bayesian Engine](#6-running-the-online-bayesian-engine)
7. [Output files](#7-output-files)
8. [Parameter reference](#8-parameter-reference)
9. [One-shot pipeline script](#9-one-shot-pipeline-script)

---

## 1. How it works

```
Raw LiDAR + Labels
      │
      ▼
extract/<dataset>.py        → unified_<dataset>.json
                               Scenes → frames → objects, Chamfer distance, ego velocity
      │
      ▼
extract/generate_csv_<dataset>.py  → data/csv/<dataset>_training_data.csv
                               One row per consecutive frame pair
      │
      ▼
python -m online_model       → output/<dataset>/<dataset>_inference.json
                               3 independent Bayesian models, one per dataset
                               First seed_fraction of scenes  →  cold-start training
                               Remaining scenes               →  inference + online updates
```

### Decision rule

At each streaming frame the engine predicts a confidence score with uncertainty:

```
lower_bound = predicted_mean − uncertainty_weight × predicted_std
if lower_bound > confidence_threshold → REUSE   (skip detector)
else                                  → FULL_DETECTION
```

Every `audit_interval`-th frame is forced to `FULL_DETECTION` regardless of the
prediction, and the resulting ground-truth confidence is fed back as a rank-1
Bayesian posterior update.

### Features used

| Feature | Description |
|---|---|
| `chamfer_dist` | Symmetric Chamfer distance between consecutive downsampled point clouds |
| `ego_vel` | Ego vehicle speed (m/s) |
| `obj_count` | Number of labelled objects in the current frame |
| `avg_dist` | Mean distance from ego to all objects |
| `fastest_obj_vel` | Speed of the fastest object in the previous frame |
| `nearest_obj_dist` | Distance to the nearest object |
| `farthest_obj_dist` | Distance to the farthest object |

---

## 2. Repository layout

```
Point-Cloud-Similarity-Score/
├── extract/
│   ├── extract_nuscenes.py          nuScenes keyframe extraction
│   ├── extract_nuscenes_full.py     nuScenes all-sweep extraction (~20 Hz)
│   ├── extract_kitti.py             KITTI raw drive extraction
│   ├── extract_waymo.py             Waymo Open Dataset extraction (no TF required)
│   ├── generate_csv_nuscenes.py
│   ├── generate_csv_nuscenes_full.py
│   ├── generate_csv_kitti.py
│   └── generate_csv_waymo.py
├── online_model/
│   ├── engine.py                    BayesianLinearRegression + OnlinePerceptionEngine
│   ├── simulation.py                Seed/stream split, inference loop, JSON output
│   ├── types.py                     FrameDecision, PerformanceMetrics dataclasses
│   ├── visualization.py             Learning curve, confidence comparison, histogram
│   └── __main__.py                  CLI entry point
├── confidence_scorer/
│   └── scorer.py                    Ground-truth confidence score calculator
├── data/
│   └── csv/                         Generated CSVs live here
├── output/                          Inference JSON + plots written here
├── models/                          Saved Bayesian posterior states (.npz)
└── run_pipeline.sh                  End-to-end bash script
```

---

## 3. Prerequisites

### Python environment

The project was developed with Python 3.8 (conda `openmmlab` environment).

```bash
conda activate openmmlab
pip install numpy scipy scikit-learn pandas matplotlib pyquaternion nuscenes-devkit
```

For Waymo extraction (no TensorFlow required):

```bash
pip install waymo-open-dataset-tf-2-12-0   # only the protobuf stubs are used
```

### Data roots

Set the correct paths in each extraction script before running:

| Script | Variable | Expected path |
|---|---|---|
| `extract_nuscenes.py` | `DATAROOT` | directory containing `v1.0-mini/` or `v1.0-trainval/` |
| `extract_nuscenes_full.py` | `DATAROOT`, `PKL_DIR` | same nuScenes root + pkl annotation directory |
| `extract_kitti.py` | `DATAROOT` | directory containing `*_sync/` drive folders |
| `extract_waymo.py` | `DATAROOT` | directory containing `raw_data/` and `waymo_processed_data_v0_5_0/` |

---

## 4. Data preparation

Each dataset goes through two stages before training:

1. **Extraction** — reads raw sensor data and annotations, computes Chamfer distances
   and ego velocity, and writes a unified JSON file.
2. **CSV generation** — reads the JSON, computes per-frame features and ground-truth
   confidence scores, and writes a CSV ready for the online model.

The JSON files are intermediate artefacts (large). The CSVs are the direct inputs to
the online model.

---

## 5. Step-by-step pipeline

Run all commands from the **project root** (`Point-Cloud-Similarity-Score/`).

### 5.1 nuScenes (keyframes, 2 Hz)

```bash
# Stage 1 — extract
python extract/extract_nuscenes.py
# Writes: unified_nuscenes_mini.json  (or unified_nuscenes.json for full split)

# Stage 2 — CSV
python extract/generate_csv_nuscenes.py
# Writes: data/csv/nuscenes_training_data.csv
```

Expected output of stage 1 (mini split, 10 scenes):
```
[1/10] Processing scene: scene-0061 ...  → 39 frames extracted
...
Saved 10 scenes to 'unified_nuscenes_mini.json'
```

Expected output of stage 2:
```
Generated 390 rows -> data/csv/nuscenes_training_data.csv
```

### 5.2 nuScenes Full (all sweeps, ~20 Hz)

Requires the interpolated annotation pkl files from `data/csv/intermediate_ann_new/`.

```bash
# Stage 1 — extract
python extract/extract_nuscenes_full.py
# Writes: unified_nuscenes_full.json

# Stage 2 — CSV
python extract/generate_csv_nuscenes_full.py
# Writes: data/csv/nuscenes_full_training_data.csv
```

### 5.3 KITTI

```bash
# Stage 1 — extract
python extract/extract_kitti.py
# Writes: unified_kitti.json

# Stage 2 — CSV
python extract/generate_csv_kitti.py
# Writes: data/csv/kitti_training_data.csv
```

Expected output of stage 1 (one drive sequence):
```
Found 1 possible drive sequence(s).
  → Processing drive: 2011_09_26_drive_0002_sync
    Loaded 5 tracklets.
    - Extracted 77 frames.
Saved 1 scenes to 'unified_kitti.json'
```

### 5.4 Waymo Open Dataset

The extractor handles missing pre-processed npy files automatically — if a segment has
not been pre-processed by M3DETR, point clouds are decoded directly from the `.tfrecord`
using a pure-NumPy range-image decoder (no TensorFlow required).

```bash
# Stage 1 — extract (generates npy files on first run if missing, ~5 min per segment)
python extract/extract_waymo.py
# Writes: unified_waymo.json
#         waymo_processed_data_v0_5_0/<segment>/*.npy  (cached for future runs)

# Stage 2 — CSV
python extract/generate_csv_waymo.py
# Writes: data/csv/waymo_training_data.csv
```

Expected output of stage 1:
```
Found 3 segment(s)
  [1/3] Processing 'segment-146...'
        197 npy files present — no generation needed
    → 197 frames, 6125 total objects
  [2/3] Processing 'segment-157...'
        generating missing point clouds ...
    → 197 frames, 4831 total objects
  [3/3] Processing 'segment-167...'
    → 198 frames, 4072 total objects
Saved 3 scenes to 'unified_waymo.json'
```

---

## 6. Running the Online Bayesian Engine

### Single dataset

```bash
python -m online_model \
    --dataset waymo \
    --seed-fraction 0.2 \
    --confidence-threshold 0.85 \
    --audit-interval 5 \
    --output-dir output \
    --model-state-dir models
```

This uses the **first 20% of scenes** (aligned to scene boundaries) to cold-start the
Bayesian model, then streams the remaining 80% through the inference + update loop.

### All three datasets at once

```bash
python -m online_model \
    --dataset all \
    --seed-fraction 0.2 \
    --output-dir output \
    --model-state-dir models
```

Three independent models are trained and three separate JSON logs are written.

### Reload a saved model and continue on new data

```bash
# The .npz state contains the full Bayesian posterior + frozen scaler.
# Load it in Python:
from online_model.engine import OnlinePerceptionEngine
engine = OnlinePerceptionEngine()
engine.load_state("models/waymo.npz")
# then call engine.process_frame(...) for each new frame
```

---

## 7. Output files

After running the online model the following are written under `output/<dataset>/`:

### `<dataset>_inference.json`

One file per dataset. Top-level structure:

```json
{
  "dataset": "waymo",
  "seed_fraction": 0.2,
  "seed_scenes": [
    "segment-1464917900451858484_1960_000_1980_000_with_camera_labels"
  ],
  "streaming_scenes": [
    "segment-15724298772299989727_5386_410_5406_410_with_camera_labels",
    "segment-16751706457322889693_4475_240_4495_240_with_camera_labels"
  ],
  "metrics": {
    "total_frames": 393,
    "reuse_count": 42,
    "full_detection_count": 351,
    "reuse_rate_pct": 10.69,
    "safety_violations": 0,
    "avg_prediction_error": 0.034,
    "cumulative_latency_saved_ms": 3339.0
  },
  "scenes": {
    "segment-15724...": [
      {
        "frame_idx": 0,
        "predicted_confidence": 0.312,
        "predicted_std": 0.048,
        "actual_confidence": 0.301,
        "decision": "FULL_DETECTION",
        "prediction_error": 0.011,
        "is_audit_frame": true
      },
      ...
    ],
    "segment-16751...": [ ... ]
  }
}
```

**Key fields per frame:**

| Field | Description |
|---|---|
| `frame_idx` | Local index within the scene (resets to 0 at each scene boundary) |
| `predicted_confidence` | Model's point estimate (clipped to [0, 1]) |
| `predicted_std` | Predictive standard deviation — quantifies epistemic + aleatoric uncertainty |
| `actual_confidence` | Ground-truth confidence from the scorer |
| `decision` | `"REUSE"` or `"FULL_DETECTION"` |
| `prediction_error` | `|actual − predicted|` — only non-zero on audit frames |
| `is_audit_frame` | `true` if this frame triggered a posterior update |
| `timestamp_us` | Source timestamp in microseconds (present when available) |

**Seed scenes** are excluded from `scenes` — they were used for training only, not inference.

### Plots

| File | Description |
|---|---|
| `learning_curve.png` | Prediction error over audit frames — shows model adaptation |
| `confidence_comparison.png` | Predicted vs actual confidence over time |
| `decision_histogram.png` | Distribution of REUSE / FULL_DETECTION decisions |

### Model state

`models/<dataset>.npz` contains the full Bayesian posterior (`m`, `S`), frozen scaler
statistics, and engine hyperparameters. Load it with `engine.load_state(path)`.

---

## 8. Parameter reference

### Online model (`python -m online_model`)

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `all` | `nuscenes`, `nuscenes_full`, `kitti`, `waymo`, or `all` |
| `--csv-dir` | `data/csv` | Directory with `*_training_data.csv` files |
| `--output-dir` | `output` | Root for JSON logs and plots |
| `--model-state-dir` | _(none)_ | Save Bayesian posterior here as `{dataset}.npz` |
| `--seed-fraction` | `0.15` | Fraction of dataset used for cold-start training |
| `--split-mode` | `frame` | `frame` (exact frame-count split) or `scene` (whole-scene split) |
| `--confidence-threshold` | `0.85` | REUSE threshold on the lower confidence bound |
| `--audit-interval` | `5` | Force full detection + update every N frames |
| `--alpha` | `1.0` | Prior precision — higher = stronger weight regularisation |
| `--beta` | `25.0` | Noise precision (1 / σ²) — tune to match confidence score variance |
| `--uncertainty-weight` | `1.0` | `k` in `mean − k·std > threshold` — higher = more conservative |

### Tuning guidance

**`--seed-fraction`**  
Controls the train/infer split. `0.15` means ~15% warm-up frames and ~85% streamed
frames when using the default `--split-mode frame`. Use a larger value if the model has high prediction error at the start
of streaming (visible in `learning_curve.png`).

**`--confidence-threshold`**  
Must match the typical range of your confidence scores. If scores for a dataset
cluster around 0.3 (e.g. Waymo with distant objects), lower the threshold to
`0.35`–`0.40` accordingly.

**`--beta`**  
Reflects assumed noise in the confidence scores. `beta=25` means σ ≈ 0.2, which
is reasonable for confidence scores in [0, 1]. Increase beta if your scores are
tighter / more consistent; decrease if they are noisy.

**`--audit-interval`**  
Lower values (e.g. `3`) give more frequent updates and faster adaptation but
reduce latency savings. Higher values (e.g. `10`) maximise skip rate but slow
adaptation to scene changes.

---

## 9. One-shot pipeline script

`run_pipeline.sh` automates extraction → CSV generation → online training → inference.

```bash
# Full pipeline for all datasets
./run_pipeline.sh --datasets all --output-dir output --model-dir models

# Online model only (extraction + CSV already done)
./run_pipeline.sh --datasets waymo --skip-extract --skip-csv \
    --seed-fraction 0.2 --threshold 0.85 --audit-interval 5

# Single dataset, explicit seed fraction
./run_pipeline.sh --datasets nuscenes --seed-fraction 0.3

# Different thresholds per dataset in one run
./run_pipeline.sh --datasets all --threshold 0.85 \
    --threshold-map "nuscenes:0.90,nuscenes_full:0.80,kitti:0.70,waymo:0.40"
```

**Common flags:**

| Flag | Default | Description |
|---|---|---|
| `--datasets` | `nuscenes,kitti` | Comma-separated: `nuscenes,nuscenes_full,kitti,waymo` or `all` |
| `--skip-extract` | off | Skip stage 1 (raw → JSON) |
| `--skip-csv` | off | Skip stage 2 (JSON → CSV) |
| `--online-only` | off | Skip extraction + CSV stages and run the online Bayesian engine only |
| `--seed-fraction` | `0.15` | Passed directly to `--seed-fraction` of the online model |
| `--split-mode` | `frame` | Passed directly to `--split-mode` of the online model |
| `--threshold` | `0.85` | Default confidence threshold for REUSE decisions |
| `--threshold-map` | _(none)_ | Per-dataset overrides in `dataset:value,...` format (e.g. `kitti:0.70,waymo:0.40`) |
| `--audit-interval` | `5` | Frames between posterior updates |
| `--output-dir` | `output` | Root output directory |
| `--model-dir` | `models` | Directory for saved `.npz` posterior states |
