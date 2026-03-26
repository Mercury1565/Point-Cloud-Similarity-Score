
# PROJECT README

This project builds a lightweight predictor to decide if we should reuse the previous frame’s perception output for autonomous-driving sequences.

At a high level, we:

1. **Extract** temporally ordered scenes from nuScenes mini into a unified JSON format.
2. **Define a target confidence score** that measures how similar frame $t$ is to frame $t-1$ (based on object ID continuity + 3D bounding-box stability).
3. **Generate tabular features** for each frame transition $(t-1 \rightarrow t)$ and compute that target confidence.
4. **Train a Random Forest regressor** to predict confidence from cheap-to-compute signals (ego motion + scene stats + point cloud change proxy).
5. **Simulate a runtime policy**: if predicted confidence is high, **reuse** previous detections; otherwise run a **heavy detector**.

The end result is a small model (`confidence_rf_model.pkl`) that can drive a *compute-saving reuse policy* while tracking a simple safety proxy.

## Why this matters

Modern perception stacks often spend significant compute running big detectors every frame. But many frames are redundant (straight driving, slow ego motion, stable objects). This repo explores a pragmatic approach:

- learn a **scalar confidence** in $[0, 1]$ that correlates with “how safe is it to reuse?”
- use it to make a **skip/reuse decision** that reduces average latency

## Repository layout

- `data/` — raw dataset assets (nuScenes mini layout under `data/v1.0-mini/`, plus sensor blobs under `data/samples/`, etc.)
- `extract/` — scripts and docs for building intermediate datasets (unified JSON, CSV training table)
- `confidence_scorer/` — the core scoring library that computes the frame-to-frame “confidence score” target
- `model/` — training and simulation scripts for the lightweight regressor + reuse policy

### Key artifacts in the repo root

- `unified_nuscenes_mini.json` — extracted scene/frame/object representation (Stage 1 output)
- `training_data.csv` — tabular ML dataset with features + `target_confidence`
- `confidence_rf_model.pkl` — trained Random Forest model
- `feature_importance.png`, `predicted_vs_actual.png`, `nuscenes_similarity_trends.png` — analysis plots

## End-to-end pipeline

### 1) Extract a unified scene representation (`extract/`)

`extract/extract_nuscenes.py` reads nuScenes mini from `data/` and writes `unified_nuscenes_mini.json`.

For each scene, it produces an ordered `frame_list`. Each frame includes:

- `chamfer_distance`: a symmetric Chamfer distance between downsampled LiDAR point clouds at $t$ and $t-1$ (a proxy for raw geometric change)
- `ego_vel`: estimated ego velocity between frames
- `object_list`: 3D boxes with stable `obj_id` across time

Schema details: see [UNIFIED_JSON.md](extract/docs/UNIFIED_JSON.md)

### 2) Define the target “confidence score” (`confidence_scorer/`)

The training target is **not** hand-labeled. It’s computed deterministically from two consecutive frames:

1. **Identity continuity**: do object IDs persist? (F1 score)
2. **Geometric stability**: do matched objects’ 3D boxes overlap well? (mean 3D IoU / mIoU)
3. **Composite score**: combine F1 and mIoU into a single scalar (harmonic mean by default)

This score is intended to behave like:

- high ($\approx 1$): you can probably reuse the last inference safely
- low ($\approx 0$): the scene changed too much; rerun heavy inference

Math and implementation notes: see [CONFIDENCE_SCORER.md](confidence_scorer/docs/CONFIDENCE_SCORER.md).

### 3) Build a tabular dataset (`extract/`)

`extract/generate_csv.py` converts `unified_nuscenes_mini.json` into `training_data.csv`, where each row corresponds to a transition $(t-1 \rightarrow t)$.

Features include:

- `chamfer_dist`, `ego_vel`
- temporal dynamics: `delta_ego_vel`, `ego_accel`
- scene complexity: `obj_count`, `delta_obj_count`
- spatial dynamics: `avg_dist`

Target:

- `target_confidence` computed by `ConfidenceScorer` between frames

Feature and target definitions: see [FEATURES_AND_TARGET.md](extract/docs/FEATURES_AND_TARGET.md).

### 4) Train a lightweight regressor (`model/`)

`model/train_confidence_model.py` trains a `RandomForestRegressor` to predict `target_confidence`.

Typical outputs include:

- `confidence_rf_model.pkl`
- `predicted_vs_actual.png`
- `feature_importance.png`

The goal is a model that’s cheap enough to run every frame and still informative enough to drive a reuse decision.

### 5) Simulate a reuse policy (`model/`)

`model/simulate.py` loads the trained model + `training_data.csv` and simulates a runtime loop:

- always pay a tiny “RF latency” to predict confidence
- if predicted confidence exceeds a threshold, **REUSE** (skip heavy detection)
- otherwise, **DETECT** (pay heavy detector cost)

It reports basic ROI metrics like skip rate and total synthetic latency, plus a simple safety proxy: counts of cases where the model says “reuse” but ground-truth confidence is low.

## Other Project Docs

- [Confidence scoring internals](confidence_scorer/docs/README.md)
- [Unified JSON schema](extract/docs/UNIFIED_JSON.md)
- [Feature + target definitions](extract/docs/FEATURES_AND_TARGET.md)
- [Extraction audit + plots interpretation](extract/docs/DATA_AUDIT.md)

## Quickstart

This repo is plain Python scripts (no CLI wrapper). The typical flow is:

1. Ensure nuScenes mini data is present under `data/`.
2. Run extraction to create `unified_nuscenes_mini.json`.
3. Generate `training_data.csv`.
4. Train the model.
5. Run the skip/reuse simulation.

If you’d like, I can add a small `python -m` entrypoint or a `Makefile`/`tasks.json` to make the pipeline one-command reproducible.

## Notes & assumptions

- nuScenes samples are treated as ~2Hz ($dt \approx 0.5s$) when computing acceleration in `extract/generate_csv.py`.
- `chamfer_distance` is computed on a **downsampled** point set for speed.
- The confidence target is dataset-derived and depends on consistent `obj_id` assignment; it’s most appropriate for datasets with stable instance identities.
