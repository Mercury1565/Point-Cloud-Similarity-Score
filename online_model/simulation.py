import json
import os
import glob
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from online_model.engine import OnlinePerceptionEngine
from online_model.types import PerformanceMetrics
from online_model.visualization import (
    generate_learning_curve,
    generate_confidence_comparison,
    generate_decision_histogram,
)

FEATURE_COLUMNS = [
    "chamfer_dist",
    "ego_vel",
    "obj_count",
    "avg_dist",
    "fastest_obj_vel",
    "nearest_obj_dist",
    "farthest_obj_dist",
]


def run_simulation(
    csv_dir: str,
    visuals_output_dir: str,
    dataset: Optional[str] = None,
    confidence_threshold: float = 0.85,
    audit_interval: int = 5,
    seed_fraction: float = 0.15,
    reuse_latency_ms: float = 0.5,
    full_detection_latency_ms: float = 80.0,
    safety_threshold: float = 0.70,
    alpha: float = 1.0,
    beta: float = 25.0,
    uncertainty_weight: float = 1.0,
    model_state_dir: Optional[str] = None,
    split_mode: str = "frame",
) -> PerformanceMetrics:
    """
    Run the online learning simulation on CSV data.

    Seed / stream split
    -------------------
    ``seed_fraction`` (0 < f < 1) controls what fraction of the dataset is
    used to cold-start the Bayesian model.
    ``split_mode='frame'`` uses an exact frame-count split.
    ``split_mode='scene'`` keeps whole-scene boundaries.
    The remaining rows are streamed through the predict → decide → update loop
    and logged to the output JSON.

    Output
    ------
    One JSON file per dataset is written to ``<visuals_output_dir>/<label>/``:

        {
          "dataset":          "waymo",
          "seed_fraction":    0.2,
          "seed_scenes":      ["scene_A"],
          "streaming_scenes": ["scene_B", "scene_C"],
          "metrics": { ... aggregate numbers ... },
          "scenes": {
            "scene_B": [
              { "frame_idx": 0, "predicted_confidence": 0.82, ... },
              ...
            ],
            "scene_C": [ ... ]
          }
        }

    Temporal ordering
    -----------------
    Frames are processed scene-by-scene in the order they appear in the CSV.
    Within each scene the local frame index resets to 0, so audit frames
    (local_idx % audit_interval == 0) never straddle a scene boundary.
    """

    # ── 1. Gather CSV files — sorted for deterministic load order ─────────────
    if dataset:
        exact = os.path.join(csv_dir, f"{dataset}_training_data.csv")
        if os.path.exists(exact):
            csv_files = [exact]
        else:
            csv_files = sorted(glob.glob(os.path.join(csv_dir, f"*{dataset}*.csv")))
    else:
        csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching '{dataset}' found in: {csv_dir}" if dataset
            else f"No CSV files found in directory: {csv_dir}"
        )

    label = dataset if dataset else "combined"
    print(f"[{label}] Loading {len(csv_files)} CSV file(s)...")

    data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # ── 2. Validate ───────────────────────────────────────────────────────────
    missing_cols = [c for c in FEATURE_COLUMNS + ["target_confidence"] if c not in data.columns]
    if missing_cols:
        raise ValueError(f"[{label}] Missing columns in dataset: {missing_cols}")

    if not (0.0 < seed_fraction < 1.0):
        raise ValueError(f"seed_fraction must be in (0, 1), got {seed_fraction}")

    # ── 3. Build engine ───────────────────────────────────────────────────────
    engine = OnlinePerceptionEngine(
        confidence_threshold=confidence_threshold,
        audit_interval=audit_interval,
        seed_batch_size=1,          # overridden inside _run_scene_aware
        reuse_latency_ms=reuse_latency_ms,
        full_detection_latency_ms=full_detection_latency_ms,
        safety_threshold=safety_threshold,
        alpha=alpha,
        beta=beta,
        uncertainty_weight=uncertainty_weight,
    )

    if split_mode not in {"scene", "frame"}:
        raise ValueError(f"split_mode must be 'scene' or 'frame', got {split_mode}")

    # ── 4. Split strategy ──────────────────────────────────────────────────────
    if "scene_id" in data.columns and split_mode == "scene":
        seed_scenes, streaming_scenes = _run_scene_aware(engine, data, seed_fraction, label)
    elif "scene_id" in data.columns and split_mode == "frame":
        seed_scenes, streaming_scenes = _run_frame_aware(engine, data, seed_fraction, label)
    else:
        print(
            f"[{label}] WARNING: no 'scene_id' column — falling back to flat streaming. "
            "Re-generate CSVs with the updated extract scripts to fix this."
        )
        seed_scenes, streaming_scenes = _run_flat(engine, data, seed_fraction, label)

    # ── 5. Optionally persist model state ─────────────────────────────────────
    if model_state_dir:
        os.makedirs(model_state_dir, exist_ok=True)
        state_path = os.path.join(model_state_dir, label)
        engine.save_state(state_path)
        print(f"[{label}] Model state saved to {state_path}.npz")

    # ── 6. Aggregate metrics ──────────────────────────────────────────────────
    metrics = engine.calculate_metrics()

    # ── 7. Build output directory ─────────────────────────────────────────────
    out_dir = os.path.join(visuals_output_dir, label)
    os.makedirs(out_dir, exist_ok=True)

    # ── 8. Visualisations ─────────────────────────────────────────────────────
    generate_learning_curve(engine, os.path.join(out_dir, "learning_curve.png"))
    generate_confidence_comparison(engine, os.path.join(out_dir, "confidence_comparison.png"))
    generate_decision_histogram(engine, os.path.join(out_dir, "decision_histogram.png"))
    print(f"[{label}] Plots saved to {out_dir}/")

    # ── 9. Build per-scene inference JSON ─────────────────────────────────────
    scenes_dict = {}
    for d in engine.decisions:
        sid = d.scene_id if d.scene_id else "unknown"
        if sid not in scenes_dict:
            scenes_dict[sid] = []
        entry = {
            "frame_idx":             d.frame_idx,
            "predicted_confidence":  round(d.predicted_confidence, 6),
            "predicted_std":         round(d.predicted_std, 6),
            "actual_confidence":     round(d.actual_confidence, 6),
            "decision":              d.decision,
            "prediction_error":      round(d.prediction_error, 6),
            "is_audit_frame":        d.is_audit_frame,
        }
        if d.timestamp_us:
            entry["timestamp_us"] = d.timestamp_us
        scenes_dict[sid].append(entry)

    avg_mae = (
        sum(metrics.mean_absolute_errors) / len(metrics.mean_absolute_errors)
        if metrics.mean_absolute_errors else 0.0
    )
    total = metrics.total_frames
    output_json = {
        "dataset":          label,
        "seed_fraction":    seed_fraction,
        "seed_scenes":      seed_scenes,
        "streaming_scenes": streaming_scenes,
        "metrics": {
            "total_frames":               total,
            "reuse_count":                metrics.reuse_count,
            "full_detection_count":       metrics.full_detection_count,
            "reuse_rate_pct":             round(metrics.reuse_count / total * 100, 2) if total else 0.0,
            "safety_violations":          metrics.safety_violations,
            "avg_prediction_error":       round(avg_mae, 6),
            "cumulative_latency_saved_ms": round(metrics.cumulative_latency_saved_ms, 2),
        },
        "scenes": scenes_dict,
    }

    inference_path = os.path.join(out_dir, f"{label}_inference.json")
    with open(inference_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"[{label}] Inference log saved to {inference_path}")

    return metrics


# ── Scene-aware helpers ────────────────────────────────────────────────────────

def _run_scene_aware(
    engine: OnlinePerceptionEngine,
    data: pd.DataFrame,
    seed_fraction: float,
    label: str,
) -> Tuple[List[str], List[str]]:
    """
    Split by scene, train on seed (first seed_fraction of rows, aligned to
    scene boundaries), stream the rest.

    Returns
    -------
    seed_scenes      : list of scene IDs used for cold-start
    streaming_scenes : list of scene IDs streamed through the engine
    """
    scene_ids = list(dict.fromkeys(data["scene_id"].tolist()))
    total_rows = len(data)
    seed_target = int(total_rows * seed_fraction)

    # Accumulate complete scenes until we reach the seed target
    seed_rows = pd.DataFrame()
    seed_scene_ids: List[str] = []
    for sid in scene_ids:
        scene_data = data[data["scene_id"] == sid]
        seed_rows = pd.concat([seed_rows, scene_data], ignore_index=True)
        seed_scene_ids.append(sid)
        if len(seed_rows) >= seed_target:
            break

    if len(seed_rows) == 0:
        raise ValueError(f"[{label}] Dataset is empty after scene grouping.")

    streaming_scene_ids = [s for s in scene_ids if s not in set(seed_scene_ids)]

    X_seed = seed_rows[FEATURE_COLUMNS].values
    y_seed = seed_rows["target_confidence"].values
    engine.initialize_model(X_seed, y_seed)

    total_stream = sum(len(data[data["scene_id"] == s]) for s in streaming_scene_ids)
    print(f"[{label}] Seed    : {len(seed_scene_ids)} scene(s), {len(seed_rows)} frames "
          f"({len(seed_rows)/total_rows*100:.1f}% of dataset)")
    print(f"[{label}] Stream  : {len(streaming_scene_ids)} scene(s), {total_stream} frames")

    for sid in streaming_scene_ids:
        scene_rows = data[data["scene_id"] == sid]
        X_scene = scene_rows[FEATURE_COLUMNS].values
        y_scene = scene_rows["target_confidence"].values
        for local_idx in range(len(X_scene)):
            engine.process_frame(
                local_idx,
                X_scene[local_idx],
                float(y_scene[local_idx]),
                scene_id=str(sid),
            )

    return seed_scene_ids, streaming_scene_ids


def _run_flat(
    engine: OnlinePerceptionEngine,
    data: pd.DataFrame,
    seed_fraction: float,
    label: str,
) -> Tuple[List[str], List[str]]:
    """Legacy flat-streaming fallback for CSVs without a scene_id column."""
    X = data[FEATURE_COLUMNS].values
    y = data["target_confidence"].values
    seed_batch_size = max(1, int(len(X) * seed_fraction))

    if len(X) <= seed_batch_size:
        raise ValueError(
            f"[{label}] Dataset has {len(X)} rows but seed needs {seed_batch_size}. "
            "Lower --seed-fraction or add more data."
        )

    engine.initialize_model(X[:seed_batch_size], y[:seed_batch_size])
    print(f"[{label}] Seed    : {seed_batch_size} frames ({seed_fraction*100:.1f}%)")
    print(f"[{label}] Stream  : {len(X) - seed_batch_size} frames")

    for frame_idx in range(seed_batch_size, len(X)):
        engine.process_frame(frame_idx, X[frame_idx], float(y[frame_idx]))

    return ["flat_seed"], ["flat_stream"]


def _run_frame_aware(
    engine: OnlinePerceptionEngine,
    data: pd.DataFrame,
    seed_fraction: float,
    label: str,
) -> Tuple[List[str], List[str]]:
    """
    Frame-count split: use exactly seed_fraction of rows for warm-up, then stream
    the remaining rows. Scene IDs are still logged per-frame.
    """
    X = data[FEATURE_COLUMNS].values
    y = data["target_confidence"].values
    total_rows = len(data)
    seed_batch_size = max(1, int(total_rows * seed_fraction))

    if total_rows <= seed_batch_size:
        raise ValueError(
            f"[{label}] Dataset has {total_rows} rows but seed needs {seed_batch_size}. "
            "Lower --seed-fraction or add more data."
        )

    engine.initialize_model(X[:seed_batch_size], y[:seed_batch_size])

    # Scene-wise local frame index across the full dataset order.
    local_idx_all = data.groupby("scene_id").cumcount().to_numpy()
    scene_col = data["scene_id"].astype(str).tolist()

    seed_scenes = list(dict.fromkeys(scene_col[:seed_batch_size]))
    streaming_scenes = list(dict.fromkeys(scene_col[seed_batch_size:]))

    print(f"[{label}] Seed    : {seed_batch_size} frames ({seed_batch_size/total_rows*100:.1f}% of dataset)")
    print(f"[{label}] Stream  : {total_rows - seed_batch_size} frames ({(total_rows-seed_batch_size)/total_rows*100:.1f}%)")
    print(
        f"[{label}] Split mode: frame "
        f"(seed scenes touched: {len(seed_scenes)}, stream scenes touched: {len(streaming_scenes)})"
    )

    for i in range(seed_batch_size, total_rows):
        engine.process_frame(
            int(local_idx_all[i]),
            X[i],
            float(y[i]),
            scene_id=scene_col[i],
        )

    return seed_scenes, streaming_scenes
