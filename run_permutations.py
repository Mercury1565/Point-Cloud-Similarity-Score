"""
Run the online SGD simulation across a set of SGDRegressor parameter permutations,
separately for each dataset (kitti / nuscenes), saving visuals and a params file
into each run's own subdirectory.

Output layout:
    output/
        kitti/
            baseline/        <- params.txt + plots
            high_lr/
            ...
        nuscenes/
            baseline/
            ...
        report.txt
"""

import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from online_model.types import FrameDecision, PerformanceMetrics
from online_model.visualization import (
    generate_learning_curve,
    generate_confidence_comparison,
    generate_decision_histogram,
)

# ── Configuration ─────────────────────────────────────────────────────────────

CSV_DIR = "data/csv"
BASE_OUTPUT = "output"
SEED_BATCH_SIZE = 50
CONFIDENCE_THRESHOLD = 0.85
AUDIT_INTERVAL = 5
SAFETY_THRESHOLD = 0.70
REUSE_LATENCY_MS = 0.5
FULL_DETECTION_LATENCY_MS = 80.0

DATASETS = {
    "kitti":    "kitti_training_data.csv",
    "nuscenes": "nuscenes_training_data.csv",
}

FEATURE_COLUMNS = [
    "chamfer_dist",
    "ego_vel",
    "obj_count",
    "avg_dist",
    "fastest_obj_vel",
    "nearest_obj_dist",
    "farthest_obj_dist",
]

# Each entry: (label, SGDRegressor kwargs)
PERMUTATIONS = [
    ("baseline",           dict(learning_rate='constant', eta0=0.01,  penalty='l2', alpha=0.001, average=False)),
    ("high_lr",            dict(learning_rate='constant', eta0=0.1,   penalty='l2', alpha=0.001, average=False)),
    ("low_lr",             dict(learning_rate='constant', eta0=0.001, penalty='l2', alpha=0.001, average=False)),
    ("l1_penalty",         dict(learning_rate='constant', eta0=0.01,  penalty='l1', alpha=0.001, average=False)),
    ("high_alpha",         dict(learning_rate='constant', eta0=0.01,  penalty='l2', alpha=0.01,  average=False)),
    ("adaptive_lr",        dict(learning_rate='adaptive', eta0=0.01,  penalty='l2', alpha=0.001, average=False)),
    ("averaged",           dict(learning_rate='constant', eta0=0.01,  penalty='l2', alpha=0.001, average=True)),
    ("no_penalty_high_lr", dict(learning_rate='constant', eta0=0.05,  penalty=None, alpha=0.0,   average=False)),
]


# ── Minimal engine shell (duck-typed for the visualisation functions) ─────────

class _SGDEngine:
    """Holds simulation state in the shape the visualisation helpers expect."""

    def __init__(self, confidence_threshold: float, safety_threshold: float):
        self.confidence_threshold = confidence_threshold
        self.safety_threshold = safety_threshold
        self.decisions: List[FrameDecision] = []


# ── Core SGD simulation ───────────────────────────────────────────────────────

def _run_sgd_simulation(
    X: np.ndarray,
    y: np.ndarray,
    sgd_params: dict,
) -> _SGDEngine:
    """Seed-warm-up + streaming loop using SGDRegressor."""

    engine = _SGDEngine(CONFIDENCE_THRESHOLD, SAFETY_THRESHOLD)

    scaler = StandardScaler()
    scaler.fit(X)                               # fit on full dataset for stable scaling
    X_scaled = scaler.transform(X)

    model = SGDRegressor(**sgd_params)
    model.partial_fit(X_scaled[:SEED_BATCH_SIZE], y[:SEED_BATCH_SIZE])

    for frame_idx in range(SEED_BATCH_SIZE, len(X)):
        x_scaled = X_scaled[frame_idx].reshape(1, -1)
        y_actual  = float(y[frame_idx])

        predicted = float(np.clip(model.predict(x_scaled)[0], 0.0, 1.0))
        decision  = "REUSE" if predicted > CONFIDENCE_THRESHOLD else "FULL_DETECTION"

        is_audit = (frame_idx % AUDIT_INTERVAL == 0)
        if is_audit:
            error = abs(y_actual - predicted)
            model.partial_fit(x_scaled, [y_actual])
        else:
            error = 0.0

        engine.decisions.append(FrameDecision(
            frame_idx=frame_idx,
            predicted_confidence=predicted,
            actual_confidence=y_actual,
            decision=decision,
            prediction_error=error,
            is_audit_frame=is_audit,
        ))

    return engine


def _calculate_metrics(engine: _SGDEngine) -> PerformanceMetrics:
    reuse_count = full_count = safety_violations = 0
    mae_list: List[float] = []
    audit_frames: List[int] = []

    for d in engine.decisions:
        if d.decision == "REUSE":
            reuse_count += 1
            if d.actual_confidence < SAFETY_THRESHOLD:
                safety_violations += 1
        else:
            full_count += 1
        if d.is_audit_frame:
            mae_list.append(d.prediction_error)
            audit_frames.append(d.frame_idx)

    latency_saved = reuse_count * (FULL_DETECTION_LATENCY_MS - REUSE_LATENCY_MS)

    return PerformanceMetrics(
        total_frames=len(engine.decisions),
        reuse_count=reuse_count,
        full_detection_count=full_count,
        cumulative_latency_saved_ms=latency_saved,
        mean_absolute_errors=mae_list,
        safety_violations=safety_violations,
        audit_frames=audit_frames,
    )


# ── File writers ──────────────────────────────────────────────────────────────

def _write_params_file(out_dir: str, dataset: str, label: str, sgd_params: dict) -> None:
    with open(os.path.join(out_dir, "params.txt"), "w") as f:
        f.write(f"Run     : {label}\n")
        f.write(f"Dataset : {dataset}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\nSGDRegressor Parameters\n")
        f.write("-" * 30 + "\n")
        for key, val in sgd_params.items():
            f.write(f"  {key:<14}: {val}\n")
        f.write("\nSimulation Parameters\n")
        f.write("-" * 30 + "\n")
        f.write(f"  {'seed_batch_size':<20}: {SEED_BATCH_SIZE}\n")
        f.write(f"  {'confidence_threshold':<20}: {CONFIDENCE_THRESHOLD}\n")
        f.write(f"  {'audit_interval':<20}: {AUDIT_INTERVAL}\n")
        f.write(f"  {'safety_threshold':<20}: {SAFETY_THRESHOLD}\n")


def _write_report(all_results: list) -> None:
    """Write a single report covering all datasets and permutations."""
    report_path = os.path.join(BASE_OUTPUT, "report.txt")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    widths = [20, 14, 7, 10, 7, 9, 9, 9, 14, 18]
    headers = ["Run", "learning_rate", "eta0", "penalty", "alpha", "average",
               "REUSE %", "Avg MAE", "Safety Viol.", "Latency Saved (ms)"]
    separator = "-" * sum(w + 2 for w in widths)

    def row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    with open(report_path, "w") as f:
        f.write("SGD Permutation Report — Per-Dataset\n")
        f.write(f"Generated : {ts}\n")
        f.write(f"Data source: {CSV_DIR}\n\n")

        # ── Per-dataset table ───────────────────────────────────────────────
        datasets_seen = list(dict.fromkeys(r["dataset"] for r in all_results))
        for ds in datasets_seen:
            results = [r for r in all_results if r["dataset"] == ds]
            best_mae   = min(r["avg_mae"]   for r in results)
            best_reuse = max(r["reuse_pct"] for r in results)

            f.write(f"{'=' * len(separator)}\n")
            f.write(f"Dataset: {ds.upper()}\n")
            f.write(f"{'=' * len(separator)}\n")
            f.write(row(headers) + "\n")
            f.write(separator + "\n")

            for r in results:
                flags = ""
                if r["avg_mae"]   == best_mae:   flags += " <-- best MAE"
                if r["reuse_pct"] == best_reuse: flags += " <-- most REUSE"
                cells = [
                    r["label"],
                    r["learning_rate"],
                    r["eta0"],
                    r["penalty"],
                    r["alpha"],
                    r["average"],
                    f"{r['reuse_pct']:.1f}%",
                    f"{r['avg_mae']:.4f}",
                    r["safety_violations"],
                    f"{r['latency_saved_ms']:.0f}",
                ]
                f.write(row(cells) + flags + "\n")
            f.write("\n")

        # ── Cross-dataset compare & contrast ────────────────────────────────
        f.write(f"{'=' * len(separator)}\n")
        f.write("Cross-Dataset Compare & Contrast\n")
        f.write(f"{'=' * len(separator)}\n\n")

        for label, _ in PERMUTATIONS:
            runs = [r for r in all_results if r["label"] == label]
            f.write(f"[{label}]\n")
            for r in runs:
                f.write(
                    f"  {r['dataset']:<10}  REUSE={r['reuse_pct']:.1f}%  "
                    f"MAE={r['avg_mae']:.4f}  violations={r['safety_violations']}  "
                    f"latency_saved={r['latency_saved_ms']:.0f} ms\n"
                )
            f.write("\n")

        # ── Glossary ────────────────────────────────────────────────────────
        f.write(separator + "\n\n")
        f.write("Notes\n")
        f.write("-----\n")
        f.write("REUSE %        : fraction of frames skipping full detection (higher = more efficient).\n")
        f.write("Avg MAE        : mean absolute prediction error on audit frames (lower = better).\n")
        f.write("Safety Viol.   : REUSE decisions where actual confidence < safety threshold.\n")
        f.write("Latency Saved  : cumulative ms saved by REUSE over full detection.\n")

    print(f"\nOverall report written to {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_results = []

    for dataset_name, csv_file in DATASETS.items():
        csv_path = os.path.join(CSV_DIR, csv_file)
        print(f"\n{'#' * 60}")
        print(f"# Dataset: {dataset_name.upper()}  ({csv_path})")
        print(f"{'#' * 60}")

        data = pd.read_csv(csv_path)
        missing = [c for c in FEATURE_COLUMNS + ["target_confidence"] if c not in data.columns]
        if missing:
            print(f"  [SKIP] Missing columns: {missing}")
            continue

        X = data[FEATURE_COLUMNS].values
        y = data["target_confidence"].values

        if len(X) <= SEED_BATCH_SIZE:
            print(f"  [SKIP] Dataset too small ({len(X)} rows, need > {SEED_BATCH_SIZE})")
            continue

        for label, sgd_params in PERMUTATIONS:
            out_dir = os.path.join(BASE_OUTPUT, dataset_name, label)
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n{'=' * 50}")
            print(f"  Run : {label}")
            for k, v in sgd_params.items():
                print(f"    {k}: {v}")
            print(f"  out : {out_dir}")

            engine  = _run_sgd_simulation(X, y, sgd_params)
            metrics = _calculate_metrics(engine)

            generate_learning_curve(engine, os.path.join(out_dir, "learning_curve.png"))
            generate_confidence_comparison(engine, os.path.join(out_dir, "confidence_comparison.png"))
            generate_decision_histogram(engine, os.path.join(out_dir, "decision_histogram.png"))

            _write_params_file(out_dir, dataset_name, label, sgd_params)

            total     = metrics.total_frames
            reuse_pct = metrics.reuse_count / total * 100 if total else 0.0
            avg_mae   = (
                sum(metrics.mean_absolute_errors) / len(metrics.mean_absolute_errors)
                if metrics.mean_absolute_errors else float("nan")
            )

            all_results.append({
                "dataset":         dataset_name,
                "label":           label,
                **sgd_params,
                "reuse_pct":       reuse_pct,
                "avg_mae":         avg_mae,
                "safety_violations": metrics.safety_violations,
                "latency_saved_ms":  metrics.cumulative_latency_saved_ms,
            })

            print(f"  REUSE: {metrics.reuse_count} ({reuse_pct:.1f}%)  |  "
                  f"Safety violations: {metrics.safety_violations}  |  "
                  f"Avg MAE: {avg_mae:.4f}  |  "
                  f"Latency saved: {metrics.cumulative_latency_saved_ms:.0f} ms")

    _write_report(all_results)


if __name__ == "__main__":
    main()
