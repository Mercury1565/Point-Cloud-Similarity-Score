import os
import glob
import pandas as pd
import numpy as np
from online_model.engine import OnlinePerceptionEngine
from online_model.types import PerformanceMetrics
from online_model.visualization import (
    generate_learning_curve,
    generate_confidence_comparison,
    generate_decision_histogram
)


def run_simulation(
    csv_dir: str,
    visuals_output_dir: str,
    confidence_threshold: float = 0.85,
    audit_interval: int = 5,
    seed_batch_size: int = 50,
    reuse_latency_ms: float = 0.5,
    full_detection_latency_ms: float = 80.0,
    safety_threshold: float = 0.70
) -> PerformanceMetrics:
    """Run the online learning simulation on a directory of CSV data."""
    
    # 1. Gather all CSV files from the directory
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {csv_dir}")
    
    print(f"Loading {len(csv_files)} files for online simulation...")
    
    # 2. Load and concatenate
    df_list = []
    for file in csv_files:
        df_list.append(pd.read_csv(file))
    
    data = pd.concat(df_list, ignore_index=True)
    
    # 3. Define and validate features
    feature_columns = [
        "chamfer_dist",
        "ego_vel",
        "obj_count",
        "avg_dist",
        "fastest_obj_vel",
        "nearest_obj_dist",
        "farthest_obj_dist"
    ]
    
    # Ensure all required columns exist
    missing_cols = [c for c in feature_columns + ["target_confidence"] if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    X = data[feature_columns].values
    y = data["target_confidence"].values
    
    # Validate dataset size
    if len(X) <= seed_batch_size:
        raise ValueError(
            f"Combined dataset must have more than {seed_batch_size} rows, got {len(X)}"
        )
    
    X_seed = X[:seed_batch_size]
    y_seed = y[:seed_batch_size]
    
    # 4. Initialize Engine
    engine = OnlinePerceptionEngine(
        confidence_threshold=confidence_threshold,
        audit_interval=audit_interval,
        seed_batch_size=seed_batch_size,
        reuse_latency_ms=reuse_latency_ms,
        full_detection_latency_ms=full_detection_latency_ms,
        safety_threshold=safety_threshold
    )
    
    engine.initialize_model(X_seed, y_seed)
    
    # 5. Process Streaming Frames
    print(f"Processing {len(X) - seed_batch_size} streaming frames...")
    for frame_idx in range(seed_batch_size, len(X)):
        X_current = X[frame_idx]
        y_actual = y[frame_idx]
        engine.process_frame(frame_idx, X_current, y_actual)

    # 6. Generate Visuals
    os.makedirs(visuals_output_dir, exist_ok=True)
    
    learning_curve_path = os.path.join(visuals_output_dir, "learning_curve.png")
    generate_learning_curve(engine, learning_curve_path)
    
    confidence_comparison_path = os.path.join(visuals_output_dir, "confidence_comparison.png")
    generate_confidence_comparison(engine, confidence_comparison_path)

    decision_histogram_path = os.path.join(visuals_output_dir, "decision_histogram.png")
    generate_decision_histogram(engine, decision_histogram_path)
    
    print(f"Simulation visuals saved to {visuals_output_dir}")
    
    return engine.calculate_metrics()