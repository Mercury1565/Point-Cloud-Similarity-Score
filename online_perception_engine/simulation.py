"""Simulation runner for the Online Perception Engine."""

import os
import pandas as pd
import numpy as np
from online_perception_engine.engine import OnlinePerceptionEngine
from online_perception_engine.types import PerformanceMetrics
from online_perception_engine.visualization import (
    generate_learning_curve,
    generate_confidence_comparison
)


def run_simulation(
    csv_path: str,
    visuals_output_dir: str,
    confidence_threshold: float = 0.85,
    audit_interval: int = 5,
    seed_batch_size: int = 50,
    reuse_latency_ms: float = 0.5,
    full_detection_latency_ms: float = 80.0,
    safety_threshold: float = 0.70
) -> PerformanceMetrics:
    """Run the online learning simulation on CSV data.
    
    Args:
        csv_path: Path to CSV file containing training data
        confidence_threshold: Predicted confidence above which REUSE is chosen (default 0.85)
        audit_interval: Number of frames between model updates (default 5)
        seed_batch_size: Initial number of frames for model warm-up (default 50)
        reuse_latency_ms: Latency in milliseconds for REUSE decision (default 0.5)
        full_detection_latency_ms: Latency in milliseconds for FULL_DETECTION (default 80.0)
        safety_threshold: Actual confidence below which REUSE is unsafe (default 0.70)
    
    Returns:
        PerformanceMetrics object containing aggregated statistics
    """
    # Load CSV data using pandas
    data = pd.read_csv(csv_path)
    
    # Extract feature columns into X array
    feature_columns = [
        "chamfer_dist",
        "ego_vel",
        "delta_ego_vel",
        "ego_accel",
        "obj_count",
        "delta_obj_count",
        "avg_dist"
    ]
    X = data[feature_columns].values
    
    # Extract target_confidence column into y array
    y = data["target_confidence"].values
    
    # Validate dataset has more rows than seed_batch_size
    if len(X) <= seed_batch_size:
        raise ValueError(
            f"Dataset must have more than {seed_batch_size} rows, got {len(X)}"
        )
    
    # Split data into seed batch (first seed_batch_size rows) and streaming frames (remaining rows)
    X_seed = X[:seed_batch_size]
    y_seed = y[:seed_batch_size]
    
    # Create OnlinePerceptionEngine instance
    engine = OnlinePerceptionEngine(
        confidence_threshold=confidence_threshold,
        audit_interval=audit_interval,
        seed_batch_size=seed_batch_size,
        reuse_latency_ms=reuse_latency_ms,
        full_detection_latency_ms=full_detection_latency_ms,
        safety_threshold=safety_threshold
    )
    
    # Call initialize_model() with seed batch
    engine.initialize_model(X_seed, y_seed)
    
    # Loop through streaming frames: call process_frame() for each frame
    for frame_idx in range(seed_batch_size, len(X)):
        X_current = X[frame_idx]
        y_actual = y[frame_idx]
        engine.process_frame(frame_idx, X_current, y_actual)

     
    # Generate learning curve plot
    learning_curve_path = os.path.join(visuals_output_dir, "learning_curve.png")
    generate_learning_curve(engine, learning_curve_path)
    print(f"  - Learning curve saved to {learning_curve_path}")
    
    # Generate confidence comparison plot
    confidence_comparison_path = os.path.join(visuals_output_dir, "confidence_comparison.png")
    generate_confidence_comparison(engine, confidence_comparison_path)
    print(f"  - Confidence comparison saved to {confidence_comparison_path}")
    
    # Call calculate_metrics() and return result
    metrics = engine.calculate_metrics()
    return metrics
