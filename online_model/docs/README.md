# Online Perception Engine Model

The `online_model` is a dynamic, self-adapting system designed to optimize vehicle perception pipelines. It balances computational efficiency with perception reliability by using online learning to predict when previous perception results can be safely reused.

## Objectives

- **Computational Savings**: Reduce the frequency of expensive `FULL_DETECTION` runs by substituting them with low-latency `REUSE` operations.
- **Dynamic Adaptation**: Use an online regressor (`SGDRegressor`) to adapt to changing environments and sensor distributions in real-time.
- **Safety-First Gating**: Implement a confidence-based decision threshold to ensure results are only reused when the predicted reliability is high.

## Core Architecture

The engine orchestrates a continuous feedback loop:

1.  **Warm-up**: Initializes using a `seed_batch` of data to "warm up" the model and feature scaler.
2.  **Prediction**: For each incoming frame, the engine extracts features and predicts a `target_confidence` score.
3.  **Decision**:
    -   **REUSE**: Chosen if `predicted_confidence > confidence_threshold`. Approximate latency: ~0.5ms.
    -   **FULL_DETECTION**: Chosen otherwise. Approximate latency: ~80ms.
4.  **Audit & Update**: Every `audit_interval` frames, the system runs a full detection to get ground-truth confidence. The model is then updated incrementally using `partial_fit` to improve future predictions.

## Module Structure

-   [`engine.py`]: Contains the `OnlinePerceptionEngine` class, the main orchestrator of the logic.
-   [`simulation.py`]: Provides a runner to simulate the engine's performance on recorded data (CSV).
-   [`types.py`]: Defines structured data classes for `FrameDecision` and `PerformanceMetrics`.
-   [`visualization.py`]: Logic for generating performance plots (Learning Curves, Confidence Comparison).

## Usage

You can run a simulation of the engine using the command-line interface:

```bash
python -m online_model --output-dir output
```

### Key Parameters

-   `--confidence-threshold`: Predicted confidence required to allow `REUSE` (default: 0.85).
-   `--audit-interval`: Frequency of model updates (default: 5 frames).
-   `--seed-batch-size`: Number of initial frames for model initialization (default: 50).

## Results Interpretation

### Metrics Summary
-   **Reuse Percentage**: Indicates the effective reduction in computational load.
-   **Cumulative Latency Saved**: Total time (ms) saved compared to running full detection on every frame.
-   **Safety Violations**: Occurrences where `REUSE` was selected but the actual confidence was below the `safety_threshold`.
-   **MAE (Mean Absolute Error)**: Shows the average prediction error of the confidence model.

### Visualizations
-   **Learning Curve**: Plots prediction error over time. A downward trend indicates the model is successfully adapting.
-   **Confidence Comparison**: Overlays predicted vs. actual confidence. It helps visualize how well the model mimics the true perception reliability.
