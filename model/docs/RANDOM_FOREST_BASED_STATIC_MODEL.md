# Random Forest-Based Static Model Documentation

This document describes the implementation, training, and usage of the Random Forest-based static model in the `model` module. This model is designed to predict perception confidence and enable computational savings through inference skipping.

## Overview

The `model` module provides a solution for gating a "heavy" 3D detector (e.g., in an autonomous driving perception pipeline) using a lightweight Random Forest model. By predicting whether the current scene's perception results are likely to be highly confident, the system can decide whether to skip the heavy detector and reuse previous results, thereby saving significant computational resources.

### Key Components

- **`train_confidence_model.py`**: A script for loading training data (CSV), training a `RandomForestRegressor`, evaluating it, and saving the model.
- **`simulate.py`**: A simulation tool that evaluates the performance and ROI (Return on Investment) of the model in a real-time-like inference loop.
- **`confidence_rf_model.pkl`**: The serialized Random Forest model (generated after training).

## Model Implementation

### Training Process (`train_confidence_model.py`)

The training pipeline consists of the following steps:

1.  **Data Loading**: Loads all CSV files from `data/csv/`, merges them, and separates features from the target column (`target_confidence`).
2.  **Preprocessing**: Basic data validation to ensure no NaN or infinite values are present in the features or target.
3.  **Splitting**: Splits the data into training (80%) and testing (20%) sets.
4.  **Training**: Initializes a `RandomForestRegressor` with the following default parameters:
    - `n_estimators`: 100
    - `max_depth`: 10
    - `n_jobs`: -1 (uses all available CPU cores)
5.  **Evaluation**: Calculates performance metrics:
    - **Mean Absolute Error (MAE)**: Average magnitude of prediction errors.
    - **R² Score**: Proportion of variance explained by the model.
    - **Skip Percentage**: Percentage of frames where predicted confidence is above a threshold.
    - **Skip Accuracy**: Accuracy of identifies "safe-to-skip" frames (where both predicted and actual confidence are above the threshold).
6.  **Visualization**: Generates "Predicted vs Actual" and "Feature Importance" plots in the `output/` directory.
7.  **Serialization**: Saves the model to `confidence_rf_model.pkl` using `joblib`.

### Simulation and ROI (`simulate.py`)

The simulation script demonstrates the model's practical utility:

- **Heavy Detector Latency**: 80.0 ms (simulated)
- **RF Model Latency**: 0.5 ms (simulated)
- **Reuse Threshold**: 0.85 (confidence level required to skip detection)

#### Decision Logic
For each frame:
- Predict confidence using the RF model.
- If `pred_conf > 0.85`: Skip the heavy detector and incur only the 0.5 ms RF cost.
- Else: Run the heavy detector and incur a combined 80.5 ms cost.

#### Safety Check
A "Safety Failure" is logged if the model predicts a confidence `> 0.85` but the actual target confidence was `< 0.75`.

## Usage Instructions

### Training the Model

To train the model from your collected data:

```bash
python model/train_confidence_model.py
```
*Requirement: Ensure your training data is located in `data/csv/`.*

### Running the Simulation

To evaluate the model's performance and latency savings:

```bash
python model/simulate.py
```

## Performance Metrics

| Metric | Description |
| :--- | :--- |
| **Skip Rate** | Percentage of frames where detection was skipped. |
| **Cumulative Latency** | Total time spent during the simulation. |
| **Latency per 100 Frames** | Average time cost for every 100 frames. |
| **Safety Failures** | Count of instances where a skip was erroneously predicted. |
