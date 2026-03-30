# PROJECT README

## System Architecture

1.  **Raw Data** (nuScenes / KITTI) $\rightarrow$ **Data Extraction** (`extract/`)
2.  **Unified JSON** $\rightarrow$ **Feature & Target Generation** $\rightarrow$ **Training Data** (CSV)
3.  **Perception Gatekeeper**:
    *   **Static Model** (`model/`)
    *   **Online Engine** (`online_model/`)
4.  **REUSE / DETECT Policy** $\rightarrow$ **Computational Savings & ROI Analysis**

## Module Breakdown

### 1. Data Extraction (`extract/`)
Converts raw datasets into a standardized format and prepares training features.
-   **Unified JSON**: A hierarchical "Scene-Frame-Object" structure. See [UNIFIED_JSON.md](extract/docs/UNIFIED_JSON.md).
-   **Chamfer Distance**: A raw geometric change proxy computed between temporal point clouds.
-   **Multi-Dataset Support**: Specialized extractors for nuScenes and KITTI.
-   **Audit Tools**: visualize similarity trends and sanity-check extracted data. See [DATA_AUDIT.md](extract/docs/DATA_AUDIT.md).

### 2. Confidence Scoring (`confidence_scorer/`)
The core library that defines the "Ground Truth" for model training.
-   **Identity Tracking (F1)**: Measures how consistently object IDs persist.
-   **Geometric Stability (3D mIoU)**: Measures the spatial drift of tracked 3D bounding boxes.
-   **Composite Score**: Combines tracking and geometry (defaulting to Harmonic Mean) to penalize imbalances.
-   *Deep Dive*: [CONFIDENCE_SCORER.md](confidence_scorer/docs/CONFIDENCE_SCORER.md).

### 3. Static Random Forest Model (`model/`)
An offline approach for training a robust predictor.
-   **Training**: Trains a `RandomForestRegressor` on features like `ego_vel`, `chamfer_dist`, and `obj_count`.
-   **Simulation**: Evaluates real-world ROI, cumulative latency savings, and "Safety Failures" (erroneous skips).
-   *Implementation Details*: [RANDOM_FOREST_BASED_STATIC_MODEL.md](model/docs/RANDOM_FOREST_BASED_STATIC_MODEL.md).

### 4. Online Perception Engine (`online_model/`)
A self-adapting system that learns your vehicle's specific environment in real-time.
-   **Online Learning**: Uses `SGDRegressor` with incremental updates (`partial_fit`).
-   **Feedback Loop**: Periodically audits predictions with full detections to refine the model.
-   **Dynamic Thresholding**: Adjusts to changing sensor noise or traffic patterns.
-   *Engine Docs*: [online_model/README.md](online_model/docs/README.md).

## Features & Target Definition

| Feature | Definition |
| :--- | :--- |
| `chamfer_dist` | Symmetric distance between point clouds at $t$ and $t-1$. |
| `ego_vel` | Scalar velocity of the vehicle. |
| `obj_count` | Total unique objects detected in the current frame. |
| `avg_dist` | Mean distance of objects relative to the ego-vehicle. |
| **`target_confidence`** | **Ground Truth Label** (Harmonic mean of F1 and 3D mIoU). |

See the full list in [FEATURES_AND_TARGET.md](extract/docs/FEATURES_AND_TARGET.md).

## 🛠️ Quickstart

### 1. Data Preparation
Ensure your raw data is in `data/`, then run extraction:
```bash
python extract/extract_nuscenes.py
python extract/generate_csv_nuscenes.py
```

```bash
python extract/extract_kitti.py
python extract/generate_csv_kitti.py
```

### 2. Static Pipeline (Training & Simulation)
```bash
python model/train_confidence_model.py
python model/simulate.py
```

### 3. Online Adaptation
Simulate the self-adapting engine on your recorded data:
```bash
python -m online_model --confidence-threshold 0.85
```

## Analysis & Results
The pipeline generates several artifacts in the `output/` directory for interpreting performance:
-   `predicted_vs_actual.png`: Accuracy of confidence predictions.
-   `feature_importance.png`: Which signals (e.g., ego-speed vs. geometric change) drive the skip decision.
-   `nuscenes_similarity_trends.png`: Visual correlation between raw sensor change and perception stability.

---

### Referenced Documentation
- [Extraction Schema](extract/docs/UNIFIED_JSON.md)
- [Scoring Logic](confidence_scorer/docs/CONFIDENCE_SCORER.md)
- [Static Model Documentation](model/docs/RANDOM_FOREST_BASED_STATIC_MODEL.md)
- [Online Engine Documentation](online_model/docs/README.md)
