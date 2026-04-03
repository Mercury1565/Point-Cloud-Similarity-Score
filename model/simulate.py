import os
import time
import pandas as pd
import numpy as np
import joblib
import pickle
import glob

FEATURE_COLUMNS = [
    "chamfer_dist",
    "ego_vel",
    "obj_count",
    "avg_dist",
    "fastest_obj_vel",
    "nearest_obj_dist",
    "farthest_obj_dist",
]

HEAVY_DETECTOR_LATENCY = 80.0   # ms
RF_MODEL_LATENCY       = 0.5    # ms
REUSE_THRESHOLD        = 0.85
SAFETY_THRESHOLD       = 0.75


def load_pkl(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def main():
    model_path = "confidence_rf_model.pkl"
    data_dir   = os.path.join("data", "csv")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {data_dir}")
        return

    print(f"Loading model: {model_path}")
    model = load_pkl(model_path)

    print(f"Loading {len(csv_files)} data files from {data_dir}...")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    if "target_confidence" not in df.columns:
        print("Error: 'target_confidence' column missing from data.")
        return

    has_scene_id = "scene_id" in df.columns

    # ── Scene-aware or flat processing ────────────────────────────────────────
    records = []   # one dict per frame, used for output CSV

    if has_scene_id:
        # Drop rows where scene_id is NaN (CSVs generated before scene_id was added)
        df = df.dropna(subset=["scene_id"])
        scene_ids = list(dict.fromkeys(df["scene_id"].tolist()))
        print(f"\nProcessing {len(scene_ids)} scenes sequentially...")
        start_time = time.time()

        for sid in scene_ids:
            scene_df = df[df["scene_id"] == sid].reset_index(drop=True)
            if scene_df.empty:
                continue
            # Pass DataFrame (not .values) so feature names match what the model was fitted with
            X_scene = scene_df[FEATURE_COLUMNS]
            y_scene = scene_df["target_confidence"].values

            preds = model.predict(X_scene)

            for local_idx, (pred_conf, actual) in enumerate(zip(preds, y_scene)):
                decision = "REUSE" if pred_conf > REUSE_THRESHOLD else "FULL_DETECTION"
                safety_violation = (decision == "REUSE" and actual < SAFETY_THRESHOLD)
                records.append({
                    "scene_id":            sid,
                    "frame_idx":           local_idx,
                    "predicted_confidence": round(float(pred_conf), 6),
                    "actual_confidence":    round(float(actual), 6),
                    "decision":            decision,
                    "safety_violation":    safety_violation,
                })
    else:
        print("\nWarning: no scene_id column — processing flat (re-generate CSVs to fix this).")
        X      = df[FEATURE_COLUMNS].values
        y      = df["target_confidence"].values
        start_time = time.time()
        preds  = model.predict(X)

        for i, (pred_conf, actual) in enumerate(zip(preds, y)):
            decision = "REUSE" if pred_conf > REUSE_THRESHOLD else "FULL_DETECTION"
            safety_violation = (decision == "REUSE" and actual < SAFETY_THRESHOLD)
            records.append({
                "scene_id":            "unknown",
                "frame_idx":           i,
                "predicted_confidence": round(float(pred_conf), 6),
                "actual_confidence":    round(float(actual), 6),
                "decision":            decision,
                "safety_violation":    safety_violation,
            })

    sim_dur = time.time() - start_time
    print(f"Simulation completed in {sim_dur:.3f}s\n")

    # ── Save sequential inference output ──────────────────────────────────────
    out_df      = pd.DataFrame(records)
    output_path = os.path.join(output_dir, "inference_output.csv")
    out_df.to_csv(output_path, index=False)
    print(f"Inference output saved to {output_path}")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    total_frames  = len(records)
    reuse_count   = sum(1 for r in records if r["decision"] == "REUSE")
    detect_count  = total_frames - reuse_count
    safety_fails  = sum(1 for r in records if r["safety_violation"])
    cum_latency   = (reuse_count * RF_MODEL_LATENCY
                     + detect_count * (RF_MODEL_LATENCY + HEAVY_DETECTOR_LATENCY))

    skip_rate        = (reuse_count / total_frames) * 100 if total_frames else 0
    avg_latency_100  = (cum_latency / total_frames) * 100 if total_frames else 0

    print("=" * 55)
    print("                PERFORMANCE METRICS")
    print("=" * 55)
    print(f"Files Processed        : {len(csv_files)}")
    print(f"Total Frames Processed : {total_frames}")
    print(f"Skip Rate              : {skip_rate:.2f}%")
    print(f"Cumulative Latency     : {cum_latency:.2f} ms")
    print(f"Latency per 100 Frames : {avg_latency_100:.2f} ms")
    print(f"Safety Failures        : {safety_fails} (REUSE but actual < {SAFETY_THRESHOLD})")
    print("=" * 55)


if __name__ == "__main__":
    main()
