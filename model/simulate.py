import os
import time
import pandas as pd
import numpy as np
import joblib
import pickle
import glob

def load_pkl(path):
    """Safely load a pickle file either via joblib or pickle."""
    try:
        return joblib.load(path)
    except Exception:
        with open(path, 'rb') as f:
            return pickle.load(f)

def main():
    model_path = 'confidence_rf_model.pkl'
    # Updated to point to the directory
    data_dir = os.path.join('data', 'csv')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    # 1. Find and load all CSV files
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {data_dir}")
        return

    print(f"Loading model: {model_path}")
    model = load_pkl(model_path)
    
    print(f"Loading {len(csv_files)} data files from {data_dir}...")
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    
    # Define System Constants
    HEAVY_DETECTOR_LATENCY = 80.0  # ms
    RF_MODEL_LATENCY = 0.5         # ms
    REUSE_THRESHOLD = 0.85
    
    total_frames = len(df)
    reuse_count = 0
    detect_count = 0
    cumulative_latency_system = 0.0
    safety_failures = 0
    
    # Isolate features and target confidence
    if 'target_confidence' not in df.columns:
        print("Error: 'target_confidence' column missing from data.")
        return
        
    features = df.drop(columns=['target_confidence'])
    targets = df['target_confidence'].values
    
    print(f"\nRunning Simulation Loop on {total_frames} frames...")
    start_time = time.time()
    
    # Simulation Loop
    # Note: iterating via iloc can be slow for very large DataFrames. 
    # For massive datasets, consider model.predict(features) in bulk first.
    for i in range(total_frames):
        frame_features = features.iloc[[i]]
        actual_target = targets[i]
        
        # Predict confidence
        pred_conf = model.predict(frame_features)[0]
        
        # Decision Logic
        if pred_conf > REUSE_THRESHOLD:
            cost = RF_MODEL_LATENCY
            reuse_count += 1
            
            # Safety Check: erroneous skip
            if actual_target < 0.75:
                safety_failures += 1
        else:
            cost = RF_MODEL_LATENCY + HEAVY_DETECTOR_LATENCY
            detect_count += 1
            
        cumulative_latency_system += cost
        
    sim_dur = time.time() - start_time
    print(f"Simulation completed in {sim_dur:.3f} seconds.\n")
    
    # Performance Metrics
    skip_rate = (reuse_count / total_frames) * 100.0 if total_frames > 0 else 0
    avg_latency_100 = (cumulative_latency_system / total_frames) * 100 if total_frames > 0 else 0
    
    print("=" * 55)
    print("                PERFORMANCE METRICS")
    print("=" * 55)
    print(f"Files Processed        : {len(csv_files)}")
    print(f"Total Frames Processed : {total_frames}")
    print(f"Skip Rate              : {skip_rate:.2f}%")
    print(f"Cumulative Latency     : {cumulative_latency_system:.2f} ms")
    print(f"Latency per 100 Frames : {avg_latency_100:.2f} ms")
    print(f"Safety Check           : {safety_failures} failures (where predicted > {REUSE_THRESHOLD} but actual < 0.75)")
    print("=" * 55)

if __name__ == "__main__":
    main()