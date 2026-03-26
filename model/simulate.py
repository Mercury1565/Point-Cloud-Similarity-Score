import os
import time
import pandas as pd
import numpy as np
import joblib
import pickle

def load_pkl(path):
    """Safely load a pickle file either via joblib or pickle."""
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    model_path = 'confidence_rf_model.pkl'
    data_path = 'training_data.csv'
    
    print(f"Loading assets: {model_path} and {data_path}...")
    model = load_pkl(model_path)
    df = pd.read_csv(data_path)
    
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
    features = df.drop(columns=['target_confidence'])
    targets = df['target_confidence'].values
    
    print("\nRunning Simulation Loop...")
    start_time = time.time()
    
    # Simulation Loop
    for i in range(total_frames):
        # Extract features for current frame
        frame_features = features.iloc[[i]]
        actual_target = targets[i]
        
        # Predict confidence using the lightweight RF model
        pred_conf = model.predict(frame_features)[0]
        
        # Decision Logic
        if pred_conf > REUSE_THRESHOLD:
            action = "REUSE"
            cost = RF_MODEL_LATENCY
            reuse_count += 1
            
            # Safety Check: erroneous skip
            if actual_target < 0.75:
                safety_failures += 1
        else:
            action = "DETECT"
            cost = RF_MODEL_LATENCY + HEAVY_DETECTOR_LATENCY
            detect_count += 1
            
        cumulative_latency_system += cost
        
    sim_dur = time.time() - start_time
    print(f"Simulation completed in {sim_dur:.3f} seconds.\n")
    
    # Performance Metrics Calculations
    skip_rate = (reuse_count / total_frames) * 100.0

    # Calculate latency per 100 frames
    avg_latency_100 = (cumulative_latency_system / total_frames) * 100
    
    # Print Metrics
    print("=" * 55)
    print("                PERFORMANCE METRICS")
    print("=" * 55)
    print(f"Total Frames Processed : {total_frames}")
    print(f"Skip Rate              : {skip_rate:.2f}%")
    print(f"Cumulative Latency     : {cumulative_latency_system:.2f} ms")
    print(f"Latency per 100 Frames : {avg_latency_100:.2f} ms")
    print(f"Safety Check           : {safety_failures} failures (where predicted > {REUSE_THRESHOLD} but actual < 0.75)")
    print("=" * 55)

if __name__ == "__main__":
    main()
