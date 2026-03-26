"""
Prepares a tabular ML dataset from unified_nuscenes_mini.json.
Generates temporal features and the ConfidenceScore target for each frame transition.
"""

import json
import csv
import math
import sys
from pathlib import Path

# Add parent directory to path to find confidence_scorer
sys.path.append(str(Path(__file__).resolve().parent.parent))
from confidence_scorer.scorer import ConfidenceScorer


def calculate_avg_dist(objects: list) -> float:
    """Calculate the mean Euclidean distance of all objects to the ego vehicle."""
    if not objects:
        return 0.0
    
    total_dist = 0.0
    for obj in objects:
        x, y = obj['bbox'][0], obj['bbox'][1]
        dist = math.sqrt(x**2 + y**2)
        total_dist += dist
        
    avg = total_dist / len(objects)
    # Handle nan/inf just in case math weirdness happens
    if not math.isfinite(avg):
        return 0.0
    return avg


def safe_float(val: float) -> float:
    """Replaces NaN/Inf with 0.0"""
    if not math.isfinite(val):
        return 0.0
    return val


def main():
    scorer = ConfidenceScorer()
    
    # Load the unified JSON representation
    with open('unified_nuscenes_mini.json', 'r') as f:
        data = json.load(f)
        
    dataset = []
    
    # Process each scene
    for scene in data:
        frames = scene['frame_list']
        # Note: nuScenes samples are captured at ~2Hz, which means dt is ~0.5s.
        dt = 0.5
        
        # We start looking backwards, so skip the first frame (t=0)
        for t in range(1, len(frames)):
            frame_t = frames[t]
            frame_prev = frames[t-1]
            
            # 1. Direct Features
            chamfer_dist = frame_t['chamfer_distance']
            ego_vel = frame_t['ego_vel']
            
            # 2. Temporal Physics
            delta_ego_vel = ego_vel - frame_prev['ego_vel']
            ego_accel = delta_ego_vel / dt
            
            # 3. Scene Complexity
            objs_t = frame_t['object_list']
            objs_prev = frame_prev['object_list']
            
            obj_count = len(objs_t)
            delta_obj_count = obj_count - len(objs_prev)
            
            # 4. Spatial Dynamics
            avg_dist = calculate_avg_dist(objs_t)
            
            # 5. Target Generation
            target_confidence = scorer.calculate_score(objs_t, objs_prev)['confidence_score']
            
            # Clean elements
            row = [
                safe_float(chamfer_dist),
                safe_float(ego_vel),
                safe_float(delta_ego_vel),
                safe_float(ego_accel),
                float(obj_count),
                float(delta_obj_count),
                safe_float(avg_dist),
                safe_float(target_confidence)
            ]
            dataset.append(row)
            
    # Write to CSV
    headers = [
        "chamfer_dist", "ego_vel", "delta_ego_vel", "ego_accel", 
        "obj_count", "delta_obj_count", "avg_dist", "target_confidence"
    ]
    
    output_path = "training_data.csv"
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(dataset)
        
    # Validation & Logging
    print(f"✅ Generated tabular dataset at '{output_path}'")
    print(f"Total training samples generated: {len(dataset)}")
    print("-" * 50)
    print("First 5 rows (Preview):")
    
    # Print header nicely aligned
    header_fmt = "".join([f"{h:>15}" for h in headers])
    print(header_fmt)
    
    # Print top 5 values nicely formatted
    for row in dataset[:5]:
        row_fmt = "".join([f"{val:>15.4f}" for val in row])
        print(row_fmt)


if __name__ == "__main__":
    main()
