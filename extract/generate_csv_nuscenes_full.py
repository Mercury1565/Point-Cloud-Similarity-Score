"""
unified_nuscenes_full.json → data/csv/nuscenes_full_training_data.csv

Logic is identical to generate_csv_nuscenes.py.  The only differences are:
  - Input:  unified_nuscenes_full.json  (~20 Hz sweeps, interpolated annotations)
  - Output: data/csv/nuscenes_full_training_data.csv

Run extract/extract_nuscenes_full.py first to generate the input file.

Usage:
    python extract/generate_csv_nuscenes_full.py
"""

import json
import csv
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from confidence_scorer.scorer import ConfidenceScorer


def calculate_avg_dist(objects: list) -> float:
    if not objects:
        return 0.0
    total = 0.0
    for obj in objects:
        x, y = obj["bbox"][0], obj["bbox"][1]
        total += math.sqrt(x ** 2 + y ** 2)
    return total / len(objects)


def calculate_fastest_vel(objects: list) -> float:
    if not objects:
        return 0.0
    max_v = 0.0
    for obj in objects:
        v_vec = obj.get("velocity", [0.0, 0.0, 0.0])
        v_scalar = math.sqrt(sum(x ** 2 for x in v_vec))
        if v_scalar > max_v:
            max_v = v_scalar
    return max_v


def calculate_extreme_distances(objects: list) -> tuple:
    if not objects:
        return 0.0, 0.0
    dists = [math.sqrt(obj["bbox"][0] ** 2 + obj["bbox"][1] ** 2) for obj in objects]
    return min(dists), max(dists)


def safe_float(val: float) -> float:
    return val if math.isfinite(val) else 0.0


def main():
    # Reset label weighting to baseline
    m_f1 = 0.5
    m_miou = 0.5
    scorer = ConfidenceScorer(w_f1=m_f1, w_miou=m_miou)

    input_file = Path("unified_nuscenes_full.json")
    if not input_file.exists():
        raise FileNotFoundError(
            f"{input_file} not found. Run extract/extract_nuscenes_full.py first."
        )

    with open(input_file, "r") as f:
        data = json.load(f)

    dataset = []
    skipped_pairs = 0

    for scene in data:
        frames = scene["frame_list"]
        # At ~20 Hz, consecutive frames are ~0.05 s apart.
        # dt is not used directly as a feature but noted for reference.

        for t in range(1, len(frames)):
            frame_t    = frames[t]
            frame_prev = frames[t - 1]

            objs_t    = frame_t["object_list"]
            objs_prev = frame_prev["object_list"]

            # Skip pairs where both frames have no objects — no useful signal.
            if not objs_t and not objs_prev:
                skipped_pairs += 1
                continue

            chamfer_dist       = frame_t["chamfer_distance"]
            ego_vel            = frame_t["ego_vel"]
            obj_count          = len(objs_t)
            avg_dist           = calculate_avg_dist(objs_t)
            fastest_obj_vel    = calculate_fastest_vel(objs_prev)
            nearest_obj_dist, farthest_obj_dist = calculate_extreme_distances(objs_prev)
            score = scorer.calculate_score(objs_t, objs_prev)
            f1 = score["f1"]
            miou = score["miou"]
            target_confidence = score["confidence_score"]

            row = [
                scene["scene_id"],
                safe_float(chamfer_dist),
                safe_float(ego_vel),
                float(obj_count),
                safe_float(avg_dist),
                safe_float(fastest_obj_vel),
                safe_float(nearest_obj_dist),
                safe_float(farthest_obj_dist),
                safe_float(f1),
                safe_float(miou),
                safe_float(target_confidence),
            ]
            dataset.append(row)

    headers = [
        "scene_id",
        "chamfer_dist",
        "ego_vel",
        "obj_count",
        "avg_dist",
        "fastest_obj_vel",
        "nearest_obj_dist",
        "farthest_obj_dist",
        "f1",
        "miou",
        "target_confidence",
    ]

    output_dir  = Path("data/csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "nuscenes_full_training_data.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(dataset)

    print(f"Generated {len(dataset)} rows -> {output_file}")
    if skipped_pairs:
        print(f"Skipped {skipped_pairs} frame pairs with no objects in either frame")
    print()

    header_fmt = "".join([f"{h:>18}" for h in headers])
    print(header_fmt)
    for row in dataset[:5]:
        print("".join([
            f"{v:>18}" if isinstance(v, str) else f"{v:>18.4f}"
            for v in row
        ]))


if __name__ == "__main__":
    main()
