"""
nuScenes full-frequency extractor (non-keyframe, ~20 Hz).

Differences from extract_nuscenes.py (keyframe-only, 2 Hz):
  - Iterates the sample_data["next"] chain for LIDAR_TOP (every LiDAR sweep)
    instead of the sample["next"] chain (only annotated keyframes).
  - Object annotations come from the interpolated pkl files in
    PKL_DIR rather than from nuScenes sample_annotation records,
    which only exist on keyframes.
  - Output: unified_nuscenes_full.json  (same schema as unified_nuscenes_mini.json)

Usage:
    python extract/extract_nuscenes_full.py
"""

import json
import os
import pickle
import glob
import numpy as np
from scipy.spatial import cKDTree
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

# ── Configuration ─────────────────────────────────────────────────────────────
DATAROOT   = "/media/tersiteab/e4d56274-b9be-4c36-8ee8-22a4e69c1bc9/home/tersiteab/Documents/PointCloudResearch/frame_similarity/nu_data"
VERSION    = "v1.0-mini"
PKL_DIR    = "data/csv/intermediate_ann_new"
OUTPUT     = "unified_nuscenes_full.json"
LIDAR_CHAN = "LIDAR_TOP"
DOWNSAMPLE = 500
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


# ── Helpers (identical to extract_nuscenes.py) ─────────────────────────────────
def load_pointcloud(path: str) -> np.ndarray:
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    return pts[:, :3]


def downsample(pts: np.ndarray, n: int) -> np.ndarray:
    if len(pts) <= n:
        return pts
    idx = rng.choice(len(pts), size=n, replace=False)
    return pts[idx]


def chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a, k=1)
    d_ba, _ = tree_a.query(pts_b, k=1)
    return float(np.mean(d_ab) + np.mean(d_ba))


def quat_to_yaw(rotation: list) -> float:
    return float(Quaternion(rotation).yaw_pitch_roll[0])


def build_keyframe_object_list(nusc: NuScenes, sample_token: str) -> list:
    """Fallback annotation loader for keyframes missing in interpolated pkl."""
    sample = nusc.get("sample", sample_token)
    objects = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        x, y, z = ann["translation"]
        w, l, h = ann["size"]
        yaw = quat_to_yaw(ann["rotation"])

        vel = nusc.box_velocity(ann["token"])
        if np.any(np.isnan(vel)):
            vel = np.array([0.0, 0.0, 0.0])

        objects.append({
            "obj_id": ann["instance_token"],
            "label": ann["category_name"],
            "bbox": [x, y, z, w, l, h, yaw],
            "velocity": vel.tolist(),
        })
    return objects


# ── PKL loading ────────────────────────────────────────────────────────────────
def load_all_pkl(pkl_dir: str) -> dict:
    """
    Load all scene_nkf_annotations*.pkl files and merge into a flat dict:
        { scene_name (str) → { frame_token (str) → frame_dict } }
    """
    merged = {}
    pkl_files = sorted(glob.glob(os.path.join(pkl_dir, "scene_nkf_annotations*.pkl")))
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files found in: {pkl_dir}")
    for path in pkl_files:
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Each file contains exactly one scene
        for scene_name, scene_frames in data.items():
            merged[scene_name] = scene_frames
    print(f"Loaded interpolated annotations for {len(merged)} scenes from {len(pkl_files)} pkl files")
    return merged


# ── Scene extraction ───────────────────────────────────────────────────────────
def extract_scene_full(nusc: NuScenes, scene: dict, pkl_scene: dict) -> dict:
    """
    Extract all LIDAR_TOP sweeps (~20 Hz) for a nuScenes scene, pairing each
    sweep with its interpolated object annotations from pkl_scene.

    Args:
        nusc:       NuScenes API handle.
        scene:      nuScenes scene record.
        pkl_scene:  { frame_token → frame_dict } for this scene (from pkl).
                    frame_token == sample_data token for LIDAR_TOP.

    Returns a scene dict in the unified JSON format.
    """
    scene_out = {
        "scene_id":   scene["token"],
        "scene_name": scene["name"],
        "dataset_id": "nuScenes_full",
        "frame_list": [],
    }

    # Start from the first LIDAR_TOP sample_data record for this scene.
    first_sample  = nusc.get("sample", scene["first_sample_token"])
    sd_token      = first_sample["data"][LIDAR_CHAN]

    prev_pts      = None
    prev_ego_pose = None
    n_missing_ann = 0
    n_keyframe_fallback = 0

    while sd_token:
        sd       = nusc.get("sample_data", sd_token)
        ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])

        # ── LiDAR point cloud & Chamfer distance ──────────────────────────
        lidar_path = nusc.get_sample_data_path(sd_token)
        curr_pts   = downsample(load_pointcloud(lidar_path), DOWNSAMPLE)
        cd         = chamfer_distance(curr_pts, prev_pts) if prev_pts is not None else 0.0

        # ── Ego velocity ──────────────────────────────────────────────────
        if prev_ego_pose is None:
            ego_vel = 0.0
        else:
            dt = (ego_pose["timestamp"] - prev_ego_pose["timestamp"]) * 1e-6
            if dt > 0:
                delta   = np.array(ego_pose["translation"]) - np.array(prev_ego_pose["translation"])
                ego_vel = float(np.linalg.norm(delta) / dt)
            else:
                ego_vel = 0.0

        # ── Object annotations from pkl ───────────────────────────────────
        # frame_token in pkl == sd_token (LIDAR_TOP sample_data token)
        pkl_frame   = pkl_scene.get(sd_token)
        object_list = []

        if pkl_frame is not None:
            for inst_token, obj in pkl_frame["objects"].items():
                x, y, z = obj["translation"]
                w, l, h = obj["size"]
                yaw     = quat_to_yaw(obj["rotation"])
                vel     = obj.get("velocity", [0.0, 0.0, 0.0])

                object_list.append({
                    "obj_id":   inst_token,
                    "label":    obj["category_name"],
                    "bbox":     [x, y, z, w, l, h, yaw],
                    "velocity": vel,
                })
        else:
            # Interpolated pkl can omit keyframes. Fall back to native keyframe annotations.
            if sd.get("is_key_frame", False) and sd.get("sample_token"):
                object_list = build_keyframe_object_list(nusc, sd["sample_token"])
                n_keyframe_fallback += 1
            else:
                # For genuinely missing non-keyframe annotations, keep empty.
                n_missing_ann += 1

        scene_out["frame_list"].append({
            "frame_id":         sd_token,
            "sample_token":     sd.get("sample_token", ""),  # keyframe this sweep belongs to
            "is_key_frame":     sd.get("is_key_frame", False),
            "chamfer_distance": cd,
            "ego_vel":          ego_vel,
            "object_list":      object_list,
        })

        prev_pts      = curr_pts
        prev_ego_pose = ego_pose
        sd_token      = sd["next"] if sd["next"] else None

    if n_keyframe_fallback:
        print(
            f"    [{scene['name']}] recovered {n_keyframe_fallback} missing keyframes "
            f"using nuScenes sample annotations"
        )
    if n_missing_ann:
        print(f"    [{scene['name']}] {n_missing_ann} non-keyframe sweeps had no annotation entry")

    return scene_out


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Load interpolated annotations
    pkl_data = load_all_pkl(PKL_DIR)
    pkl_scene_names = set(pkl_data.keys())

    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

    all_scenes = []
    skipped    = []

    for i, scene in enumerate(nusc.scene):
        name = scene["name"]
        if name not in pkl_scene_names:
            print(f"[{i+1}/{len(nusc.scene)}] Skipping '{name}' — no pkl annotations")
            skipped.append(name)
            continue

        print(f"[{i+1}/{len(nusc.scene)}] Processing '{name}' ({scene['token']})")
        scene_data = extract_scene_full(nusc, scene, pkl_data[name])
        all_scenes.append(scene_data)
        print(f"  -> {len(scene_data['frame_list'])} sweeps extracted")

    with open(OUTPUT, "w") as f:
        json.dump(all_scenes, f)

    print(f"\nSaved {len(all_scenes)} scenes to '{OUTPUT}'")
    if skipped:
        print(f"Skipped (no pkl): {skipped}")


if __name__ == "__main__":
    main()
