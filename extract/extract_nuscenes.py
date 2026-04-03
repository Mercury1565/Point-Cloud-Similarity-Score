import json
import os
import numpy as np
from scipy.spatial import cKDTree
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

# ── Configuration ─────────────────────────────────────────────────────────────
DATAROOT = "/media/tersiteab/e4d56274-b9be-4c36-8ee8-22a4e69c1bc9/home/tersiteab/Documents/PointCloudResearch/frame_similarity/nu_data"

# DATAROOT = "data/nuscenes"
VERSION  = "v1.0-mini"
OUTPUT   = "unified_nuscenes_mini.json"
LIDAR_CHAN = "LIDAR_TOP"
DOWNSAMPLE = 500
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_pointcloud(path: str) -> np.ndarray:
    """Load a nuScenes .pcd.bin file → (N, 3) xyz array."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    return pts[:, :3]


def downsample(pts: np.ndarray, n: int) -> np.ndarray:
    """Randomly downsample to at most n points."""
    if len(pts) <= n:
        return pts
    idx = rng.choice(len(pts), size=n, replace=False)
    return pts[idx]


def chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Symmetric Chamfer distance between two (N,3) point clouds."""
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a, k=1)   # each point in A → nearest in B
    d_ba, _ = tree_a.query(pts_b, k=1)   # each point in B → nearest in A
    return float(np.mean(d_ab) + np.mean(d_ba))


def quat_to_yaw(rotation: list) -> float:
    """Convert quaternion [w, x, y, z] to yaw (rotation about Z axis)."""
    q = Quaternion(rotation)
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y² + z²))
    yaw = q.yaw_pitch_roll[0]
    return float(yaw)


def get_lidar_sample_data(nusc: NuScenes, sample_token: str):
    """Return the LIDAR_TOP sample_data record for a given sample."""
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"][LIDAR_CHAN]
    return nusc.get("sample_data", sd_token)


# ── Main extraction ───────────────────────────────────────────────────────────
def extract_scene(nusc: NuScenes, scene: dict) -> dict:
    scene_out = {
        "scene_id":   scene["token"],
        "dataset_id": "nuScenes",
        "frame_list": [],
    }

    prev_pts       = None
    prev_ego_pose  = None
    sample_token   = scene["first_sample_token"]

    while sample_token:
        sample = nusc.get("sample", sample_token)

        # ── LIDAR_TOP sample data & ego_pose ──────────────────────────────
        sd = get_lidar_sample_data(nusc, sample_token)
        ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])

        # ── Chamfer distance ──────────────────────────────────────────────
        lidar_path = nusc.get_sample_data_path(sd["token"])
        curr_pts = downsample(load_pointcloud(lidar_path), DOWNSAMPLE)

        if prev_pts is None:
            cd = 0.0
        else:
            cd = chamfer_distance(curr_pts, prev_pts)

        # ── Ego velocity ──────────────────────────────────────────────────
        if prev_ego_pose is None:
            ego_vel = 0.0
        else:
            dt = (ego_pose["timestamp"] - prev_ego_pose["timestamp"]) * 1e-6  # µs → s
            if dt > 0:
                delta = np.array(ego_pose["translation"]) - np.array(prev_ego_pose["translation"])
                ego_vel = float(np.linalg.norm(delta) / dt)
            else:
                ego_vel = 0.0

        # ── Object list ───────────────────────────────────────────────────
        object_list = []
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            x, y, z = ann["translation"]
            w, l, h = ann["size"]
            yaw = quat_to_yaw(ann["rotation"])
            
            # ── Get Object Velocity ───────────────────────────────────────────
            # Using nusc.box_velocity to get [vx, vy, vz] in m/s
            vel = nusc.box_velocity(ann["token"])
            if np.any(np.isnan(vel)):
                vel = np.array([0.0, 0.0, 0.0])

            object_list.append({
                "obj_id": ann["instance_token"],
                "label":  ann["category_name"],
                "bbox":   [x, y, z, w, l, h, yaw],
                "velocity": vel.tolist(),
            })

        # ── Assemble frame ────────────────────────────────────────────────
        frame = {
            "frame_id":         sample_token,
            "chamfer_distance": cd,
            "ego_vel":          ego_vel,
            "object_list":      object_list,
        }
        scene_out["frame_list"].append(frame)

        # ── Advance ───────────────────────────────────────────────────────
        prev_pts      = curr_pts
        prev_ego_pose = ego_pose
        sample_token  = sample["next"] if sample["next"] else None

    return scene_out


def main():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    all_scenes = []
    for i, scene in enumerate(nusc.scene):
        print(f"\n[{i+1}/{len(nusc.scene)}] Processing scene: {scene['name']} ({scene['token']})")
        scene_data = extract_scene(nusc, scene)
        all_scenes.append(scene_data)
        print(f"  → {len(scene_data['frame_list'])} frames extracted")

    with open(OUTPUT, "w") as f:
        json.dump(all_scenes, f, indent=2)

    print(f"\n✅  Saved {len(all_scenes)} scenes to '{OUTPUT}'")


if __name__ == "__main__":
    main()
