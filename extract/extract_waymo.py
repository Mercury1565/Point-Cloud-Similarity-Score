"""
Waymo Open Dataset extractor → unified_waymo.json

Data layout expected at DATAROOT
---------------------------------
<DATAROOT>/
    raw_data/
        segment-<hash>_with_camera_labels.tfrecord   ← annotations + poses
    waymo_processed_data_v0_5_0/
        segment-<hash>_with_camera_labels/
            0000.npy, 0001.npy, ...                  ← pre-processed point clouds (generated
                                                        by M3DETR or by this script)
                                                        shape (N,6): x,y,z,intensity,elongation,NLZ

If a segment's npy directory does not exist, this script generates the npy files
directly from the .tfrecord using a pure-Python / NumPy range-image decoder.
TensorFlow is NOT required.

Usage:
    python extract/extract_waymo.py
"""

import json
import os
import struct
import zlib
import glob
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from pyquaternion import Quaternion  # kept for API parity; not used in this file

try:
    from waymo_open_dataset import dataset_pb2
    _WAYMO_AVAILABLE = True
except ImportError:
    _WAYMO_AVAILABLE = False

# ── Configuration ──────────────────────────────────────────────────────────────
DATAROOT      = "/media/tersiteab/e4d56274-b9be-4c36-8ee8-22a4e69c1bc9/home/tersiteab/Documents/ViT/detection/M3DETR/data/waymo"
RAW_DATA_DIR  = os.path.join(DATAROOT, "raw_data")
PROCESSED_DIR = os.path.join(DATAROOT, "waymo_processed_data_v0_5_0")
OUTPUT        = "unified_waymo.json"
DOWNSAMPLE    = 500
RANDOM_SEED   = 42

WAYMO_CLASSES = ["unknown", "Vehicle", "Pedestrian", "Sign", "Cyclist"]

rng = np.random.default_rng(RANDOM_SEED)


# ── Pure-Python TFRecord reader ────────────────────────────────────────────────
def _read_tfrecord(path: str):
    """
    Yield raw byte strings for each record in a .tfrecord file without importing
    TensorFlow.  TFRecord format: [length:uint64][crc32:uint32][data][crc32:uint32]
    """
    with open(path, "rb") as f:
        while True:
            header = f.read(12)            # 8-byte length + 4-byte masked CRC
            if not header:
                break
            if len(header) < 12:
                raise IOError(f"Truncated TFRecord header in {path}")
            length = struct.unpack("<Q", header[:8])[0]
            data   = f.read(length)
            f.read(4)                      # masked CRC of data — skip
            yield data


# ── Range image → point cloud (pure numpy, no TF) ─────────────────────────────
def _decompress_range_image(compressed: bytes) -> np.ndarray:
    """Decompress a zlib-compressed Waymo MatrixFloat range image → (H,W,4) float32."""
    ri_proto = dataset_pb2.MatrixFloat()
    ri_proto.ParseFromString(zlib.decompress(compressed))
    dims = list(ri_proto.shape.dims)   # [H, W, 4]
    return np.array(ri_proto.data, dtype=np.float32).reshape(dims)


def _range_image_to_points(ri: np.ndarray, inclinations: np.ndarray,
                            extrinsic: np.ndarray) -> np.ndarray:
    """
    Convert a (H,W,4) range image to a (N,6) point cloud in the vehicle frame.

    Channels: [range, intensity, elongation, NLZ_flag]

    Args:
        ri:           Range image array (H, W, 4).
        inclinations: Per-beam elevation angles (H,), ordered so that row 0
                      corresponds to inclinations[0].  Pass the calibration
                      beam_inclinations *reversed* so row-0 = top beam.
        extrinsic:    4×4 lidar-to-vehicle transform.

    Returns:
        (N, 6) array: [x, y, z, intensity, elongation, NLZ] in vehicle frame.
        Returns empty (0,6) array if no valid points.
    """
    H, W = ri.shape[:2]

    # Azimuth angles: column 0 → ≈+π, column W-1 → ≈-π  (left-hand sweep)
    azimuths = np.pi - (np.arange(W, dtype=np.float32) + 0.5) / W * 2.0 * np.pi

    el_grid, az_grid = np.meshgrid(inclinations, azimuths, indexing="ij")  # (H, W)

    ranges = ri[:, :, 0]
    valid  = ranges > 0.0
    if not valid.any():
        return np.zeros((0, 6), dtype=np.float32)

    r   = ranges[valid]
    el  = el_grid[valid]
    az  = az_grid[valid]

    cos_el = np.cos(el)
    x_local = r * cos_el * np.cos(az)
    y_local = r * cos_el * np.sin(az)
    z_local = r * np.sin(el)

    # Transform to vehicle frame
    pts_h = np.column_stack([x_local, y_local, z_local,
                              np.ones(len(x_local), dtype=np.float32)])
    pts_vehicle = (extrinsic @ pts_h.T).T[:, :3].astype(np.float32)

    intensity   = ri[:, :, 1][valid].reshape(-1, 1)
    elongation  = ri[:, :, 2][valid].reshape(-1, 1)
    nlz         = ri[:, :, 3][valid].reshape(-1, 1)

    return np.concatenate([pts_vehicle, intensity, elongation, nlz], axis=1)


def _frame_to_pointcloud(frame) -> np.ndarray:
    """
    Convert a Waymo Frame proto to a (N,6) point cloud using all 5 lidars,
    first return only.  No TF required.

    Returns (N,6): [x, y, z, intensity, elongation, NLZ] in vehicle frame.
    """
    # Build calibration lookup: name → (inclinations, extrinsic)
    cal_lookup = {}
    for cal in frame.context.laser_calibrations:
        extrinsic = np.array(cal.extrinsic.transform, dtype=np.float64).reshape(4, 4)
        if len(cal.beam_inclinations) > 0:
            # Reverse so row-0 in the range image = highest-elevation beam
            inclinations = np.array(cal.beam_inclinations, dtype=np.float32)[::-1].copy()
        else:
            # Side lidars: uniform distribution between min and max, then reverse
            H_default = 200  # placeholder; corrected below after ri parse
            inclinations = None  # filled after we know ri height
        cal_lookup[cal.name] = (cal, extrinsic, inclinations)

    all_points = []
    for laser in frame.lasers:
        if not laser.ri_return1.range_image_compressed:
            continue
        try:
            ri = _decompress_range_image(laser.ri_return1.range_image_compressed)
        except Exception:
            continue

        cal_entry = cal_lookup.get(laser.name)
        if cal_entry is None:
            continue
        cal, extrinsic, inclinations = cal_entry

        if inclinations is None:
            # Uniform inclinations for side lidars
            H = ri.shape[0]
            inclinations = np.linspace(
                cal.beam_inclination_min, cal.beam_inclination_max, H,
                dtype=np.float32
            )[::-1].copy()

        pts = _range_image_to_points(ri, inclinations, extrinsic)
        if pts.shape[0] > 0:
            all_points.append(pts)

    if not all_points:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(all_points, axis=0)


# ── Generate npy files for a segment that has no pre-processed dir ────────────
def generate_npy_files(tfrecord_path: str, npy_dir: str) -> int:
    """
    Decode every frame in tfrecord_path and save <npy_dir>/<frame_cnt:04d>.npy.

    Returns the number of frames written.
    """
    os.makedirs(npy_dir, exist_ok=True)
    written = 0
    for frame_cnt, raw_bytes in enumerate(_read_tfrecord(tfrecord_path)):
        npy_path = os.path.join(npy_dir, f"{frame_cnt:04d}.npy")
        if os.path.exists(npy_path):
            written += 1
            continue
        frame = dataset_pb2.Frame()
        frame.ParseFromString(raw_bytes)
        pts = _frame_to_pointcloud(frame)
        np.save(npy_path, pts)
        written += 1
    return written


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_pointcloud(npy_path: str) -> np.ndarray:
    """Load pre-processed Waymo .npy → (N,3) xyz array (first 3 columns)."""
    pts = np.load(npy_path)               # (N,6): x,y,z,intensity,elongation,NLZ
    return pts[:, :3].astype(np.float32)


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


def pose_to_translation(pose_flat) -> np.ndarray:
    """Extract translation vector from a flat 16-element 4×4 pose matrix."""
    return np.array(pose_flat, dtype=np.float64).reshape(4, 4)[:3, 3]


# ── Per-segment extraction ─────────────────────────────────────────────────────
def extract_segment(tfrecord_path: str, npy_dir: str) -> dict:
    """
    Process a single Waymo segment.

    If npy_dir does not exist or is incomplete, npy files are generated on-the-fly
    from the tfrecord before extraction proceeds.

    Returns a scene dict in the unified JSON format.
    """
    segment_name = os.path.splitext(os.path.basename(tfrecord_path))[0]

    # Generate npy files for any frames not yet on disk (handles missing dir
    # and partially-processed segments alike).
    existing = len(glob.glob(os.path.join(npy_dir, "*.npy"))) if os.path.isdir(npy_dir) else 0
    # Count frames in tfrecord to compare
    total_frames = sum(1 for _ in _read_tfrecord(tfrecord_path))
    if existing < total_frames:
        print(f"    {existing}/{total_frames} npy files present — generating missing point clouds ...")
        n = generate_npy_files(tfrecord_path, npy_dir)
        print(f"    → {n} npy files ready in {npy_dir}")

    scene_out = {
        "scene_id":   segment_name,
        "dataset_id": "waymo",
        "frame_list": [],
    }

    prev_pts       = None
    prev_pose_t    = None
    prev_timestamp = None

    for frame_cnt, raw_bytes in enumerate(_read_tfrecord(tfrecord_path)):
        # ── Load proto ────────────────────────────────────────────────────────
        frame = dataset_pb2.Frame()
        frame.ParseFromString(raw_bytes)

        # ── Point cloud from pre-processed .npy ──────────────────────────────
        npy_path = os.path.join(npy_dir, f"{frame_cnt:04d}.npy")
        if not os.path.exists(npy_path):
            continue

        curr_pts = downsample(load_pointcloud(npy_path), DOWNSAMPLE)

        # ── Chamfer distance ──────────────────────────────────────────────────
        cd = chamfer_distance(curr_pts, prev_pts) if prev_pts is not None else 0.0

        # ── Ego velocity from consecutive poses ───────────────────────────────
        pose_t    = pose_to_translation(frame.pose.transform)
        timestamp = frame.timestamp_micros         # microseconds

        if prev_pose_t is not None and prev_timestamp is not None:
            dt = (timestamp - prev_timestamp) * 1e-6   # µs → s
            ego_vel = float(np.linalg.norm(pose_t - prev_pose_t) / dt) if dt > 0 else 0.0
        else:
            ego_vel = 0.0

        # ── Object annotations from laser_labels ──────────────────────────────
        object_list = []
        for label in frame.laser_labels:
            box      = label.box
            class_id = label.type
            category = WAYMO_CLASSES[class_id] if class_id < len(WAYMO_CLASSES) else "unknown"

            x, y, z = box.center_x, box.center_y, box.center_z
            w, l, h = box.width, box.length, box.height
            yaw     = box.heading

            vx = getattr(label.metadata, "speed_x", 0.0)
            vy = getattr(label.metadata, "speed_y", 0.0)

            object_list.append({
                "obj_id":   label.id,
                "label":    category,
                "bbox":     [x, y, z, w, l, h, yaw],
                "velocity": [vx, vy, 0.0],
            })

        scene_out["frame_list"].append({
            "frame_id":         frame.context.name + f"_{frame_cnt:04d}",
            "chamfer_distance": cd,
            "ego_vel":          ego_vel,
            "object_list":      object_list,
        })

        prev_pts       = curr_pts
        prev_pose_t    = pose_t
        prev_timestamp = timestamp

    return scene_out


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if not _WAYMO_AVAILABLE:
        raise ImportError(
            "waymo_open_dataset is not installed.\n"
            "Run: pip install waymo-open-dataset-tf-2-12-0"
        )

    tfrecord_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.tfrecord")))
    if not tfrecord_files:
        raise FileNotFoundError(f"No .tfrecord files found in: {RAW_DATA_DIR}")

    print(f"Found {len(tfrecord_files)} segment(s) in {RAW_DATA_DIR}")

    all_scenes = []
    for i, tf_path in enumerate(tfrecord_files):
        seg_name = os.path.splitext(os.path.basename(tf_path))[0]
        npy_dir  = os.path.join(PROCESSED_DIR, seg_name)

        print(f"  [{i+1}/{len(tfrecord_files)}] Processing '{seg_name}'")
        scene = extract_segment(tf_path, npy_dir)
        all_scenes.append(scene)
        n_obj = sum(len(f["object_list"]) for f in scene["frame_list"])
        print(f"    → {len(scene['frame_list'])} frames, {n_obj} total objects")

    with open(OUTPUT, "w") as f:
        json.dump(all_scenes, f)

    print(f"\nSaved {len(all_scenes)} scenes to '{OUTPUT}'")


if __name__ == "__main__":
    main()
