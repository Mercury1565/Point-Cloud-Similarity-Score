
# Data Audit + Confidence Visualization (nuScenes mini)

This note documents what `audit_nuscenes.py` does and how to interpret its outputs.

## What the script consumes

`audit_nuscenes.py` loads the extracted Stage-1 dataset:

- **Input file:** `unified_nuscenes_mini.json`
- **Top-level structure:** a list of scenes
- **Per scene:**
	- `scene_id`
	- `frame_list`: ordered frames over time
- **Per frame (as used by the audit):**
	- `frame_id`
	- `object_list`: list of detected/annotated objects
	- `ego_vel`: ego vehicle velocity (m/s)
	- `chamfer_distance`: a per-frame-to-frame geometric change metric (stored in the JSON)

It also instantiates a `ConfidenceScorer` from `confidence_scorer/scorer.py`.

## What it computes (audit pass)

The script iterates scene-by-scene and frame-by-frame and collects:

1. **Dataset density stats**
	 - `total_scenes`: number of scenes in the JSON
	 - `total_frames`: total frame count across all scenes
	 - `total_objects`: total objects across all frames
	 - `avg_objs`: average objects per frame (`total_objects / total_frames`)

2. **Dead frame detection**
	 - A **dead frame** is any frame where `object_list` is empty.
	 - Dead frames are appended as `(scene_id, frame_id)` to a list for reporting.
	 - This is a basic sanity check: if dead frames exist, later temporal scoring and model training will have “no-object” steps that may skew learning.

3. **Ego motion range**
	 - Tracks `min_ego_vel` and `max_ego_vel` across all frames.
	 - This helps confirm extraction plausibility (e.g., urban driving shouldn’t show highway-scale velocities).

4. **Per-frame temporal confidence score**
	 - For each frame $t$, it computes a **confidence score** using the `ConfidenceScorer` between the current frame’s objects and the previous frame’s objects:
		 - For $t=0$, it compares the current `object_list` vs an **empty** previous list (`[]`).
		 - For $t>0$, it compares `frame_list[t].object_list` vs `frame_list[t-1].object_list`.
	 - It stores the returned `confidence_score` (the script labels it as “Confidence Score (Harmonic)” in the plot).

5. **Per-frame Chamfer distance (as provided by the dataset)**
	 - The script records `frame["chamfer_distance"]` into a list per-scene.
	 - For plotting, it normalizes each scene’s Chamfer series to $[0, 1]$ (min-max normalization per scene) purely for **visual comparability** with confidence.

## What it prints (audit log)

After processing all scenes it prints a **summary report**:

- Scenes processed, frames processed, average objects/frame
- Drift check: min/max ego velocity (m/s)
- Dead frames check: either “No dead frames found” or a short list of examples

This output is meant to quickly validate:

- The extraction produced enough objects per frame to train on.
- The motion range is plausible.
- There are no “empty” frames that could break assumptions of downstream temporal algorithms.

## What it produces (visualization)

The script generates a multi-panel time series figure:

- **Output file:** `nuscenes_similarity_trends.png`
- **Layout:** up to 10 scenes (5 rows × 2 columns)
- **Per scene subplot:**
	- Blue line: confidence score over frame index
	- Red dashed line: *normalized* Chamfer distance over frame index

The intent is to visually validate that temporal confidence behaves sensibly relative to a geometric change proxy.

## How to interpret the results (what “good” looks like)

The audit log and the visualization confirm that Stage 1 data extraction is high-quality and logically sound. It has a dense, active dataset with roughly 46 objects per frame and a realistic ego-velocity range (0–54 km/h), typical for urban Singapore/Boston scenes in nuScenes.

Here is a breakdown of what the results tell us:

### 1) The audit log: sanity confirmed

- **Zero dead frames:** This is excellent. It means every frame has at least one object to track, ensuring the Random Forest will always have data to process during training.
- **Velocity range:** A max of ~15.16 m/s is very reasonable for city driving. It provides enough motion to challenge the similarity model without being so fast that the LiDAR data becomes overly degraded by motion.

### 2) The visualization: key trends

The plots show the relationship between the **confidence score** (blue) and the **normalized Chamfer distance** (red dashed).

- **Initial “jump” at frame 0:** In every scene, it will typically see a spike at $t=0$. This is expected, because the script computes the first confidence score by comparing frame 0 against an *empty* previous-frame list (a deliberate handling for the missing $t-1$).

- **Inverse correlation (the “correct” behavior):** In many scenes, when the red line spikes (larger geometric difference), the blue line dips (lower temporal consistency).
	- **Why this matters:** it indicates the features are physically meaningful. When raw point clouds change drastically (high Chamfer), detection/association consistency tends to drop. That’s exactly the kind of signal a downstream similarity model (e.g., a Random Forest) can learn.

- **High confidence scenes:** Some scenes show steady blue curves near the top of the range, suggesting stable motion (e.g., straight driving) and high temporal redundancy—good candidates for **inference reuse**.

- **Highly dynamic scenes:** Some scenes show large confidence swings, which likely correspond to turns, intersections, occlusions, or dense traffic. In these cases, a reuse policy should more often decide “do inference” rather than reuse stale results.

### 3) Structural observations

- **The “gap” between curves:** In some scenes, it may see Chamfer moving a lot while confidence remains relatively stable. This usually isn’t a bug—it can mean the raw point cloud is changing due to background structure (vegetation, poles, map elements) while the tracked object boxes remain stable.

- **Stability of annotations vs raw points:** The blue confidence line is often smoother than the red Chamfer line. That suggests annotations/boxes are temporally stable even when LiDAR points are noisy.

## Notes / limitations

- Chamfer is normalized **per scene** for plotting only, so absolute Chamfer values **across different scenes** are not directly comparable in the figure.
- The frame-0 confidence score is computed against an empty previous list by design; interpret that first point as a boundary condition rather than a true temporal comparison.

