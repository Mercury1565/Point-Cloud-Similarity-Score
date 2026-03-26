## Unified Dataset Schema (`.json`)

The `unified_nuscenes_mini.json` acts as a standardized middle-layer between raw LiDAR datasets and the Machine Learning pipeline. It organizes spatial and temporal data into a hierarchical "Scene-Frame-Object" structure.

### 1. Root Structure
The root is a **list of Scene objects**, representing continuous driving sequences.

| Key | Type | Description |
| :--- | :--- | :--- |
| `scene_id` | `string` | Unique token identifying the specific driving sequence. |
| `dataset_id` | `string` | The source dataset |
| `frame_list` | `list` | A chronological sequence of frames within the scene. |

### 2. Frame Object
Each entry in the `frame_list` represents a single snapshot in time (typically at 2Hz for nuScenes).

| Key | Type | Description |
| :--- | :--- | :--- |
| `frame_id` | `string` | Unique timestamp or sample token for the frame. |
| `chamfer_distance`| `float` | Symmetric distance between point cloud at $t$ and $t-1$. Measures raw sensor change. |
| `ego_vel` | `float` | The scalar velocity ($m/s$) of the sensor-carrying vehicle. |
| `object_list` | `list` | All annotated/detected 3D objects present in this frame. |

### 3. Object Entry
Contained within the `object_list`, these represent the "Ground Truth" perception state.

* **`obj_id` (`string`)**: The unique Instance Token. **Crucial for temporal tracking**; if the same car appears in ten frames, it must have the same `obj_id` in all ten.
* **`label` (`string`)**: The class category (e.g., `vehicle.car`, `human.pedestrian`).
* **`bbox` (`list`)**: A 7-element geometric vector:
    * `[0:3]`: **Center Position** ($x, y, z$) in global coordinates.
    * `[3:6]`: **Dimensions** ($w, l, h$) in meters.
    * `[6]`:   **Yaw** (Rotation) in radians about the Z-axis.