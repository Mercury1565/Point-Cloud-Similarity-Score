# Features List

| Feature          | Definition                                                                 |
|------------------|----------------------------------------------------------------------------|
| chamfer_dist     | The symmetric distance between two point clouds (the current frame t and the previous frame t−1). |
| ego_vel          | The scalar velocity (m/s) of the sensor-carrying vehicle at the current timestamp. |
| delta_ego_vel    | The change in velocity since the last frame (velₜ − velₜ₋₁). |
| ego_accel        | The calculated acceleration (m/s²) over the time interval (usually Δvel/0.5s). |
| obj_count        | Total number of unique 3D bounding boxes annotated in the current frame. |
| delta_obj_count  | The difference in object count between frames (countₜ − countₜ₋₁). |
| avg_dist         | The mean Euclidean distance of all detected objects relative to the ego-vehicle. |

# Target

`target_confidence` serves as the "Ground Truth" label. It represents a mathematical measurement of how much the 3D scene has changed between two frames. If the score is high (close to 1.0), the detections from the previous frame are still accurate enough to be reused.

The `target_confidence` value is calculated through a three-stage process:

### 1. Identity Matching (The F1-Score)

The IdentityMatcher compares object identities between the current frame (t) and the previous frame (t−1).

- **True Positives (TP):** Objects present in both frames (matched by ID)  
- **False Positives (FP):** New objects entering the scene  
- **False Negatives (FN):** Objects that disappeared  

**Calculation:**  
Computes the F1-score (balance of Precision and Recall) to measure identity stability.

### 2. Geometric Consistency (The 3D mIoU): Mean Intersection Over Union

For matched objects (TPs), the GeometryCalculator checks spatial consistency.

- **3D IoU:** 
    When the `IdentityMatcher` finds a match (e.g., **Car #42** exists in both frames), the `GeometryCalculator` zooms in on that specific car to see how much it moved. 

    **IoU (Intersection over Union)** is a score from **0 to 1** that measures overlap:
    * **1.0:** The car hasn't moved a millimeter. The old box and the new box are perfectly stacked.
    * **0.5:** The car moved halfway out of its previous spot.
    * **0.0:** The car moved so fast it’s completely outside its previous box.

    Since 3D boxes are complex, the code breaks it into two simpler problems:
    1.  **The "Footprint" (BEV Overlap):** It looks down from the sky (Bird's Eye View). It uses the **Shapely** library to draw two rotated rectangles (using $x, y, w, l,$ and $yaw$). It calculates the area where these two rectangles overlap.
    2.  **The "Height" (Z-overlap):** it looks from the side. If one box is on the ground and the other is floating in the air, the overlap is zero. 
    3.  **The Volume:** It multiplies the **Overlap Area** by the **Overlap Height**. 

- **Mean IoU (mIoU):**  
    The **3D IoU** only tells about **one car**. But a single scene might have 50 cars, 20 pedestrians, and 5 buses.

    The **mIoU (Mean IoU)** is the average of every single matched object's IoU score. 

### 3. Composite Score (Harmonic Mean)

The CompositeScorer combines F1-score and mIoU into a single `target_confidence`.

**Formula:**
```
Score = 1 / ( (w_F1 / F1) + (w_mIoU / mIoU) )
```
- Default weights: `w_F1 = 0.5`, `w_mIoU = 0.5`

**Why Harmonic Mean?**
- Penalizes low values strongly  
- Ensures both identity and geometry must be good  