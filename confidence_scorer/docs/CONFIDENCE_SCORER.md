# Confidence Scorer

The `ConfidenceScorer` package evaluates the frame-to-frame similarity of 3D objects detected in autonomous driving datasets (like nuScenes or KITTI). The core logic is implemented in `confidence_scorer/scorer.py`.

This document explains the inner math and programming steps used to evaluate two contiguous frames (Frame $t$ and Frame $t-1$).

---

## 1. High-Level Flow

When `ConfidenceScorer.calculate_score(frame_t, frame_t_minus_1)` is called, the pipeline executes in three stages:

1. **Identity Matching**: Do the object IDs track continuously between frames? (Yields an $F_1$ score).
2. **Geometric Similarity**: For the objects that *do* match, how much did their bounding boxes shift or deform? (Yields a $mIoU$ score).
3. **Composite Scoring**: Combine the identity and geometric metrics to output a final `confidence_score`.

---

## 2. Component Details

### A. Identity Matching (`IdentityMatcher`)

The `IdentityMatcher` class deals purely with the `obj_id` fields, treating the persistence of these IDs across frames as a binary classification problem.

1. **True Positives (TP)**: IDs that exist in both $t$ and $t-1$.
2. **False Positives (FP)**: IDs that exist only in $t$ (newly appeared or hallucinations).
3. **False Negatives (FN)**: IDs that exist only in $t-1$ (missed detections or objects that left the scene).

With these counts, it computes:
- $\text{Precision} = \frac{TP}{TP + FP}$
- $\text{Recall} = \frac{TP}{TP + FN}$
- $F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

**Output**: It returns the $F_1$ score and a list of `tp_pairs`—the matched pairs of bounding boxes that successfully persisted across both frames.

### B. Geometric Similarity (`GeometryCalculator`)

For every matched pair (every True Positive), the `GeometryCalculator` measures the exact 3D spatial drift between the bounding box at time $t$ and time $t-1$ using **Intersection over Union (IoU)**.

This is tricky for 3D bounding boxes because cars rotate (yaw), and they are not completely axis-aligned. The codebase solves this by splitting the 3D volume calculation into two independent overlaps:

1. **Bird's Eye View (BEV) Overlap (2D)**:
   - Takes the $x, y$ coordinates, $width, length$, and the $yaw$ angle.
   - Leverages the external `shapely` library. It draws a 2D rectangle at the origin, rotates it by $yaw$, translates it to $x, y$, and then calculates the exact intersecting 2D planar area of the two rotated polygons (`poly1.intersection(poly2).area`).

2. **Height Overlap (1D)**:
   - Evaluates the $z$ axis and the box $height$.
   - Calculates the simple 1D interval overlap between $[z_1, z_1 + h_1]$ and $[z_2, z_2 + h_2]$.

**3D IoU Integration**:
- $\text{Intersection Volume} = \text{BEV Area} \times \text{Height Overlap}$
- $\text{Union Volume} = \text{Volume}_1 + \text{Volume}_2 - \text{Intersection Volume}$
- $\text{IoU} = \frac{\text{Intersection Volume}}{\text{Union Volume}}$

The final $mIoU$ (mean IoU) is the average IoU across *all* matching pairs.

### C. Composite Scoring (`CompositeScorer`)

We now have two metrics:
- An $F_1$ score measuring **Tracking Continuity**.
- A $mIoU$ score measuring **Spatial Stability**.

The `CompositeScorer` merges them into a single metric. Three aggregation methods are provided:

1. **Harmonic Mean (Default)**:
   $$ \text{Score} = \frac{1}{\frac{w_{f1}}{F_1} + \frac{w_{miou}}{mIoU}} $$
   *Why Harmonic Mean?* It heavily penalises large imbalances. If a tracker holds IDs perfectly ($F_1 = 1.0$) but the boxes jitter wildly into incorrect shapes ($mIoU = 0.1$), the Harmonic Mean drags the score drastically down, immediately highlighting the failure.

2. **Arithmetic Mean**:
   $$ \text{Score} = (w_{f1} \times F_1) + (w_{miou} \times mIoU) $$
   A simpler, more forgiving weighted average. Be careful, as a catastrophic failure in one metric might be masked by perfection in the other.

3. **Min Threshold (Conservative)**:
   $$ \text{Score} = \min(F_1, mIoU) $$
   Yields the strict minimum, ensuring the pipeline's confidence is only as high as its weakest link.

*(Note: In all division operations, a tiny $\epsilon$ (e.g., $1e-6$) is added to the denominators to prevent `ZeroDivisionError` floating point crashes when arrays are empty).*

---

## Conclusion
Ultimately, `ConfidenceScorer` operates by stripping the 3D complex autonomous scene into isolated bounding-box intersections and ID set operations. By penalising tracking disconnections and physical jitter uniformly, it returns an objective score mapping between $0.0$ and $1.0$ representing frame-to-frame model consistency.
