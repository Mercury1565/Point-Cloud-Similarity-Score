"""Core scoring components for evaluating frame-to-frame similarity in 3D object detection datasets.
Provides IdentityMatcher (F1-score), GeometryCalculator (3D mIoU), and CompositeScorer methods.
"""

import math
from typing import Dict, List, Any, Tuple
from .types import DetectionObject, ScoreResult
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

class IdentityMatcher:
    """Handles identity matching between frames using F1-score."""

    @staticmethod
    def calculate_f1_score(
        frame_t: List[DetectionObject],
        frame_t_minus_1: List[DetectionObject]
    ) -> Tuple[float, List[Tuple[DetectionObject, DetectionObject]]]:
        """Calculate F1-score and return matched pairs (true positives)."""
        ids_t = {obj['obj_id'] for obj in frame_t}
        ids_t_minus_1 = {obj['obj_id'] for obj in frame_t_minus_1}

        tp_ids = ids_t.intersection(ids_t_minus_1)
        tp = len(tp_ids)
        fp = len(ids_t - ids_t_minus_1)
        fn = len(ids_t_minus_1 - ids_t)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        tp_pairs = []
        for obj_t in frame_t:
            if obj_t['obj_id'] in tp_ids:
                obj_t_minus_1 = next(obj for obj in frame_t_minus_1 if obj['obj_id'] == obj_t['obj_id'])
                tp_pairs.append((obj_t, obj_t_minus_1))

        return f1_score, tp_pairs

    # vx, vy
    # validation process
    # torch 3d


class GeometryCalculator:
    """Calculates 3D Intersection over Union (IoU) for matched object pairs."""

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def calculate_miou(self, tp_pairs: List[Tuple[DetectionObject, DetectionObject]]) -> float:
        """Calculate mean 3D IoU across all matched pairs."""
        if not tp_pairs:
            return 0.0
        total_iou = sum(self.calculate_3d_iou(pair[0]['bbox'], pair[1]['bbox']) for pair in tp_pairs)
        return total_iou / len(tp_pairs)

    def calculate_3d_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate 3D IoU for a single pair of bounding boxes."""
        x1, y1, z1, w1, l1, h1, yaw1 = bbox1
        x2, y2, z2, w2, l2, h2, yaw2 = bbox2

        bev_overlap = self._calculate_bev_overlap(x1, y1, w1, l1, yaw1, x2, y2, w2, l2, yaw2)
        height_overlap = self._calculate_height_overlap(z1, h1, z2, h2)

        intersection_volume = bev_overlap * height_overlap
        volume1 = w1 * l1 * h1
        volume2 = w2 * l2 * h2
        union_volume = volume1 + volume2 - intersection_volume

        if union_volume > self.epsilon:
            return intersection_volume / union_volume
        return 0.0

    def _calculate_bev_overlap(self, x1, y1, w1, l1, yaw1, x2, y2, w2, l2, yaw2) -> float:
        """Calculate Bird's Eye View rotated rectangle overlap using shapely."""

        rect1_corners = [(-w1/2, -l1/2), (w1/2, -l1/2), (w1/2, l1/2), (-w1/2, l1/2)]
        poly1 = Polygon(rect1_corners)
        poly1 = rotate(poly1, yaw1, origin=(0, 0), use_radians=True)
        poly1 = translate(poly1, xoff=x1, yoff=y1)

        rect2_corners = [(-w2/2, -l2/2), (w2/2, -l2/2), (w2/2, l2/2), (-w2/2, l2/2)]
        poly2 = Polygon(rect2_corners)
        poly2 = rotate(poly2, yaw2, origin=(0, 0), use_radians=True)
        poly2 = translate(poly2, xoff=x2, yoff=y2)

        return poly1.intersection(poly2).area

    def _calculate_height_overlap(self, z1: float, h1: float, z2: float, h2: float) -> float:
        """Calculate 1D overlap along the z-axis."""
        top1 = z1 + h1
        top2 = z2 + h2
        overlap_bottom = max(z1, z2)
        overlap_top = min(top1, top2)
        return max(0.0, overlap_top - overlap_bottom)

class CompositeScorer:
    """Combines F1-score and mIoU into a single confidence score."""

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def harmonic_mean(self, f1: float, miou: float, w_f1: float, w_miou: float) -> float:
        """Calculate weighted harmonic mean."""
        f1_safe = f1 + self.epsilon
        miou_safe = miou + self.epsilon
        term1 = w_f1 / f1_safe
        term2 = w_miou / miou_safe
        result = 1.0 / (term1 + term2)
        return min(1.0, max(0.0, result))

    def arithmetic_mean(self, f1: float, miou: float, w_f1: float, w_miou: float) -> float:
        """Calculate weighted arithmetic mean."""
        result = w_f1 * f1 + w_miou * miou
        return min(1.0, max(0.0, result))

    def min_threshold(self, f1: float, miou: float) -> float:
        """Return the minimum of the two metrics."""
        return min(f1, miou)

class ConfidenceScorer:
    """Main orchestrator for evaluating frame-to-frame similarity."""

    def __init__(self, w_f1: float = 0.5, w_miou: float = 0.5, epsilon: float = 1e-6):
        self.w_f1 = w_f1
        self.w_miou = w_miou
        self.epsilon = epsilon
        self.geometry_calc = GeometryCalculator(epsilon)
        self.composite_scorer = CompositeScorer(epsilon)

    def calculate_score(
        self,
        frame_t: List[DetectionObject],
        frame_t_minus_1: List[DetectionObject]
    ) -> ScoreResult:
        """Calculate the composite confidence score between two frames."""
        from .validation import validate_frame

        validate_frame(frame_t)
        validate_frame(frame_t_minus_1)

        if not frame_t and not frame_t_minus_1:
            return {"confidence_score": 0.0, "f1": 0.0, "miou": 0.0}

        f1, tp_pairs = IdentityMatcher.calculate_f1_score(frame_t, frame_t_minus_1)

        miou = self.geometry_calc.calculate_miou(tp_pairs) if tp_pairs else 0.0
        conf = self.calculate_composite_score(f1, miou, method="harmonic")

        return {
            "confidence_score": conf,
            "f1": f1,
            "miou": miou
        }

    def calculate_composite_score(self, f1: float, miou: float, method: str = "harmonic", **kwargs) -> float:
        """Calculate composite score using the specified method (harmonic, arithmetic, min_threshold)."""
        if f1 == 0.0 and miou == 0.0:
            return 0.0
        if method == "harmonic":
            return self.composite_scorer.harmonic_mean(f1, miou, self.w_f1, self.w_miou)
        elif method == "arithmetic":
            return self.composite_scorer.arithmetic_mean(f1, miou, self.w_f1, self.w_miou)
        elif method == "min_threshold":
            return self.composite_scorer.min_threshold(f1, miou)
        raise ValueError(f"Unknown method {method}. Valid methods: harmonic, arithmetic, min_threshold")
