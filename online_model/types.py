from dataclasses import dataclass
from typing import List


@dataclass
class FrameDecision:
    frame_idx: int
    predicted_confidence: float
    actual_confidence: float
    decision: str
    prediction_error: float
    is_audit_frame: bool


@dataclass
class PerformanceMetrics:
    total_frames: int
    reuse_count: int
    full_detection_count: int
    cumulative_latency_saved_ms: float
    mean_absolute_errors: List[float]
    safety_violations: int
    audit_frames: List[int]
