"""Core data types for the Online Perception Engine."""

from dataclasses import dataclass
from typing import List


@dataclass
class FrameDecision:
    """Represents a single frame's inference decision.
    
    Attributes:
        frame_idx: Index of the frame in the sequence
        predicted_confidence: Model's predicted confidence score (clipped to [0.0, 1.0])
        actual_confidence: Ground-truth confidence score
        decision: Either "REUSE" or "FULL_DETECTION"
        prediction_error: Absolute difference between predicted and actual confidence
        is_audit_frame: Whether this frame triggered a model update
    """
    frame_idx: int
    predicted_confidence: float
    actual_confidence: float
    decision: str
    prediction_error: float
    is_audit_frame: bool


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics for the online learning system.
    
    Attributes:
        total_frames: Total number of frames processed
        reuse_count: Number of REUSE decisions made
        full_detection_count: Number of FULL_DETECTION decisions made
        cumulative_latency_saved_ms: Total latency saved by REUSE decisions
        mean_absolute_errors: List of prediction errors from audit frames
        safety_violations: Count of REUSE decisions with actual confidence below safety threshold
        audit_frames: List of frame indices that were audit frames
    """
    total_frames: int
    reuse_count: int
    full_detection_count: int
    cumulative_latency_saved_ms: float
    mean_absolute_errors: List[float]
    safety_violations: int
    audit_frames: List[int]
