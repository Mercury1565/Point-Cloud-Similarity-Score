"""Type definitions for the ConfidenceScorer package."""

from typing import TypedDict, List


class DetectionObject(TypedDict):
    """Represents a detected object in a frame."""
    obj_id: str
    label: str
    bbox: List[float]


class ScoreResult(TypedDict):
    """Result of confidence score calculation."""
    confidence_score: float
    f1: float
    miou: float


class BoundingBox3D(TypedDict):
    """3D bounding box representation."""
    x: float
    y: float
    z: float
    w: float
    l: float
    h: float
    yaw: float
