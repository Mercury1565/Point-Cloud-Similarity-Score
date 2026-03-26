from confidence_scorer.scorer import (
    ConfidenceScorer,
    IdentityMatcher,
    GeometryCalculator,
    CompositeScorer,
)

# Import types
from confidence_scorer.types import DetectionObject, ScoreResult, BoundingBox3D

# Import validation functions
from confidence_scorer.validation import validate_frame, validate_bbox

__all__ = [
    # Primary interface
    "ConfidenceScorer",
    # Core components
    "IdentityMatcher",
    "GeometryCalculator",
    "CompositeScorer",
    # Types
    "DetectionObject",
    "ScoreResult",
    "BoundingBox3D",
    # Validation
    "validate_frame",
    "validate_bbox",
]
