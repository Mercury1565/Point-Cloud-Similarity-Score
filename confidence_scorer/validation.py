"""Validation utilities for input data."""

from typing import List, Dict, Any
import math


def validate_bbox(bbox: Any) -> None:
    """Validate bounding box format and dimensions."""
    if not isinstance(bbox, list):
        raise ValueError("bbox must be a list of 7 floats")
    
    if len(bbox) != 7:
        raise ValueError("bbox must be a list of 7 floats")
    
    for i, val in enumerate(bbox):
        if not isinstance(val, (int, float)):
            raise ValueError("bbox must be a list of 7 floats")
        if not math.isfinite(val):
            raise ValueError(f"bbox element at index {i} must be a finite number")
    
    w, l, h = bbox[3], bbox[4], bbox[5]
    if w <= 0 or l <= 0 or h <= 0:
        raise ValueError("bbox dimensions (w, l, h) must be positive")


def validate_frame(frame: Any) -> None:
    """Validate frame structure."""
    if not isinstance(frame, list):
        raise ValueError("frame must be a list")
    
    for i, obj in enumerate(frame):
        if not isinstance(obj, dict):
            raise ValueError(f"frame object at index {i} must be a dictionary")
        
        if 'obj_id' not in obj:
            raise ValueError(f"frame object at index {i} missing required key 'obj_id'")
        if 'label' not in obj:
            raise ValueError(f"frame object at index {i} missing required key 'label'")
        if 'bbox' not in obj:
            raise ValueError(f"frame object at index {i} missing required key 'bbox'")
        
        if not isinstance(obj['obj_id'], str) or not obj['obj_id']:
            raise ValueError(f"frame object at index {i} has invalid obj_id (must be non-empty string)")
        
        if not isinstance(obj['label'], str) or not obj['label']:
            raise ValueError(f"frame object at index {i} has invalid label (must be non-empty string)")
        
        try:
            validate_bbox(obj['bbox'])
        except ValueError as e:
            raise ValueError(f"frame object at index {i} has invalid bbox: {str(e)}")
