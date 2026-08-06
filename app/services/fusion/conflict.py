"""Conflict predicates for overlapping model detections."""

from __future__ import annotations

from app.models.contracts import Detection
from app.services.fusion.iou import intersection_over_union


def is_class_conflict(first: Detection, second: Detection, iou_threshold: float) -> bool:
    """Return whether overlapping detections disagree on their class."""
    return (
        first.class_name != second.class_name
        and intersection_over_union(first.box, second.box) >= iou_threshold
    )
