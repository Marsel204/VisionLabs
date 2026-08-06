"""Uncertainty and object-size feature calculations."""

from __future__ import annotations

from collections.abc import Sequence

from app.models.contracts import Detection


def average_confidence(detections: Sequence[Detection]) -> float:
    """Return mean confidence, or zero for an image without detections."""
    return sum(item.confidence for item in detections) / len(detections) if detections else 0.0


def confidence_uncertainty(detections: Sequence[Detection]) -> float:
    """Return normalized uncertainty where low confidence means high uncertainty."""
    return 1.0 - average_confidence(detections)


def small_object_ratio(detections: Sequence[Detection], minimum_area: float) -> float:
    """Return the fraction of detections smaller than ``minimum_area``."""
    if not detections:
        return 0.0
    return sum(item.box.area < minimum_area for item in detections) / len(detections)
