"""Pure decision rules used by :class:`FusionEngine`."""

from __future__ import annotations

from collections.abc import Sequence

from app.models.contracts import Detection as ModelDetection
from app.services.fusion.confidence import exceeds_confidence_difference
from app.services.fusion.fusion_models import FusionConfig, FusionStatus


def decide_status(
    detections: Sequence[ModelDetection],
    config: FusionConfig,
) -> FusionStatus:
    """Apply fusion rules to one matched group."""
    if any(item.box.area < config.minimum_box_area for item in detections):
        return FusionStatus.NEEDS_REVIEW
    classes = {item.class_name for item in detections}
    sources = {item.source for item in detections}
    if len(classes) > 1:
        return FusionStatus.CONFLICT
    if len(sources) < 2:
        return FusionStatus.NEEDS_REVIEW
    if exceeds_confidence_difference(
        [item.confidence for item in detections], config.confidence_difference
    ):
        return FusionStatus.NEEDS_REVIEW
    return FusionStatus.ACCEPTED if config.merge_same_class else FusionStatus.NEEDS_REVIEW
