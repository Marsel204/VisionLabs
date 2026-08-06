"""Public data models used by the label fusion service."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from app.services.annotation.domain import AnnotationSource, BoundingBox


class FusionStatus(StrEnum):
    """Decision assigned to one unified detection."""

    ACCEPTED = "accepted"
    NEEDS_REVIEW = "needs_review"
    CONFLICT = "conflict"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Validated thresholds and switches controlling label fusion."""

    iou_threshold: float = 0.60
    confidence_difference: float = 0.25
    minimum_box_area: float = 0.001
    merge_same_class: bool = True
    enable_duplicate_removal: bool = True
    overlap_removal_iou_threshold: float = 0.50
    overlap_removal_containment_threshold: float = 0.80
    overlap_removal_same_class_only: bool = True

    def validate(self) -> None:
        """Raise ``ValueError`` when a setting is outside its valid range."""
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if not 0.0 <= self.confidence_difference <= 1.0:
            raise ValueError("confidence_difference must be in [0, 1]")
        if not 0.0 <= self.minimum_box_area <= 1.0:
            raise ValueError("minimum_box_area must be in [0, 1]")
        if not 0.0 < self.overlap_removal_iou_threshold <= 1.0:
            raise ValueError("overlap_removal_iou_threshold must be in (0, 1]")
        if not 0.0 < self.overlap_removal_containment_threshold <= 1.0:
            raise ValueError("overlap_removal_containment_threshold must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class Detection:
    """One model prediction after fusion rules have been applied."""

    class_name: str
    confidence: float
    bbox: BoundingBox
    source_model: str
    id: UUID = field(default_factory=uuid4)
    mask: np.ndarray | None = None
    status: FusionStatus = FusionStatus.NEEDS_REVIEW
    accepted: bool = False
    review: bool = True
    rejected: bool = False
    matched_ids: tuple[UUID, ...] = ()
    matched_iou: float | None = None
    source_models: frozenset[AnnotationSource] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.status is FusionStatus.ACCEPTED and not self.accepted:
            raise ValueError("accepted detections must set accepted=True")


@dataclass(frozen=True, slots=True)
class FusionStatistics:
    """Summary metrics for one fusion run."""

    accepted: int = 0
    needs_review: int = 0
    rejected: int = 0
    conflicts: int = 0
    duplicate_removed: int = 0
    average_iou: float = 0.0


@dataclass(frozen=True, slots=True)
class FusionResult:
    """Unified detections and statistics returned by the fusion engine."""

    detections: tuple[Detection, ...]
    statistics: FusionStatistics
