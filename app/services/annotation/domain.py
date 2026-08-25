"""Vendor-neutral annotation entities and validation rules."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from uuid import UUID, uuid4

from app.core.exceptions import ServiceError

TARGET_CLASSES = frozenset({"motorcycle", "car", "bus", "truck"})


class AnnotationValidationError(ServiceError):
    """Raised when an annotation violates domain invariants."""


class AnnotationSource(StrEnum):
    """Origin of an annotation used for provenance and review prioritization."""

    HUMAN = "human"
    GROUNDING_DINO = "grounding_dino"
    SAM2 = "sam2"
    YOLO = "yolo"
    FLORENCE2 = "florence2"
    LOCATE_ANYTHING = "locate_anything"
    FUSED = "fused"


class ReviewStatus(StrEnum):
    """Human review lifecycle for an annotation."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    MODIFIED = "modified"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An axis-aligned box in normalized ``xyxy`` coordinates."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if any(value != value for value in values):
            raise AnnotationValidationError("bounding box coordinates cannot be NaN")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise AnnotationValidationError("bounding box coordinates must be normalized to [0, 1]")
        if self.left >= self.right or self.top >= self.bottom:
            raise AnnotationValidationError("bounding box must have positive width and height")

    @property
    def width(self) -> float:
        """Return normalized width."""
        return self.right - self.left

    @property
    def height(self) -> float:
        """Return normalized height."""
        return self.bottom - self.top

    @property
    def area(self) -> float:
        """Return normalized area."""
        return self.width * self.height

    def to_yolo(self) -> tuple[float, float, float, float]:
        """Convert normalized ``xyxy`` coordinates to YOLO ``xywh``."""
        return (
            (self.left + self.right) / 2,
            (self.top + self.bottom) / 2,
            self.width,
            self.height,
        )

    def intersection_over_min(self, other: BoundingBox) -> float:
        """Calculate intersection over minimum area (containment ratio) with another box."""
        inter_left = max(self.left, other.left)
        inter_top = max(self.top, other.top)
        inter_right = min(self.right, other.right)
        inter_bottom = min(self.bottom, other.bottom)

        if inter_left >= inter_right or inter_top >= inter_bottom:
            return 0.0

        inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
        min_area = min(self.area, other.area)
        return inter_area / min_area if min_area > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class Annotation:
    """One object annotation with confidence, provenance, and review state."""

    class_name: str
    box: BoundingBox
    confidence: float | None = None
    source: AnnotationSource = AnnotationSource.HUMAN
    review_status: ReviewStatus = ReviewStatus.PENDING
    occluded: bool = False
    truncated: bool = False
    annotation_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if self.class_name not in TARGET_CLASSES:
            raise AnnotationValidationError(f"unsupported target class: {self.class_name}")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise AnnotationValidationError("confidence must be between 0 and 1")

    def accept(self) -> Annotation:
        """Return a copy marked as accepted by human review."""
        return replace(self, review_status=ReviewStatus.ACCEPTED)

    def reject(self) -> Annotation:
        """Return a copy marked as rejected by human review."""
        return replace(self, review_status=ReviewStatus.REJECTED)

    def modify(self, box: BoundingBox, class_name: str | None = None) -> Annotation:
        """Return a human-modified copy while preserving provenance."""
        return replace(
            self,
            box=box,
            class_name=class_name or self.class_name,
            source=AnnotationSource.HUMAN,
            review_status=ReviewStatus.MODIFIED,
        )


@dataclass(frozen=True, slots=True)
class AnnotationDocument:
    """Immutable annotation state for one source image."""

    image_path: Path
    image_width: int
    image_height: int
    annotations: tuple[Annotation, ...] = ()

    def __post_init__(self) -> None:
        if not self.image_path.is_file():
            raise AnnotationValidationError(f"image does not exist: {self.image_path}")
        if self.image_width < 1 or self.image_height < 1:
            raise AnnotationValidationError("image dimensions must be positive")
        ids = [annotation.annotation_id for annotation in self.annotations]
        if len(ids) != len(set(ids)):
            raise AnnotationValidationError("annotation IDs must be unique within a document")

    def add(self, annotation: Annotation) -> AnnotationDocument:
        """Return a document containing one additional annotation."""
        if any(item.annotation_id == annotation.annotation_id for item in self.annotations):
            raise AnnotationValidationError(f"duplicate annotation ID: {annotation.annotation_id}")
        return replace(self, annotations=(*self.annotations, annotation))

    def remove(self, annotation_id: UUID) -> AnnotationDocument:
        """Return a document without the requested annotation."""
        if not any(item.annotation_id == annotation_id for item in self.annotations):
            raise AnnotationValidationError(f"annotation not found: {annotation_id}")
        return replace(
            self,
            annotations=tuple(
                item for item in self.annotations if item.annotation_id != annotation_id
            ),
        )

    def update(self, annotation: Annotation) -> AnnotationDocument:
        """Return a document with an existing annotation replaced."""
        if not any(item.annotation_id == annotation.annotation_id for item in self.annotations):
            raise AnnotationValidationError(f"annotation not found: {annotation.annotation_id}")
        return replace(
            self,
            annotations=tuple(
                annotation if item.annotation_id == annotation.annotation_id else item
                for item in self.annotations
            ),
        )
