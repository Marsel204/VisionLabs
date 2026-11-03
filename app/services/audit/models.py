"""Data structures for the annotation audit (missed-object and wrong-box detection)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from app.services.annotation.domain import BoundingBox


class FindingKind(StrEnum):
    """Category of a suspected annotation problem."""

    POSSIBLY_MISSED = "possibly_missed"
    MISALIGNED = "misaligned"
    CLASS_MISMATCH = "class_mismatch"
    DUPLICATE = "duplicate"
    UNSUPPORTED = "unsupported"
    IMPLAUSIBLE_SHAPE = "implausible_shape"


@dataclass(frozen=True, slots=True)
class AuditBox:
    """An existing annotation reduced to what the audit needs."""

    id: str
    class_name: str
    box: BoundingBox
    confidence: float | None = None
    source: str = "human"


@dataclass(frozen=True, slots=True)
class Prediction:
    """A reference detection produced by a high-recall detector pass."""

    class_name: str
    box: BoundingBox
    confidence: float


@dataclass(frozen=True, slots=True)
class OcclusionEstimate:
    """Fraction of one box hidden behind nearer boxes."""

    annotation_id: str
    fraction: float
    occluder_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AuditFinding:
    """One suspected problem with an annotation, or one suspected missing object."""

    kind: FindingKind
    severity: float
    message: str
    annotation_id: str | None = None
    box: BoundingBox | None = None
    class_name: str | None = None
    confidence: float | None = None
    occlusion: float = 0.0
    likely_occluded: bool = False


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """Thresholds controlling what the audit reports."""

    min_prediction_confidence: float = 0.10
    missed_min_confidence: float = 0.30
    # Candidates that overlap an existing box are likely occluded, and detectors score
    # partly hidden objects low, so their confidence threshold is relaxed by this factor.
    occluded_confidence_relief: float = 0.5
    match_iou: float = 0.45
    relaxed_match_iou: float = 0.25
    realign_iou: float = 0.70
    misaligned_min_confidence: float = 0.30
    class_mismatch_min_confidence: float = 0.50
    duplicate_iou: float = 0.70
    duplicate_containment: float = 0.90
    # A small box inside a large one is a distinct (usually occluded) object, not a duplicate.
    duplicate_min_area_ratio: float = 0.60
    candidate_nms_iou: float = 0.60
    unsupported_iou: float = 0.25
    occluded_threshold: float = 0.25
    depth_tolerance: float = 0.03
    min_severity: float = 0.15
    min_pixel_area: float = 16.0
    min_aspect: float = 0.10
    max_aspect: float = 10.0


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Audit outcome for one image."""

    findings: tuple[AuditFinding, ...] = field(default_factory=tuple)
    occlusion: tuple[OcclusionEstimate, ...] = field(default_factory=tuple)

    def count(self, kind: FindingKind) -> int:
        """Return the number of findings of one kind."""
        return sum(1 for finding in self.findings if finding.kind is kind)
