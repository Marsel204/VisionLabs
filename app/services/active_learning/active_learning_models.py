"""Typed inputs and outputs for active-learning analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.models.contracts import Detection as ModelDetection
from app.services.fusion.fusion_models import FusionResult


class DifficultyLevel(StrEnum):
    """Human-readable difficulty bands."""

    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


class RecommendedAction(StrEnum):
    """Suggested next action for an image."""

    ACCEPT_AUTOMATICALLY = "accept_automatically"
    QUICK_REVIEW = "quick_review"
    NEEDS_REVIEW = "needs_review"
    MANUAL_ANNOTATION_REQUIRED = "manual_annotation_required"


class RankingMode(StrEnum):
    """Supported review queue ordering modes."""

    HIGHEST_DIFFICULTY = "highest_difficulty"
    LOWEST_DIFFICULTY = "lowest_difficulty"
    MOST_CONFLICTS = "most_conflicts"
    MOST_OCCLUSION = "most_occlusion"
    HIGHEST_DENSITY = "highest_density"
    LOWEST_CONFIDENCE = "lowest_confidence"
    MOST_MISSING_OBJECTS = "most_missing_objects"


@dataclass(frozen=True, slots=True)
class ActiveLearningConfig:
    """Validated weights and thresholds for difficulty scoring."""

    confidence_weight: float = 0.20
    density_weight: float = 0.20
    occlusion_weight: float = 0.15
    disagreement_weight: float = 0.10
    conflict_weight: float = 0.15
    small_object_weight: float = 0.10
    missing_detection_weight: float = 0.10
    duplicate_weight: float = 0.05
    motorcycle_weight: float = 0.05
    density_reference: int = 50
    small_object_area: float = 0.001
    motorcycle_threshold: int = 20
    motorcycle_small_ratio: float = 0.30
    hard_threshold: float = 60.0
    extreme_threshold: float = 81.0
    cache_path: Path = Path.home() / ".cache" / "traffic-annotator" / "active-learning.sqlite"

    def validate(self) -> None:
        """Raise ``ValueError`` for invalid weights or thresholds."""
        weights = (
            self.confidence_weight,
            self.density_weight,
            self.occlusion_weight,
            self.disagreement_weight,
            self.conflict_weight,
            self.small_object_weight,
            self.missing_detection_weight,
            self.duplicate_weight,
            self.motorcycle_weight,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("active-learning weights must be non-negative and non-zero")
        if self.density_reference < 1 or self.motorcycle_threshold < 0:
            raise ValueError("object thresholds must be non-negative")
        if not 0.0 <= self.small_object_area <= 1.0:
            raise ValueError("small_object_area must be in [0, 1]")
        if not 0.0 <= self.motorcycle_small_ratio <= 1.0:
            raise ValueError("motorcycle_small_ratio must be in [0, 1]")
        if not 0.0 <= self.hard_threshold <= 100.0:
            raise ValueError("hard_threshold must be in [0, 100]")
        if not 0.0 <= self.extreme_threshold <= 100.0:
            raise ValueError("extreme_threshold must be in [0, 100]")


@dataclass(frozen=True, slots=True)
class ImageAnalysis:
    """Inference and fusion data required to score one image."""

    image_path: Path
    detections: tuple[ModelDetection, ...] = ()
    fusion_result: FusionResult | None = None
    image_area: float = 1.0


@dataclass(frozen=True, slots=True)
class DifficultyFeatures:
    """Normalized feature values contributing to an image score."""

    confidence_uncertainty: float
    density: float
    occlusion: float
    disagreement: float
    conflict: float
    small_object_ratio: float
    missing_detection: float
    duplicate_detections: float
    motorcycle_priority: float
    object_count: int
    motorcycle_count: int
    conflict_count: int
    missing_count: int
    duplicate_count: int
    average_confidence: float


@dataclass(frozen=True, slots=True)
class DifficultyResult:
    """Difficulty score and review metadata for one image."""

    image_path: Path
    difficulty_score: float
    difficulty_level: DifficultyLevel
    recommended_action: RecommendedAction
    review_priority: int
    features: DifficultyFeatures
    collections: frozenset[str] = frozenset()
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActiveLearningStatistics:
    """Aggregate statistics for a collection of scored images."""

    image_count: int
    average_difficulty: float
    average_confidence: float
    hard_image_count: int
    easy_image_count: int
    conflict_count: int
    average_object_count: float
    average_motorcycle_count: float
