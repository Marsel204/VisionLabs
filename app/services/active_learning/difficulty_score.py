"""Weighted difficulty scoring and review recommendations."""

from __future__ import annotations

from app.services.active_learning.active_learning_models import (
    ActiveLearningConfig,
    DifficultyFeatures,
    DifficultyLevel,
    RecommendedAction,
)


def calculate_score(features: DifficultyFeatures, config: ActiveLearningConfig) -> float:
    """Calculate a weighted difficulty score in the range 0 to 100."""
    weighted = (
        features.confidence_uncertainty * config.confidence_weight
        + features.density * config.density_weight
        + features.occlusion * config.occlusion_weight
        + features.disagreement * config.disagreement_weight
        + features.conflict * config.conflict_weight
        + features.small_object_ratio * config.small_object_weight
        + features.missing_detection * config.missing_detection_weight
        + features.duplicate_detections * config.duplicate_weight
        + features.motorcycle_priority * config.motorcycle_weight
    )
    total_weight = sum(
        (
            config.confidence_weight,
            config.density_weight,
            config.occlusion_weight,
            config.disagreement_weight,
            config.conflict_weight,
            config.small_object_weight,
            config.missing_detection_weight,
            config.duplicate_weight,
            config.motorcycle_weight,
        )
    )
    return max(0.0, min(100.0, weighted / total_weight * 100.0))


def classify_score(score: float, config: ActiveLearningConfig) -> DifficultyLevel:
    """Map a score to the configured difficulty bands."""
    if score <= 20.0:
        return DifficultyLevel.VERY_EASY
    if score <= 40.0:
        return DifficultyLevel.EASY
    if score <= 60.0:
        return DifficultyLevel.MEDIUM
    if score < config.extreme_threshold:
        return DifficultyLevel.HARD
    return DifficultyLevel.EXTREME


def recommended_action(score: float, config: ActiveLearningConfig) -> RecommendedAction:
    """Return the recommended human action for a difficulty score."""
    if score <= 20.0:
        return RecommendedAction.ACCEPT_AUTOMATICALLY
    if score <= 40.0:
        return RecommendedAction.QUICK_REVIEW
    if score < config.hard_threshold:
        return RecommendedAction.NEEDS_REVIEW
    return RecommendedAction.MANUAL_ANNOTATION_REQUIRED
