"""Aggregate active-learning dataset statistics."""

from __future__ import annotations

from collections.abc import Iterable

from app.services.active_learning.active_learning_models import (
    ActiveLearningStatistics,
    DifficultyLevel,
    DifficultyResult,
)


def summarize(results: Iterable[DifficultyResult]) -> ActiveLearningStatistics:
    """Return aggregate difficulty, confidence, density, and conflict metrics."""
    values = list(results)
    if not values:
        return ActiveLearningStatistics(0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0)
    return ActiveLearningStatistics(
        image_count=len(values),
        average_difficulty=sum(item.difficulty_score for item in values) / len(values),
        average_confidence=sum(item.features.average_confidence for item in values) / len(values),
        hard_image_count=sum(
            item.difficulty_level in {DifficultyLevel.HARD, DifficultyLevel.EXTREME}
            for item in values
        ),
        easy_image_count=sum(
            item.difficulty_level in {DifficultyLevel.VERY_EASY, DifficultyLevel.EASY}
            for item in values
        ),
        conflict_count=sum(item.features.conflict_count for item in values),
        average_object_count=sum(item.features.object_count for item in values) / len(values),
        average_motorcycle_count=sum(item.features.motorcycle_count for item in values)
        / len(values),
    )
