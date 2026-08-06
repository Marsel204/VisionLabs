"""Stable sorting and filtering for active-learning queues."""

from __future__ import annotations

from collections.abc import Iterable

from app.services.active_learning.active_learning_models import DifficultyResult, RankingMode


def rank_results(
    results: Iterable[DifficultyResult], mode: RankingMode = RankingMode.HIGHEST_DIFFICULTY
) -> list[DifficultyResult]:
    """Return results sorted by the selected review priority."""
    values = list(results)
    reverse = mode is not RankingMode.LOWEST_DIFFICULTY
    key = {
        RankingMode.HIGHEST_DIFFICULTY: lambda item: item.difficulty_score,
        RankingMode.LOWEST_DIFFICULTY: lambda item: item.difficulty_score,
        RankingMode.MOST_CONFLICTS: lambda item: item.features.conflict_count,
        RankingMode.MOST_OCCLUSION: lambda item: item.features.occlusion,
        RankingMode.HIGHEST_DENSITY: lambda item: item.features.object_count,
        RankingMode.LOWEST_CONFIDENCE: lambda item: item.features.average_confidence,
        RankingMode.MOST_MISSING_OBJECTS: lambda item: item.features.missing_count,
    }[mode]
    if mode is RankingMode.LOWEST_CONFIDENCE:
        reverse = False
    return sorted(values, key=lambda item: (key(item), str(item.image_path)), reverse=reverse)


def filter_results(
    results: Iterable[DifficultyResult],
    *,
    hard_only: bool = False,
    easy_only: bool = False,
    conflicts_only: bool = False,
    dense_only: bool = False,
    motorcycles_only: bool = False,
) -> list[DifficultyResult]:
    """Filter scored images by review-oriented predicates."""
    filtered = list(results)
    if hard_only:
        filtered = [item for item in filtered if item.difficulty_score >= 60.0]
    if easy_only:
        filtered = [item for item in filtered if item.difficulty_score <= 40.0]
    if conflicts_only:
        filtered = [item for item in filtered if item.features.conflict_count > 0]
    if dense_only:
        filtered = [item for item in filtered if "CrowdedTraffic" in item.collections]
    if motorcycles_only:
        filtered = [item for item in filtered if item.features.motorcycle_count > 0]
    return filtered
