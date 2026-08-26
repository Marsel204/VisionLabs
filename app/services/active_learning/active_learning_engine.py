"""Parallel, cached active-learning difficulty analysis."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from threading import Lock

import numpy as np

from app.services.active_learning.active_learning_models import (
    ActiveLearningConfig,
    DifficultyFeatures,
    DifficultyResult,
    ImageAnalysis,
    RankingMode,
)
from app.services.active_learning.difficulty_score import (
    calculate_score,
    classify_score,
    recommended_action,
)
from app.services.active_learning.disagreement import disagreement_features
from app.services.active_learning.ranking import rank_results
from app.services.active_learning.uncertainty import (
    average_confidence,
    confidence_uncertainty,
    small_object_ratio,
)

LOGGER = logging.getLogger(__name__)


class _DifficultyCache:
    """SQLite cache for difficulty results keyed by an input signature."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS difficulty_cache "
            "(image_path TEXT PRIMARY KEY, signature TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        self._connection.commit()
        self._lock = Lock()

    def get(self, image_path: Path, signature: str) -> DifficultyResult | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM difficulty_cache WHERE image_path=? AND signature=?",
                (str(image_path), signature),
            ).fetchone()
        if row is None:
            return None
        return _result_from_payload(json.loads(row[0]))

    def put(self, result: DifficultyResult, signature: str) -> None:
        payload = json.dumps(_result_payload(result), sort_keys=True, default=_json_default)
        with self._lock:
            self._connection.execute(
                "INSERT INTO difficulty_cache(image_path, signature, payload) VALUES(?,?,?) "
                "ON CONFLICT(image_path) DO UPDATE SET signature=excluded.signature, "
                "payload=excluded.payload",
                (str(result.image_path), signature, payload),
            )
            self._connection.commit()

    def remove(self, image_path: Path) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM difficulty_cache WHERE image_path=?",
                (str(image_path),),
            )
            self._connection.commit()

    def close(self) -> None:
        """Close the cache database."""
        with self._lock:
            self._connection.close()


class ActiveLearningEngine:
    """Calculate, cache, rank, and summarize image difficulty."""

    def __init__(self, config: ActiveLearningConfig | None = None) -> None:
        self.config = config or ActiveLearningConfig()
        self.config.validate()
        self._cache = _DifficultyCache(self.config.cache_path)

    def close(self) -> None:
        """Close the persistent difficulty cache."""
        self._cache.close()

    def remove(self, image_path: Path) -> None:
        """Remove an image from the persistent difficulty cache."""
        self._cache.remove(image_path)

    def score(self, analysis: ImageAnalysis) -> DifficultyResult:
        """Score one image, returning a cached result when inputs are unchanged."""
        signature = _signature(analysis, self.config)
        cached = self._cache.get(analysis.image_path, signature)
        if cached is not None:
            LOGGER.debug("Difficulty cache hit: %s", analysis.image_path)
            return replace(cached, cached=True)
        LOGGER.info("Difficulty calculation started: %s", analysis.image_path)
        result = self._calculate(analysis)
        self._cache.put(result, signature)
        LOGGER.info("Image scored: %s (%.1f)", analysis.image_path, result.difficulty_score)
        LOGGER.info("Cache updated: %s", analysis.image_path)
        return result

    def score_many(
        self,
        analyses: Iterable[ImageAnalysis],
        *,
        max_workers: int | None = None,
        progress: Callable[[int, int], None] | None = None,
        ranking: RankingMode = RankingMode.HIGHEST_DIFFICULTY,
    ) -> list[DifficultyResult]:
        """Score images in parallel and return them in the requested order."""
        values = list(analyses)
        if not values:
            return []
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.score, analysis) for analysis in values]
            results: list[DifficultyResult] = []
            for future in futures:
                results.append(future.result())
                completed += 1
                if progress:
                    progress(completed, len(values))
        ranked = rank_results(results, ranking)
        LOGGER.info("Ranking complete: %d images", len(ranked))
        return ranked

    def _calculate(self, analysis: ImageAnalysis) -> DifficultyResult:
        detections = analysis.detections
        disagreement, missing, duplicates, occlusion = disagreement_features(detections)
        fusion_conflicts = (
            analysis.fusion_result.statistics.conflicts if analysis.fusion_result is not None else 0
        )
        conflict_count = fusion_conflicts
        object_count = len(detections)
        motorcycle_count = sum(item.class_name.lower() == "motorcycle" for item in detections)
        small_ratio = small_object_ratio(detections, self.config.small_object_area)
        density = min(1.0, object_count / self.config.density_reference)
        conflict_ratio = min(1.0, conflict_count / max(1, object_count))
        missing_ratio = min(1.0, missing / max(1, object_count))
        duplicate_ratio = min(1.0, duplicates / max(1, object_count))
        motorcycle_priority = min(
            1.0,
            (1.0 if motorcycle_count > self.config.motorcycle_threshold else 0.0)
            + (small_ratio if motorcycle_count else 0.0),
        )
        features = DifficultyFeatures(
            confidence_uncertainty=confidence_uncertainty(detections),
            density=density,
            occlusion=occlusion,
            disagreement=disagreement,
            conflict=conflict_ratio,
            small_object_ratio=small_ratio,
            missing_detection=missing_ratio,
            duplicate_detections=duplicate_ratio,
            motorcycle_priority=motorcycle_priority,
            object_count=object_count,
            motorcycle_count=motorcycle_count,
            conflict_count=conflict_count,
            missing_count=missing,
            duplicate_count=int(duplicates),
            average_confidence=average_confidence(detections),
        )
        score = calculate_score(features, self.config)
        return DifficultyResult(
            image_path=analysis.image_path,
            difficulty_score=score,
            difficulty_level=classify_score(score, self.config),
            recommended_action=recommended_action(score, self.config),
            review_priority=round(score),
            features=features,
            collections=_collections(features, score),
        )


def _collections(features: DifficultyFeatures, score: float) -> frozenset[str]:
    collections: set[str] = set()
    if score >= 60.0:
        collections.add("HardExamples")
    if features.object_count >= 50:
        collections.add("CrowdedTraffic")
    if features.occlusion >= 0.50:
        collections.add("Occlusion")
    if features.confidence_uncertainty >= 0.50:
        collections.add("LowConfidence")
    if features.motorcycle_count:
        collections.add("DenseMotorcycles")
    if features.small_object_ratio >= 0.30:
        collections.add("SmallObjects")
    if features.conflict_count:
        collections.add("Conflicts")
    return frozenset(collections)


def _signature(analysis: ImageAnalysis, config: ActiveLearningConfig) -> str:
    payload: dict[str, object] = {
        "config": {key: str(value) for key, value in asdict(config).items()},
        "detections": [
            (
                item.class_name,
                item.confidence,
                item.box.left,
                item.box.top,
                item.box.right,
                item.box.bottom,
                str(item.source),
            )
            for item in analysis.detections
        ],
        "fusion": (
            [
                (item.class_name, item.confidence, item.status.value)
                for item in analysis.fusion_result.detections
            ]
            if analysis.fusion_result is not None
            else None
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _result_payload(result: DifficultyResult) -> dict[str, object]:
    payload = asdict(result)
    payload["image_path"] = str(result.image_path)
    payload["difficulty_level"] = result.difficulty_level.value
    payload["recommended_action"] = result.recommended_action.value
    payload["collections"] = list(result.collections)
    return payload


def _json_default(value: object) -> object:
    """Convert NumPy scalar values produced by vectorized metrics to JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _result_from_payload(payload: dict[str, object]) -> DifficultyResult:
    from app.services.active_learning.active_learning_models import (
        DifficultyLevel,
        RecommendedAction,
    )

    raw_features = payload["features"]
    if not isinstance(raw_features, dict):
        raise ValueError("invalid cached difficulty features")
    features = DifficultyFeatures(**raw_features)  # type: ignore[arg-type]
    raw_collections = payload.get("collections", [])
    if not isinstance(raw_collections, list):
        raise ValueError("invalid cached difficulty collections")
    raw_metadata = payload.get("metadata", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError("invalid cached difficulty metadata")
    return DifficultyResult(
        image_path=Path(str(payload["image_path"])),
        difficulty_score=float(str(payload["difficulty_score"])),
        difficulty_level=DifficultyLevel(str(payload["difficulty_level"])),
        recommended_action=RecommendedAction(str(payload["recommended_action"])),
        review_priority=int(str(payload["review_priority"])),
        features=features,
        collections=frozenset(str(item) for item in raw_collections),
        metadata=raw_metadata,
    )
