from pathlib import Path

from app.models.contracts import Detection
from app.services.active_learning import (
    ActiveLearningConfig,
    ActiveLearningEngine,
    DifficultyLevel,
    ImageAnalysis,
    RankingMode,
)
from app.services.active_learning.ranking import filter_results, rank_results
from app.services.active_learning.statistics import summarize
from app.services.annotation.domain import AnnotationSource, BoundingBox


def model_detection(
    class_name: str, confidence: float, source: AnnotationSource
) -> Detection:
    return Detection(
        class_name,
        BoundingBox(0.1, 0.1, 0.4, 0.4),
        confidence,
        source,
    )


def test_difficulty_score_prioritizes_low_confidence_and_density(tmp_path: Path) -> None:
    config = ActiveLearningConfig(cache_path=tmp_path / "cache.sqlite")
    engine = ActiveLearningEngine(config)
    result = engine.score(
        ImageAnalysis(
            tmp_path / "hard.jpg",
            tuple(
                model_detection("motorcycle", 0.1, AnnotationSource.YOLO)
                for _ in range(60)
            ),
        )
    )
    assert result.difficulty_score > 40.0
    assert result.features.object_count == 60
    assert result.difficulty_level in {
        DifficultyLevel.MEDIUM,
        DifficultyLevel.HARD,
        DifficultyLevel.EXTREME,
    }
    engine.close()


def test_cache_is_reused_until_inputs_change(tmp_path: Path) -> None:
    engine = ActiveLearningEngine(ActiveLearningConfig(cache_path=tmp_path / "cache.sqlite"))
    analysis = ImageAnalysis(
        tmp_path / "image.jpg",
        (model_detection("car", 0.9, AnnotationSource.YOLO),),
    )
    first = engine.score(analysis)
    second = engine.score(analysis)
    assert not first.cached
    assert second.cached
    engine.close()


def test_cache_serializes_vectorized_duplicate_metrics(tmp_path: Path) -> None:
    engine = ActiveLearningEngine(ActiveLearningConfig(cache_path=tmp_path / "cache.sqlite"))
    duplicate = model_detection("car", 0.8, AnnotationSource.YOLO)
    analysis = ImageAnalysis(tmp_path / "duplicates.jpg", (duplicate, duplicate))
    result = engine.score(analysis)
    assert result.features.duplicate_count == 1
    assert engine.score(analysis).cached
    engine.close()


def test_ranking_and_filters(tmp_path: Path) -> None:
    low = ImageAnalysis(tmp_path / "low.jpg", ())
    high = ImageAnalysis(
        tmp_path / "high.jpg",
        (model_detection("motorcycle", 0.2, AnnotationSource.YOLO),),
    )
    engine = ActiveLearningEngine(ActiveLearningConfig(cache_path=tmp_path / "cache.sqlite"))
    results = [engine.score(low), engine.score(high)]
    ranked = rank_results(results, RankingMode.HIGHEST_DIFFICULTY)
    assert ranked[0].image_path == tmp_path / "high.jpg"
    assert filter_results(results, motorcycles_only=True)[0].image_path == tmp_path / "high.jpg"
    engine.close()


def test_statistics(tmp_path: Path) -> None:
    engine = ActiveLearningEngine(ActiveLearningConfig(cache_path=tmp_path / "cache.sqlite"))
    results = [
        engine.score(
            ImageAnalysis(
                tmp_path / "one.jpg",
                (model_detection("car", 0.9, AnnotationSource.YOLO),),
            )
        ),
        engine.score(
            ImageAnalysis(
                tmp_path / "two.jpg",
                (model_detection("motorcycle", 0.2, AnnotationSource.YOLO),),
            )
        ),
    ]
    statistics = summarize(results)
    assert statistics.image_count == 2
    assert statistics.average_object_count == 1.0
    assert statistics.average_motorcycle_count == 0.5
    engine.close()
