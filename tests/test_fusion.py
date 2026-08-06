from pathlib import Path

import pytest

from app.models.contracts import Detection as ModelDetection
from app.services.annotation.domain import Annotation, AnnotationSource, BoundingBox
from app.services.fusion import (
    FusionConfig,
    FusionEngine,
    FusionStatus,
    remove_overlapping_annotations,
)
from app.services.fusion.iou import intersection_over_union, pairwise_iou


def detection(
    class_name: str, box: BoundingBox, confidence: float, source: AnnotationSource
) -> ModelDetection:
    return ModelDetection(class_name, box, confidence, source)


def test_iou_and_vectorized_iou_match() -> None:
    first = BoundingBox(0.0, 0.0, 0.5, 0.5)
    second = BoundingBox(0.25, 0.25, 0.75, 0.75)
    expected = intersection_over_union(first, second)
    matrix = pairwise_iou([first], [second])
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == pytest.approx(expected)


def test_same_class_models_are_accepted() -> None:
    box = BoundingBox(0.1, 0.1, 0.5, 0.5)
    result = FusionEngine().fuse([
        detection("car", box, 0.9, AnnotationSource.YOLO),
        detection(
            "car",
            BoundingBox(0.11, 0.11, 0.51, 0.51),
            0.85,
            AnnotationSource.GROUNDING_DINO,
        ),
    ])
    assert result.detections[0].status is FusionStatus.ACCEPTED
    assert result.statistics.accepted == 1


def test_overlapping_different_classes_are_conflicts() -> None:
    box = BoundingBox(0.1, 0.1, 0.5, 0.5)
    result = FusionEngine().fuse([
        detection("car", box, 0.9, AnnotationSource.YOLO),
        detection("bus", box, 0.85, AnnotationSource.GROUNDING_DINO),
    ])
    assert result.detections[0].status is FusionStatus.CONFLICT
    assert result.statistics.conflicts == 1


def test_same_source_duplicate_keeps_highest_confidence() -> None:
    box = BoundingBox(0.1, 0.1, 0.5, 0.5)
    result = FusionEngine().fuse([
        detection("car", box, 0.4, AnnotationSource.YOLO),
        detection("car", BoundingBox(0.11, 0.11, 0.51, 0.51), 0.8, AnnotationSource.YOLO),
    ])
    assert len(result.detections) == 1
    assert result.detections[0].confidence == 0.8
    assert result.statistics.duplicate_removed == 1


def test_single_detection_and_confidence_difference_need_review() -> None:
    box = BoundingBox(0.1, 0.1, 0.5, 0.5)
    result = FusionEngine(FusionConfig(confidence_difference=0.1)).fuse([
        detection("car", box, 0.98, AnnotationSource.YOLO),
        detection("car", BoundingBox(0.11, 0.11, 0.51, 0.51), 0.7, AnnotationSource.GROUNDING_DINO),
        detection("bus", BoundingBox(0.7, 0.7, 0.8, 0.8), 0.8, AnnotationSource.YOLO),
    ])
    assert result.statistics.needs_review == 2


def test_small_box_needs_review_and_statistics_average_iou() -> None:
    box = BoundingBox(0.1, 0.1, 0.11, 0.11)
    result = FusionEngine().fuse([
        detection("car", box, 0.9, AnnotationSource.YOLO),
        detection("car", BoundingBox(0.1, 0.1, 0.11, 0.11), 0.9, AnnotationSource.GROUNDING_DINO),
    ])
    assert result.detections[0].status is FusionStatus.NEEDS_REVIEW
    assert result.statistics.average_iou == pytest.approx(1.0)


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        FusionEngine(FusionConfig(iou_threshold=0.0))


def test_fusion_configuration_can_be_loaded_from_yaml(tmp_path: Path) -> None:
    from app.configs.settings import AppSettings

    path = tmp_path / "fusion.yaml"
    path.write_text("iou_threshold: 0.7\nminimum_box_area: 0.01\n", encoding="utf-8")
    settings = AppSettings.from_fusion_yaml(path)
    assert settings.fusion.iou_threshold == 0.7
    assert settings.fusion.minimum_box_area == 0.01


def test_overlap_cleanup_keeps_highest_confidence_same_class() -> None:
    annotations = (
        Annotation("car", BoundingBox(0.1, 0.1, 0.5, 0.5), 0.4),
        Annotation("car", BoundingBox(0.11, 0.11, 0.51, 0.51), 0.9),
    )
    kept, removed = remove_overlapping_annotations(annotations)
    assert removed == 1
    assert len(kept) == 1
    assert kept[0].confidence == 0.9


def test_overlap_cleanup_preserves_different_classes() -> None:
    annotations = (
        Annotation("car", BoundingBox(0.1, 0.1, 0.5, 0.5), 0.9),
        Annotation("bus", BoundingBox(0.1, 0.1, 0.5, 0.5), 0.8),
    )
    kept, removed = remove_overlapping_annotations(annotations)
    assert removed == 0
    assert len(kept) == 2


def test_overlap_cleanup_removes_contained_lower_confidence_box() -> None:
    annotations = (
        Annotation("car", BoundingBox(0.1, 0.1, 0.7, 0.7), 0.8),
        Annotation("car", BoundingBox(0.3, 0.3, 0.5, 0.5), 0.9),
    )
    kept, removed = remove_overlapping_annotations(annotations)
    assert removed == 1
    assert kept[0].confidence == 0.9
