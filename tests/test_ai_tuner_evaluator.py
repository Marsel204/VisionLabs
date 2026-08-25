"""Unit tests for GroundTruthEvaluator."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.services.ai_tuner.evaluator import GroundTruthEvaluator
from app.services.annotation.domain import (
    Annotation,
    AnnotationDocument,
    BoundingBox,
)
from app.services.auto_label.models import AutoLabelDetection, AutoLabelResult


@pytest.fixture
def dummy_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "sample_eval.jpg"
    img = Image.new("RGB", (640, 480), color=(200, 200, 200))
    img.save(img_path)
    return img_path


def test_evaluator_perfect_match(dummy_image: Path) -> None:
    evaluator = GroundTruthEvaluator(iou_threshold=0.50)

    # 2 Ground Truth objects: 1 car, 1 truck
    gt_doc = AnnotationDocument(
        image_path=dummy_image,
        image_width=640,
        image_height=480,
        annotations=(
            Annotation(class_name="car", box=BoundingBox(0.1, 0.1, 0.3, 0.3)),
            Annotation(class_name="truck", box=BoundingBox(0.5, 0.5, 0.8, 0.8)),
        ),
    )

    # Predictions match ground truth
    pred_res = AutoLabelResult(
        image_path=dummy_image,
        image_width=640,
        image_height=480,
        detections=[
            AutoLabelDetection(
                class_name="car", confidence=0.90, box=BoundingBox(0.1, 0.1, 0.3, 0.3)
            ),
            AutoLabelDetection(
                class_name="truck", confidence=0.85, box=BoundingBox(0.5, 0.5, 0.8, 0.8)
            ),
        ],
    )

    report = evaluator.evaluate({dummy_image: pred_res}, {dummy_image: gt_doc}, extract_crops=True)

    assert report.overall_macro_f1 == 1.0
    assert report.overall_precision == 1.0
    assert report.overall_recall == 1.0
    assert report.total_matches == 2
    assert report.total_ground_truth == 2
    assert len(report.error_crops) == 0


def test_evaluator_missed_and_false_positive(dummy_image: Path) -> None:
    evaluator = GroundTruthEvaluator(iou_threshold=0.50)

    gt_doc = AnnotationDocument(
        image_path=dummy_image,
        image_width=640,
        image_height=480,
        annotations=(
            Annotation(class_name="car", box=BoundingBox(0.1, 0.1, 0.3, 0.3)),
            Annotation(class_name="truck", box=BoundingBox(0.5, 0.5, 0.8, 0.8)),
        ),
    )

    # Prediction missed truck, detected car, and detected false positive motorcycle
    pred_res = AutoLabelResult(
        image_path=dummy_image,
        image_width=640,
        image_height=480,
        detections=[
            AutoLabelDetection(
                class_name="car", confidence=0.90, box=BoundingBox(0.1, 0.1, 0.3, 0.3)
            ),
            AutoLabelDetection(
                class_name="motorcycle", confidence=0.70, box=BoundingBox(0.8, 0.8, 0.9, 0.9)
            ),
        ],
    )

    report = evaluator.evaluate({dummy_image: pred_res}, {dummy_image: gt_doc}, extract_crops=True)

    assert report.class_metrics["car"].f1_score == 1.0
    assert report.class_metrics["truck"].recall == 0.0
    assert report.class_metrics["truck"].false_negatives == 1
    assert report.class_metrics["motorcycle"].false_positives == 1
    assert len(report.error_crops) >= 1
    assert any(
        ec.class_name == "truck" and ec.error_type == "missed_ground_truth"
        for ec in report.error_crops
    )
