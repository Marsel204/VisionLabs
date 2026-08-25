"""Unit tests for FastParametricSolver."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.services.ai_tuner.parametric_solver import FastParametricSolver
from app.services.annotation.domain import Annotation, AnnotationDocument, BoundingBox
from app.services.auto_label.models import AutoLabelDetection, AutoLabelResult


@pytest.fixture
def dummy_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "param_eval.jpg"
    img = Image.new("RGB", (640, 480), color=(180, 180, 180))
    img.save(img_path)
    return img_path


def test_parametric_solver_filters_low_conf_noise(dummy_image: Path) -> None:
    solver = FastParametricSolver()

    gt_doc = AnnotationDocument(
        image_path=dummy_image,
        image_width=640,
        image_height=480,
        annotations=(Annotation(class_name="car", box=BoundingBox(0.1, 0.1, 0.3, 0.3)),),
    )

    # 1 real detection (conf 0.85) + 3 low-confidence noisy detections (conf 0.20, 0.25, 0.30)
    raw_res = AutoLabelResult(
        image_path=dummy_image,
        image_width=640,
        image_height=480,
        detections=[
            AutoLabelDetection(
                class_name="car", confidence=0.85, box=BoundingBox(0.1, 0.1, 0.3, 0.3)
            ),
            AutoLabelDetection(
                class_name="car", confidence=0.20, box=BoundingBox(0.4, 0.4, 0.5, 0.5)
            ),
            AutoLabelDetection(
                class_name="car", confidence=0.25, box=BoundingBox(0.6, 0.6, 0.7, 0.7)
            ),
            AutoLabelDetection(
                class_name="car", confidence=0.30, box=BoundingBox(0.8, 0.8, 0.9, 0.9)
            ),
        ],
    )

    opt_conf, opt_iou, report = solver.optimize_thresholds(
        {dummy_image: raw_res}, {dummy_image: gt_doc}, initial_conf=0.15
    )

    # Solver should raise confidence threshold above 0.30 to filter out noise
    assert opt_conf >= 0.35
    assert report.overall_macro_f1 == 1.0
    assert report.total_matches == 1
