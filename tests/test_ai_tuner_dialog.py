"""UI and interaction unit tests for AITunerDialog."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from app.services.ai_tuner.models import TunerIteration, TunerResult
from app.services.annotation.domain import Annotation, AnnotationDocument, BoundingBox
from app.services.auto_label.models import AutoLabelClass, AutoLabelConfig
from app.ui.dialogs.ai_tuner_dialog import AITunerDialog


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def sample_data(tmp_path: Path) -> tuple[list[Path], dict[Path, AnnotationDocument]]:
    paths = []
    docs = {}
    for i in range(2):
        p = tmp_path / f"dialog_test_{i}.jpg"
        img = Image.new("RGB", (640, 480), color=(120, 120, 120))
        img.save(p)
        paths.append(p)
        docs[p] = AnnotationDocument(
            image_path=p,
            image_width=640,
            image_height=480,
            annotations=(Annotation(class_name="car", box=BoundingBox(0.1, 0.1, 0.3, 0.3)),),
        )
    return paths, docs


def test_ai_tuner_dialog_init(
    qapp: QApplication, sample_data: tuple[list[Path], dict[Path, AnnotationDocument]]
) -> None:
    paths, docs = sample_data
    config = AutoLabelConfig(
        classes=[
            AutoLabelClass(name="car", prompt="car"),
            AutoLabelClass(name="truck", prompt="truck"),
        ]
    )

    dialog = AITunerDialog(
        sample_images=paths,
        ground_truth=docs,
        current_config=config,
    )

    assert "AI Auto-Tuner" in dialog.windowTitle()
    assert dialog.start_btn.isEnabled() is True
    assert dialog.apply_btn.isEnabled() is False
    assert "Found 2" in dialog.gt_status_text.text()
    assert dialog.diff_table.rowCount() == 2


def test_ai_tuner_dialog_empty_gt(
    qapp: QApplication, sample_data: tuple[list[Path], dict[Path, AnnotationDocument]]
) -> None:
    paths, _ = sample_data
    config = AutoLabelConfig()

    dialog = AITunerDialog(
        sample_images=paths,
        ground_truth={},  # No ground truth
        current_config=config,
    )

    assert dialog.start_btn.isEnabled() is False
    assert "No ground-truth annotations found" in dialog.gt_status_text.text()


def test_ai_tuner_dialog_progress_and_apply(
    qapp: QApplication, sample_data: tuple[list[Path], dict[Path, AnnotationDocument]]
) -> None:
    paths, docs = sample_data
    initial_config = AutoLabelConfig(
        classes=[AutoLabelClass(name="car", prompt="sedan")],
        confidence_threshold=0.35,
    )

    dialog = AITunerDialog(
        sample_images=paths,
        ground_truth=docs,
        current_config=initial_config,
    )

    # Simulate an iteration update
    iter_update = TunerIteration(
        iteration_index=1,
        f1_score=0.88,
        precision=0.90,
        recall=0.86,
        prompt_updates={"car": "passenger car, sedan, suv"},
        confidence_threshold=0.40,
        iou_threshold=0.50,
        diagnostics="Recall improved",
        llm_reasoning="Added SUV keyword",
        elapsed_seconds=1.2,
    )
    dialog._on_iteration_step(iter_update)

    assert dialog.progress_bar.value() == 88
    assert "88% F1" in dialog.score_tracker_label.text()

    # Simulate completion
    final_config = AutoLabelConfig(
        classes=[AutoLabelClass(name="car", prompt="passenger car, sedan, suv")],
        confidence_threshold=0.40,
        box_iou_threshold=0.50,
    )
    res = TunerResult(
        initial_config=initial_config,
        final_config=final_config,
        initial_f1=0.50,
        final_f1=0.88,
        target_reached=True,
        iterations=[iter_update],
        total_elapsed_seconds=2.5,
        summary="Optimized successfully",
    )
    dialog._on_tuning_completed(res)

    assert dialog.apply_btn.isEnabled() is True
    assert "Target Reached! 🎉" in dialog.step_badge.text()

    applied_configs = []
    dialog.tuning_applied.connect(lambda cfg: applied_configs.append(cfg))
    dialog._on_apply_clicked()

    assert len(applied_configs) == 1
    assert applied_configs[0].confidence_threshold == 0.40
    assert applied_configs[0].classes[0].prompt == "passenger car, sedan, suv"
