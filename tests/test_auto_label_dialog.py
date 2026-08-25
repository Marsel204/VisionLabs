"""Unit tests for Roboflow-style AutoLabelDialog and MainWindow integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from app.configs.settings import AppSettings
from app.services.annotation.domain import AnnotationDocument, BoundingBox
from app.services.auto_label.engine import AutoLabelEngine
from app.services.auto_label.models import (
    AutoLabelClass,
    AutoLabelConfig,
    AutoLabelDetection,
    AutoLabelPipelineMode,
    AutoLabelResult,
)
from app.ui.dialogs.auto_label_dialog import AutoLabelDialog, ClassCardWidget
from app.ui.main_window import MainWindow


@pytest.fixture
def sample_images(tmp_path: Path) -> list[Path]:
    """Create a pair of temporary sample images."""
    paths = []
    for name in ("img1.jpg", "img2.jpg"):
        p = tmp_path / name
        Image.new("RGB", (640, 480), color=(50, 100, 150)).save(p)
        paths.append(p)
    return paths


def test_class_card_widget(qapp: QApplication) -> None:
    """Test ClassCardWidget interactions."""
    cls_item = AutoLabelClass(name="truck", prompt="semi truck", color="#ef5350")
    card = ClassCardWidget(cls_item)

    assert card.name_edit.text() == "truck"
    assert card.prompt_edit.text() == "semi truck"

    # Test editing name & prompt
    card.name_edit.setText("delivery_truck")
    assert cls_item.name == "delivery_truck"

    card.prompt_edit.setText("large commercial truck")
    assert cls_item.prompt == "large commercial truck"

    deleted_signal_called = []
    card.deleted.connect(lambda c: deleted_signal_called.append(c))
    card.delete_btn.click()
    assert len(deleted_signal_called) == 1
    assert deleted_signal_called[0] is card


def test_auto_label_dialog_lifecycle(sample_images: list[Path], qapp: QApplication) -> None:
    """Test AutoLabelDialog creation, class management, and model switching."""
    mock_engine = MagicMock(spec=AutoLabelEngine)
    dialog = AutoLabelDialog(
        image_paths=sample_images,
        engine=mock_engine,
    )

    assert dialog.image_list.count() == 2
    assert dialog.current_image_path == sample_images[0]
    assert len(dialog.classes) == 4  # Default classes

    # Test adding a class
    initial_count = len(dialog.classes)
    dialog._add_new_class()
    assert len(dialog.classes) == initial_count + 1
    assert dialog.class_count_badge.text() == str(initial_count + 1)

    # Test removing a class
    first_card = dialog._class_cards[0]
    dialog._remove_class_card(first_card)
    assert len(dialog.classes) == initial_count
    assert dialog.class_count_badge.text() == str(initial_count)

    # Test model combo switching
    dialog.model_combo.setCurrentIndex(1)
    mode_val = dialog.model_combo.currentData()
    assert mode_val == AutoLabelPipelineMode.DINO_BOXES.value
    assert dialog.model_badge.text() == "Box labels"

    # Test YOLO options
    yolo_idx = [
        i for i in range(dialog.model_combo.count())
        if dialog.model_combo.itemData(i) == AutoLabelPipelineMode.YOLO_SAM2_MASKS.value
    ][0]
    dialog.model_combo.setCurrentIndex(yolo_idx)
    assert dialog.model_badge.text() == "Mask labels"
    assert not dialog.yolo_weights_btn.isHidden()
    assert "YOLO" in dialog.model_combo.currentText()

    yolo_box_idx = [
        i for i in range(dialog.model_combo.count())
        if dialog.model_combo.itemData(i) == AutoLabelPipelineMode.YOLO_BOXES.value
    ][0]
    dialog.model_combo.setCurrentIndex(yolo_box_idx)
    assert dialog.model_badge.text() == "Box labels"
    assert not dialog.yolo_weights_btn.isHidden()

    # Test Florence-2 VLM option
    vlm_idx = [
        i for i in range(dialog.model_combo.count())
        if dialog.model_combo.itemData(i) == AutoLabelPipelineMode.VLM_SAM2_MASKS.value
    ][0]
    dialog.model_combo.setCurrentIndex(vlm_idx)
    assert dialog.model_badge.text() == "Mask labels"
    assert "Florence-2" in dialog.model_combo.currentText()

    # Test custom weights picker
    with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=("/path/to/custom_weights.pt", "All")):
        with patch("ultralytics.YOLO") as mock_yolo_cls:
            mock_custom_yolo = MagicMock()
            mock_yolo_cls.return_value = mock_custom_yolo
            dialog._choose_custom_yolo_weights()
            assert dialog.engine._yolo_detector is mock_custom_yolo
            assert dialog.yolo_weights_btn.text() == "📦 custom_weights.pt"

    # Non-YOLO mode hides weights button
    dialog.model_combo.setCurrentIndex(0)
    assert dialog.yolo_weights_btn.isHidden()

    # Test Ensemble checkboxes toggling
    dialog.dino_chk.setChecked(True)
    dialog.yolo_chk.setChecked(True)
    dialog.florence_chk.setChecked(True)
    assert "Fused (3 models)" in dialog.model_badge.text()
    assert dialog.model_combo.currentData() == AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS.value

    config = dialog._get_current_config()
    assert config.enable_grounding_dino is True
    assert config.enable_yolo is True
    assert config.enable_florence2 is True
    assert config.mode == AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS

    # Test Clear All
    dialog._clear_all_classes()
    assert len(dialog.classes) == 0
    assert dialog.class_count_badge.text() == "0"


def test_auto_label_dialog_single_preview(sample_images: list[Path], qapp: QApplication) -> None:
    """Test executing single image preview and applying annotations."""
    mock_engine = MagicMock(spec=AutoLabelEngine)
    sample_result = AutoLabelResult(
        image_path=sample_images[0],
        image_width=640,
        image_height=480,
        detections=[
            AutoLabelDetection(
                class_name="truck",
                confidence=0.92,
                box=BoundingBox(0.1, 0.1, 0.4, 0.4),
                color="#ef5350",
                polygon_pixels=[[64.0, 48.0], [256.0, 48.0], [256.0, 192.0], [64.0, 192.0]],
                polygon_normalized=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]],
            )
        ],
        elapsed_seconds=0.15,
    )
    mock_engine.run_preview.return_value = sample_result

    dialog = AutoLabelDialog(
        image_paths=sample_images,
        engine=mock_engine,
    )

    # Trigger preview
    dialog._run_single_preview()
    assert mock_engine.run_preview.call_count == len(sample_images)
    assert "Found" in dialog.result_stats_label.text()

    # Test applying preview
    applied_results = []
    dialog.preview_applied.connect(lambda res: applied_results.append(res))
    with patch("PySide6.QtWidgets.QMessageBox.information"):
        dialog._apply_preview_to_image()
    assert len(applied_results) == 1
    assert applied_results[0].detections[0].class_name == "truck"


def test_auto_label_dialog_cherry_pick_and_randomize(sample_images: list[Path], qapp: QApplication) -> None:
    """Test 4-sample randomization, cherry-pick dialog, and grid view toggle."""
    mock_engine = MagicMock(spec=AutoLabelEngine)
    sample_result = AutoLabelResult(
        image_path=sample_images[0],
        image_width=640,
        image_height=480,
        detections=[],
        elapsed_seconds=0.05,
    )
    mock_engine.run_preview.return_value = sample_result

    dialog = AutoLabelDialog(
        image_paths=sample_images,
        engine=mock_engine,
    )

    # Test randomizing samples
    dialog._randomize_preview_samples()
    assert len(dialog.preview_image_paths) == len(sample_images)

    # Test grid vs single view switching
    dialog._set_preview_view_mode(1)
    assert dialog.preview_views_stack.currentIndex() == 1
    dialog._set_preview_view_mode(0)
    assert dialog.preview_views_stack.currentIndex() == 0

    # Test card zoom
    dialog._on_card_zoomed(sample_images[0])
    assert dialog.preview_views_stack.currentIndex() == 1
    assert dialog.current_image_path == sample_images[0]


def test_main_window_auto_label_action_and_handler(
    sample_images: list[Path], qapp: QApplication
) -> None:
    """Verify MainWindow registers Auto Label action and opens dialog."""
    settings = AppSettings()
    window = MainWindow(settings.fusion, settings.active_learning)

    actions = {
        a.text(): a
        for a in window.findChildren(object)
        if hasattr(a, "text") and hasattr(a, "triggered")
    }

    assert "⚡ Auto Label..." in actions
    auto_label_action = actions["⚡ Auto Label..."]
    assert auto_label_action.shortcut().toString() == "Ctrl+Shift+A"

    # Test opening dialog without crashing
    with patch("app.ui.dialogs.auto_label_dialog.AutoLabelDialog.exec") as mock_exec:
        window._project_documents = {
            p: AnnotationDocument(p, 640, 480) for p in sample_images
        }
        window._open_auto_label_dialog()
        mock_exec.assert_called_once()
