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


def test_auto_label_single_picture_apply_flow(sample_images: list[Path], qapp: QApplication) -> None:
    """Test full workflow for 1 picture: preview -> apply -> MainWindow document and canvas updated."""
    settings = AppSettings()
    window = MainWindow(settings.fusion, settings.active_learning)
    single_image = sample_images[0]
    initial_doc = AnnotationDocument(single_image, 640, 480)
    window._project_documents = {single_image: initial_doc}
    window._load_image(single_image)

    mock_engine = MagicMock(spec=AutoLabelEngine)
    sample_result = AutoLabelResult(
        image_path=single_image,
        image_width=640,
        image_height=480,
        detections=[
            AutoLabelDetection(
                class_name="motorcycle",
                confidence=0.88,
                box=BoundingBox(0.2, 0.2, 0.6, 0.6),
                color="#ff9800",
            ),
            # Invalid out of bounds / degenerate box that should be safely handled
            AutoLabelDetection(
                class_name="motorcycle",
                confidence=0.75,
                box=BoundingBox(0.1, 0.1, 0.15, 0.15),
                color="#ff9800",
            ),
        ],
        elapsed_seconds=0.10,
    )
    mock_engine.run_preview.return_value = sample_result

    dialog = AutoLabelDialog(
        image_paths=[single_image],
        current_image_path=single_image,
        engine=mock_engine,
        ground_truth=window._project_documents,
        parent=window,
    )
    dialog.batch_completed.connect(window._on_auto_label_batch_completed)

    # 1. Preview
    dialog._run_single_preview()
    assert dialog.apply_btn.text() == "✔ Apply to Current Image"

    # 2. Apply annotation for the 1 picture
    with patch("PySide6.QtWidgets.QMessageBox.information"):
        dialog._apply_preview_to_image()

    # 3. Verify MainWindow state
    assert len(window._document.annotations) == 2
    assert window._document.annotations[0].class_name == "motorcycle"
    assert single_image in window._project_documents
    assert len(window._project_documents[single_image].annotations) == 2

    # Verify history undo works cleanly
    assert window._history.can_undo
    window._undo_annotation_edit()
    assert len(window._document.annotations) == 0


def test_image_browser_fast_path_refresh(sample_images: list[Path], qapp: QApplication) -> None:
    """Verify _refresh_image_browser_order uses fast-path in-place updates when ordering is unchanged."""
    settings = AppSettings()
    window = MainWindow(settings.fusion, settings.active_learning)
    single_image = sample_images[0]
    window._project_documents = {single_image: AnnotationDocument(single_image, 640, 480)}
    window._load_image(single_image)
    window.image_browser.set_paths([single_image], {single_image: 0})

    with patch.object(window.image_browser, "set_paths") as mock_set_paths:
        with patch.object(window.image_browser, "update_annotation_count") as mock_update_count:
            window._refresh_image_browser_order(preserve_current=True)
            # Ordering is unchanged, should call update_annotation_count and NOT rebuild list via set_paths
            mock_update_count.assert_called_once_with(single_image, 0)
            mock_set_paths.assert_not_called()


def test_auto_label_apply_idempotent_and_empty_handling(
    sample_images: list[Path], qapp: QApplication
) -> None:
    """Verify applying preview multiple times does not create duplicates and handles empty preview safely."""
    dialog = AutoLabelDialog(image_paths=[sample_images[0]])
    dialog._latest_results = {}
    dialog._latest_result = None

    # Apply with no preview ready
    with patch("PySide6.QtWidgets.QMessageBox.information") as mock_info:
        dialog._apply_preview_to_image()
        mock_info.assert_called_once()

    # Now set preview with detections and apply twice
    sample_result = AutoLabelResult(
        image_path=sample_images[0],
        image_width=640,
        image_height=480,
        detections=[
            AutoLabelDetection(
                class_name="car",
                confidence=0.95,
                box=BoundingBox(0.1, 0.1, 0.5, 0.5),
                color="#29b6f6",
            )
        ],
        elapsed_seconds=0.05,
    )
    dialog._latest_results = {sample_images[0]: sample_result}
    dialog._latest_result = sample_result

    with patch("PySide6.QtWidgets.QMessageBox.information"):
        dialog._apply_preview_to_image()
        assert len(dialog.ground_truth[sample_images[0]].annotations) == 1

        # Second apply of same preview does not duplicate the box
        dialog._apply_preview_to_image()
        assert len(dialog.ground_truth[sample_images[0]].annotations) == 1


def test_main_window_delete_picture_from_database(
    sample_images: list[Path], qapp: QApplication
) -> None:
    """Verify deleting picture from database updates project documents, disk, and canvas."""
    from PySide6.QtWidgets import QMessageBox

    settings = AppSettings()
    window = MainWindow(settings.fusion, settings.active_learning)
    img1, img2 = sample_images[0], sample_images[1]
    window._project_documents = {
        img1: AnnotationDocument(img1, 640, 480),
        img2: AnnotationDocument(img2, 640, 480),
    }
    window.image_browser.set_paths([img1, img2], {img1: 0, img2: 0})
    window._load_image(img1)

    # 1. Cancel deletion via confirmation dialog
    with patch("PySide6.QtWidgets.QMessageBox.question", return_value=QMessageBox.StandardButton.No):
        result = window._delete_picture_from_database(img1, confirm=True)
        assert result is False
        assert img1 in window._project_documents
        assert img1.exists()

    # 2. Confirm deletion of img1
    with patch("PySide6.QtWidgets.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
        result = window._delete_picture_from_database(img1, confirm=True)
        assert result is True
        assert img1 not in window._project_documents
        assert not img1.exists()
        assert window.image_browser.count() == 1
        # MainWindow switches to remaining image
        assert window._document is not None
        assert window._document.image_path == img2

    # 3. Delete the last remaining picture
    result = window._delete_picture_from_database(img2, confirm=False)
    assert result is True
    assert img2 not in window._project_documents
    assert not img2.exists()
    assert window.image_browser.count() == 0
    assert window._document is None
    assert window._history is None
    assert window.canvas._document is None


def test_annotation_canvas_clear(sample_images: list[Path], qapp: QApplication) -> None:
    """Verify AnnotationCanvas.clear() resets scene, document, and items."""
    from app.ui.canvas.annotation_canvas import AnnotationCanvas

    canvas = AnnotationCanvas()
    img = sample_images[0]
    doc = AnnotationDocument(img, 640, 480)
    canvas.set_document(doc)
    assert canvas._document is not None
    assert canvas._image_item is not None

    canvas.clear()
    assert canvas._document is None
    assert canvas._image_item is None
    assert canvas._selected is None
    assert len(canvas._annotation_items) == 0


def test_auto_label_dialog_multi_yolo_ui(sample_images: list[Path], qapp: QApplication) -> None:
    """Test managing 2-3 simultaneous YOLO models in AutoLabelDialog."""
    mock_engine = MagicMock()
    dialog = AutoLabelDialog(image_paths=sample_images, engine=mock_engine)

    # Initial state: 1 model (yolo11n.pt)
    assert dialog._active_yolo_models == ["yolo11n.pt"]
    assert "yolo11n.pt" in dialog.yolo_weights_btn.text()

    # Add second model (yolov8m.pt)
    dialog._add_yolo_model("yolov8m.pt")
    assert len(dialog._active_yolo_models) == 2
    assert dialog._active_yolo_models == ["yolo11n.pt", "yolov8m.pt"]
    assert "2 YOLO" in dialog.yolo_weights_btn.text()
    assert dialog.yolo_chk.isChecked()

    # Add third model (yolo11s.pt)
    dialog._add_yolo_model("yolo11s.pt")
    assert len(dialog._active_yolo_models) == 3
    assert "3 YOLO" in dialog.yolo_weights_btn.text()

    # Attempt adding fourth model (should be capped at 3)
    dialog._add_yolo_model("yolov8x.pt")
    assert len(dialog._active_yolo_models) == 3

    # Check that _get_current_config reflects all 3 active YOLO models
    config = dialog._get_current_config()
    assert len(config.yolo_models) == 3
    assert config.yolo_models == ["yolo11n.pt", "yolov8m.pt", "yolo11s.pt"]

    # Remove second model
    dialog._remove_yolo_model(1)
    assert len(dialog._active_yolo_models) == 2
    assert dialog._active_yolo_models == ["yolo11n.pt", "yolo11s.pt"]

    # Reset models to default
    dialog._reset_yolo_models()
    assert dialog._active_yolo_models == ["yolo11n.pt"]
    assert "yolo11n.pt" in dialog.yolo_weights_btn.text()






def test_auto_label_dialog_batch_preserves_ground_truth(
    sample_images: list[Path], qapp: QApplication
) -> None:
    from PySide6.QtWidgets import QMessageBox
    from app.services.annotation.domain import Annotation

    img = sample_images[0]
    existing_ann = Annotation("car", BoundingBox(0.1, 0.1, 0.4, 0.4))
    initial_doc = AnnotationDocument(img, 640, 480, (existing_ann,))
    ground_truth = {img: initial_doc}

    engine = AutoLabelEngine()
    engine.run_preview = MagicMock(return_value=AutoLabelResult(
        image_path=img,
        image_width=640,
        image_height=480,
        detections=[AutoLabelDetection(class_name="truck", confidence=0.88, box=BoundingBox(0.5, 0.5, 0.8, 0.8))],
    ))

    dialog = AutoLabelDialog([img], ground_truth=ground_truth, engine=engine)
    with patch("PySide6.QtWidgets.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes), \
         patch("PySide6.QtWidgets.QMessageBox.information"):
        dialog._run_batch_auto_label(wait=True)

    assert img in dialog.ground_truth
    result_doc = dialog.ground_truth[img]
    assert len(result_doc.annotations) == 2
    classes = {ann.class_name for ann in result_doc.annotations}
    assert classes == {"car", "truck"}
