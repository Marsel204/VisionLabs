"""Unit tests for Florence-2 VLM integration into MainWindow GUI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from app.configs.settings import AppSettings
from app.services.annotation.domain import Annotation, AnnotationDocument, AnnotationSource, BoundingBox
from app.services.annotation.history import AnnotationHistory
from app.ui.main_window import MainWindow
from src.vlm_helper import Florence2VLM


def test_main_window_vlm_actions_registration(qapp: object) -> None:
    """Verify that MainWindow registers the Florence-2 VLM actions and shortcuts."""
    settings = AppSettings()
    window = MainWindow(settings.fusion, settings.active_learning)

    actions = {
        a.text(): a
        for a in window.findChildren(object)
        if hasattr(a, "text") and hasattr(a, "triggered")
    }

    assert "Load Florence-2 VLM" in actions
    assert "VLM Auto-Annotate" in actions
    assert "VLM Auto-Filter (DINO+SAM)" in actions

    vlm_annotate_action = actions["VLM Auto-Annotate"]
    assert vlm_annotate_action.shortcut().toString() == "Ctrl+Shift+V"


def test_main_window_toggle_vlm_filtering(qapp: object) -> None:
    """Verify toggling VLM auto-filter flag and status bar message."""
    settings = AppSettings()
    window = MainWindow(settings.fusion, settings.active_learning)

    assert window._vlm_filter_enabled is True
    window._toggle_vlm_filtering(False)
    assert window._vlm_filter_enabled is False
    assert "disabled" in window.statusBar().currentMessage()

    window._toggle_vlm_filtering(True)
    assert window._vlm_filter_enabled is True
    assert "enabled" in window.statusBar().currentMessage()


def test_main_window_vlm_auto_annotate(tmp_path: Path, qapp: object) -> None:
    """Verify that _vlm_auto_annotate adds new annotations from Florence-2 OD detections."""
    img_path = tmp_path / "scene.jpg"
    Image.new("RGB", (640, 480), color="white").save(img_path)

    doc = AnnotationDocument(img_path, 640, 480, annotations=())

    settings = AppSettings()
    window = MainWindow(settings.fusion, settings.active_learning)
    window._document = doc
    window._history = AnnotationHistory(doc)

    # Mock VLM helper with detect_objects returning <OD> detections
    mock_vlm = MagicMock(spec=Florence2VLM)
    mock_vlm.detect_objects.return_value = [
        {"label": "car", "box": [64.0, 48.0, 320.0, 240.0]},
        {"label": "motorcycle", "box": [400.0, 100.0, 600.0, 350.0]},
    ]
    window._vlm_helper = mock_vlm

    window._vlm_auto_annotate()

    assert window._document is not None
    assert len(window._document.annotations) == 2
    class_names = {ann.class_name for ann in window._document.annotations}
    assert class_names == {"car", "motorcycle"}
    # Verify provenance is FLORENCE2
    for ann in window._document.annotations:
        assert ann.source == AnnotationSource.FLORENCE2
        assert ann.confidence is None
    assert "added 2" in window.statusBar().currentMessage()


def test_main_window_vlm_auto_annotate_skips_duplicates(tmp_path: Path, qapp: object) -> None:
    """Verify that _vlm_auto_annotate skips detections overlapping with existing annotations."""
    img_path = tmp_path / "scene.jpg"
    Image.new("RGB", (640, 480), color="white").save(img_path)

    # Pre-existing annotation that overlaps with one of the detections
    existing_ann = Annotation("car", BoundingBox(0.1, 0.1, 0.5, 0.5))
    doc = AnnotationDocument(img_path, 640, 480, annotations=(existing_ann,))

    settings = AppSettings()
    window = MainWindow(settings.fusion, settings.active_learning)
    window._document = doc
    window._history = AnnotationHistory(doc)

    mock_vlm = MagicMock(spec=Florence2VLM)
    mock_vlm.detect_objects.return_value = [
        # This overlaps with existing_ann (same normalized coords)
        {"label": "car", "box": [64.0, 48.0, 320.0, 240.0]},
        # This does not overlap
        {"label": "motorcycle", "box": [400.0, 100.0, 600.0, 350.0]},
    ]
    window._vlm_helper = mock_vlm

    window._vlm_auto_annotate()

    assert window._document is not None
    # Original + 1 new (motorcycle), the overlapping car should be skipped
    assert len(window._document.annotations) == 2
    class_names = [ann.class_name for ann in window._document.annotations]
    assert class_names.count("car") == 1  # Only the original
    assert class_names.count("motorcycle") == 1
    assert "added 1" in window.statusBar().currentMessage()


def test_main_window_add_grounding_results(tmp_path: Path, qapp: object) -> None:
    """Verify that _add_grounding_results correctly parses boxes without NameError."""
    import torch

    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (640, 480), color="white").save(img_path)
    doc = AnnotationDocument(img_path, 640, 480, annotations=())

    settings = AppSettings()
    window = MainWindow(settings.fusion, settings.active_learning)
    window._document = doc
    window._history = AnnotationHistory(doc)

    results = {
        "boxes": torch.tensor([[10.0, 20.0, 50.0, 80.0]]),
        "scores": torch.tensor([0.92]),
        "labels": ["motorcycle"],
    }
    added = window._add_grounding_results(results, image_width=640, image_height=480)
    assert added == 1
    assert len(window._grounding_detections) == 1
    assert window._grounding_detections[0].class_name == "motorcycle"
    assert len(window._document.annotations) == 1


def test_main_window_vlm_check_active_image(tmp_path: Path, qapp: object) -> None:
    """Verify that _vlm_check_active_image flags mismatched labels and verifies matching labels."""
    img_path = tmp_path / "scene.jpg"
    Image.new("RGB", (640, 480), color="white").save(img_path)

    # 1 correct car annotation, 1 false positive truck (which is actually a sign)
    ann_car = Annotation("car", BoundingBox(0.1, 0.1, 0.4, 0.4), confidence=0.8)
    ann_fake_truck = Annotation("truck", BoundingBox(0.5, 0.5, 0.8, 0.8), confidence=0.7)

    doc = AnnotationDocument(img_path, 640, 480, annotations=(ann_car, ann_fake_truck))

    settings = AppSettings()
    window = MainWindow(settings.fusion, settings.active_learning)
    window._document = doc
    window._history = AnnotationHistory(doc)

    mock_vlm = MagicMock(spec=Florence2VLM)
    # VLM captions: car -> "a blue sedan car on the road", fake truck -> "a billboard sign by the road"
    mock_vlm.generate_captions_batch.return_value = [
        "a blue sedan car on the road",
        "a billboard sign by the road",
    ]
    window._vlm_helper = mock_vlm

    window._vlm_check_active_image()

    assert window._document is not None
    # Car should be verified
    v_car = window._vlm_verifications[ann_car.annotation_id]
    assert v_car["matched"] is True

    # Fake truck should be flagged as mismatch
    v_truck = window._vlm_verifications[ann_fake_truck.annotation_id]
    assert v_truck["matched"] is False

    # Check that the fake truck confidence was lowered and marked PENDING for review
    updated_truck = next(a for a in window._document.annotations if a.annotation_id == ann_fake_truck.annotation_id)
    assert updated_truck.confidence == 0.10
    assert "1 flagged as mismatches" in window.statusBar().currentMessage()


def test_vlm_batch_verify_task(tmp_path: Path, qapp: object) -> None:
    """Verify _VlmBatchVerifyTask across multiple documents."""
    from app.ui.main_window import _VlmBatchVerifyTask

    img1_path = tmp_path / "img1.jpg"
    img2_path = tmp_path / "img2.jpg"
    Image.new("RGB", (640, 480), color="white").save(img1_path)
    Image.new("RGB", (640, 480), color="white").save(img2_path)

    ann1 = Annotation("motorcycle", BoundingBox(0.1, 0.1, 0.3, 0.3), confidence=0.85)
    ann2 = Annotation("bus", BoundingBox(0.4, 0.4, 0.9, 0.9), confidence=0.60)

    doc1 = AnnotationDocument(img1_path, 640, 480, annotations=(ann1,))
    doc2 = AnnotationDocument(img2_path, 640, 480, annotations=(ann2,))

    mock_vlm = MagicMock(spec=Florence2VLM)
    # Doc 1 -> "a scooter parked", Doc 2 -> "a large tree in the park" (mismatch)
    mock_vlm.generate_captions_batch.side_effect = [
        ["a scooter parked on the sidewalk"],
        ["a large tree in the park"],
    ]

    # Test full mode (suspicious_only=False)
    task_full = _VlmBatchVerifyTask(
        documents=[doc1, doc2],
        vlm_helper=mock_vlm,
        confidence_threshold=0.25,
        suspicious_only=False,
    )

    completed_payload = None

    def _on_completed(payload: tuple[object, ...]) -> None:
        nonlocal completed_payload
        completed_payload = payload

    task_full.signals.completed.connect(_on_completed)
    task_full.run()

    assert completed_payload is not None
    results, verifications, total_checked, total_flagged = completed_payload
    assert total_checked == 2
    assert total_flagged == 1
    assert verifications[ann1.annotation_id]["matched"] is True
    assert verifications[ann2.annotation_id]["matched"] is False

    # Test suspicious-only mode where high-conf human box is skipped
    ann_low_conf = Annotation("bus", BoundingBox(0.4, 0.4, 0.9, 0.9), confidence=0.15, source=AnnotationSource.YOLO)
    doc_suspicious = AnnotationDocument(img2_path, 640, 480, annotations=(ann_low_conf,))
    mock_vlm.generate_captions_batch.side_effect = [["a large tree in the park"]]

    task_suspicious = _VlmBatchVerifyTask(
        documents=[doc_suspicious],
        vlm_helper=mock_vlm,
        confidence_threshold=0.25,
        suspicious_only=True,
    )
    completed_suspicious = None
    task_suspicious.signals.completed.connect(lambda p: setattr(task_suspicious, "res", p))
    task_suspicious.run()
    assert task_suspicious.res[2] == 1  # 1 checked
    assert task_suspicious.res[3] == 1  # 1 flagged


