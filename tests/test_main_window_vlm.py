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
