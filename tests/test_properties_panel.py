"""Unit tests for the simplified Properties panel in MainWindow."""

from __future__ import annotations

from pathlib import Path
from PIL import Image
import pytest

from app.configs.settings import AppSettings
from app.services.annotation.domain import Annotation, AnnotationDocument, BoundingBox
from app.services.annotation.history import AnnotationHistory
from app.ui.main_window import MainWindow


def test_properties_panel_initialization(qapp: object) -> None:
    """Verify that the simplified Properties panel groups and buttons are initialized."""
    settings = AppSettings()
    window = MainWindow(settings.fusion, settings.active_learning)

    assert hasattr(window, "_tools_group")
    assert hasattr(window, "_selection_group")
    assert hasattr(window, "_review_group")
    assert hasattr(window, "_crop_group")
    assert hasattr(window, "_project_group")

    assert window._draw_tool_btn.isChecked() is True
    assert window._pan_tool_btn.isChecked() is False
    assert window._selection_info_label.text() == "No annotation selected"
    assert not window._occluded_btn.isEnabled()
    assert not window._truncated_btn.isEnabled()
    assert not window._refine_sam2_btn.isEnabled()
    assert not window._occluded_btn.isChecked()
    assert not window._truncated_btn.isChecked()


def test_selection_properties_synchronization(tmp_path: Path, qapp: object) -> None:
    """Verify that selecting an annotation updates the Properties panel controls."""
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (100, 100), color="white").save(image_path)

    settings = AppSettings()
    window = MainWindow(settings.fusion, settings.active_learning)

    ann1 = Annotation("car", BoundingBox(0.1, 0.1, 0.5, 0.5), occluded=True, truncated=False)
    ann2 = Annotation("bus", BoundingBox(0.2, 0.2, 0.6, 0.6), occluded=False, truncated=True)
    doc = AnnotationDocument(image_path, 100, 100, [ann1, ann2])
    window._document = doc
    window._history = AnnotationHistory(doc)
    window._project_documents[image_path] = doc

    # Select ann1
    window._select_annotation(ann1.annotation_id)
    assert "CAR" in window._selection_info_label.text()
    assert window._occluded_btn.isEnabled()
    assert window._truncated_btn.isEnabled()
    assert window._refine_sam2_btn.isEnabled()
    assert window._occluded_btn.isChecked() is True
    assert window._truncated_btn.isChecked() is False

    # Toggle occluded
    window._toggle_selected_occluded()
    assert window._occluded_btn.isChecked() is False
    assert window._document.annotations[0].occluded is False

    # Toggle truncated
    window._toggle_selected_truncated()
    assert window._truncated_btn.isChecked() is True
    assert window._document.annotations[0].truncated is True

    # Select ann2
    window._select_annotation(ann2.annotation_id)
    assert "BUS" in window._selection_info_label.text()
    assert window._occluded_btn.isChecked() is False
    assert window._truncated_btn.isChecked() is True

    # Deselect
    window._select_annotation(None)
    assert window._selection_info_label.text() == "No annotation selected"
    assert not window._occluded_btn.isEnabled()
    assert not window._truncated_btn.isEnabled()
    assert not window._refine_sam2_btn.isEnabled()
