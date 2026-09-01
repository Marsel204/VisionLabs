"""Unit tests for dataset append logic when importing new image folders in MainWindow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.configs.settings import AppSettings
from app.services.annotation.domain import Annotation, AnnotationDocument, BoundingBox
from app.ui.main_window import MainWindow


def test_import_folder_appends_images_without_replacing_existing(
    tmp_path: Path, qapp: object
) -> None:
    """Verify that importing a second folder appends new images and keeps existing ones and annotations."""
    folder1 = tmp_path / "batch1"
    folder1.mkdir()
    img1 = folder1 / "img1.jpg"
    img2 = folder1 / "img2.png"
    Image.new("RGB", (100, 100), color="red").save(img1)
    Image.new("RGB", (100, 100), color="blue").save(img2)

    folder2 = tmp_path / "batch2"
    folder2.mkdir()
    img3 = folder2 / "img3.jpg"
    Image.new("RGB", (100, 100), color="green").save(img3)

    settings = AppSettings()
    window = MainWindow(settings.fusion, settings.active_learning)

    # 1. First import folder 1
    with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(folder1)):
        window._import_folder()

    assert len(window._project_documents) == 2
    assert img1 in window._project_documents
    assert img2 in window._project_documents

    # Add an annotation to img1
    ann = Annotation("car", BoundingBox(0.1, 0.1, 0.5, 0.5))
    doc1_with_ann = window._project_documents[img1].add(ann)
    window._project_documents[img1] = doc1_with_ann
    window._document = doc1_with_ann

    # 2. Now import folder 2
    with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(folder2)):
        window._import_folder()

    # Verify that we now have 3 total images and img1's annotations were NOT lost
    assert len(window._project_documents) == 3
    assert img1 in window._project_documents
    assert img2 in window._project_documents
    assert img3 in window._project_documents

    # Annotations on existing images should be preserved
    assert len(window._project_documents[img1].annotations) == 1
    assert window._project_documents[img1].annotations[0].class_name == "car"

    # Status bar check
    assert "Added 1 new images (3 total in dataset)" in window.statusBar().currentMessage()


def test_set_imported_project_appends_documents(tmp_path: Path, qapp: object) -> None:
    """Verify that importing a COCO/YOLO dataset result appends documents without replacing."""
    from app.services.dataset.yolo_importer import YoloImportReport, YoloImportResult

    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    Image.new("RGB", (100, 100), color="red").save(img1)
    Image.new("RGB", (100, 100), color="blue").save(img2)

    ann = Annotation("bus", BoundingBox(0.2, 0.2, 0.6, 0.6))
    doc1 = AnnotationDocument(img1, 100, 100, (ann,))

    settings = AppSettings()
    window = MainWindow(settings.fusion, settings.active_learning)
    window._project_documents = {img1: doc1}
    window._document = doc1

    doc2 = AnnotationDocument(img2, 100, 100, ())
    report = YoloImportReport(images_found=1, images_imported=1)
    result = YoloImportResult(project_root=tmp_path, documents=(doc2,), report=report)

    window._set_imported_project(result)

    assert len(window._project_documents) == 2
    assert img1 in window._project_documents
    assert img2 in window._project_documents
    assert len(window._project_documents[img1].annotations) == 1
    assert window._project_documents[img1].annotations[0].class_name == "bus"
    assert "Added 1 dataset images (2 total in dataset)" in window.statusBar().currentMessage()
