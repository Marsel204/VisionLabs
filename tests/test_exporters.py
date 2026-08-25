import cv2
import numpy as np
import yaml

from app.export.exporters import YoloExporter, split_documents
from app.services.annotation.domain import AnnotationDocument


def test_yolo_exporter_writes_standard_layout_and_variant(tmp_path) -> None:
    image_path = tmp_path / "image.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))
    source_document = AnnotationDocument(image_path, 20, 20, ())
    destination = tmp_path / "yolov11"

    result = YoloExporter(variant="yolov11").export(
        [source_document],
        destination,
    )

    assert result == destination / "dataset.yaml"
    assert (destination / "images" / source_document.image_path.name).is_file()
    assert (destination / "labels" / f"{source_document.image_path.stem}.txt").is_file()
    assert yaml.safe_load(result.read_text(encoding="utf-8"))["names"] == [
        "motorcycle",
        "car",
        "bus",
        "truck",
    ]


def test_split_export_is_seeded_and_writes_split_layout(tmp_path) -> None:
    documents = []
    for index in range(10):
        image_path = tmp_path / f"image-{index}.jpg"
        assert cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))
        documents.append(AnnotationDocument(image_path, 20, 20, ()))

    first = split_documents(documents, 0.6, 0.2, 0.2, seed=7)
    second = split_documents(documents, 0.6, 0.2, 0.2, seed=7)
    assert {
        name: [item.image_path.name for item in values]
        for name, values in first.items()
    } == {
        name: [item.image_path.name for item in values]
        for name, values in second.items()
    }

    destination = tmp_path / "split"
    result = YoloExporter(splits=first).export(documents, destination)
    assert result == destination / "dataset.yaml"
    assert (destination / "images" / "train").is_dir()
    assert (destination / "images" / "val").is_dir()
    assert (destination / "images" / "test").is_dir()
    assert yaml.safe_load(result.read_text(encoding="utf-8"))["test"] == "images/test"
