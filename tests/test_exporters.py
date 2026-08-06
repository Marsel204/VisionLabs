import cv2
import numpy as np
import yaml

from app.export.exporters import YoloExporter
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
