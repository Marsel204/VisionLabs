from pathlib import Path

import cv2
import numpy as np

from app.services.annotation.domain import Annotation, AnnotationDocument, AnnotationSource, BoundingBox
from app.services.inference.dense_motorcycle import DenseInferenceConfig, DenseMotorcycleInference


class _Value:
    def __init__(self, value):  # type: ignore[no-untyped-def]
        self.value = value

    def __getitem__(self, _index):  # type: ignore[no-untyped-def]
        return self

    def __int__(self) -> int:
        return int(self.value)

    def __float__(self) -> float:
        return float(self.value)

    def tolist(self):  # type: ignore[no-untyped-def]
        return self.value


class _Box:
    def __init__(self) -> None:
        self.cls = _Value(0)
        self.conf = _Value(0.8)
        self.xyxy = _Value([10.0, 10.0, 50.0, 50.0])


class _Result:
    boxes = [_Box()]


class _Yolo:
    names = {0: "motorcycle"}

    def __call__(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return [_Result()]


def test_dense_yolo_inference_keeps_objects_from_multiple_scales(tmp_path: Path) -> None:
    image_path = tmp_path / "dense.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((640, 640, 3), dtype=np.uint8))
    document = AnnotationDocument(image_path, 640, 640, ())
    inference = DenseMotorcycleInference(
        None,
        None,
        _Yolo(),
        DenseInferenceConfig(tile_sizes=(512, 320), nms_iou=0.45),
    )

    yolo, dino = inference.predict(document, "motorcycle", use_yolo=True)

    assert dino == []
    assert len(yolo) >= 5
    assert {item.class_name for item in yolo} == {"motorcycle"}


def test_dense_rules_reject_weak_large_vehicle_classes() -> None:
    inference = DenseMotorcycleInference(None, None, None)
    small_bus = Annotation(
        "bus",
        BoundingBox(0.1, 0.1, 0.2, 0.2),
        confidence=0.9,
        source=AnnotationSource.YOLO,
    )
    weak_truck = Annotation(
        "truck",
        BoundingBox(0.1, 0.1, 0.5, 0.5),
        confidence=0.2,
        source=AnnotationSource.YOLO,
    )
    assert not inference._accept(small_bus)
    assert not inference._accept(weak_truck)
