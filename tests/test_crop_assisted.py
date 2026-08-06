from pathlib import Path

import cv2
import numpy as np

from app.services.annotation.domain import Annotation, AnnotationDocument, BoundingBox
from app.services.crop_assisted import CropGenerator, CropMerger


def source_document(tmp_path: Path) -> AnnotationDocument:
    image_path = tmp_path / "source.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((800, 800, 3), dtype=np.uint8))
    return AnnotationDocument(
        image_path,
        800,
        800,
        (Annotation("car", BoundingBox(0.4, 0.4, 0.5, 0.5)),),
    )


def test_crop_generator_creates_overlapping_tiles_and_maps_existing_box(tmp_path: Path) -> None:
    document = source_document(tmp_path)
    session = CropGenerator().generate(document, tmp_path / "crops", tile_size=400, overlap=0.2)
    assert len(session.regions) == 9
    assert all(region.image_path.is_file() for region in session.regions)
    assert sum(len(item.annotations) for item in session.documents) == 1


def test_crop_merger_maps_local_box_to_original_coordinates(tmp_path: Path) -> None:
    document = source_document(tmp_path)
    session = CropGenerator().generate(document, tmp_path / "crops", tile_size=400, overlap=0.2)
    region = session.regions[0]
    local = AnnotationDocument(
        region.image_path,
        region.width,
        region.height,
        (Annotation("motorcycle", BoundingBox(0.1, 0.1, 0.2, 0.2)),),
    )
    documents = tuple(
        local if index == 0 else item for index, item in enumerate(session.documents)
    )
    merged = CropMerger().merge(document, session.regions, documents, remove_duplicates=False)
    added = next(item for item in merged.annotations if item.class_name == "motorcycle")
    assert added.box.left == 0.05
    assert added.box.top == 0.05


def test_crop_generator_divides_small_images_into_four_crops(tmp_path: Path) -> None:
    image_path = tmp_path / "small.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((640, 640, 3), dtype=np.uint8))
    document = AnnotationDocument(image_path, 640, 640, ())

    session = CropGenerator().generate(document, tmp_path / "small-crops")

    assert len(session.regions) == 4
    assert {(region.width, region.height) for region in session.regions} == {(320, 320)}
