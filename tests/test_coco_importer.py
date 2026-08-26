import json
from pathlib import Path

from app.export.exporters import CocoExporter
from app.services.dataset.coco_importer import CocoImporter


def write_coco_fixture(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "source-images"
    image_root.mkdir()
    (image_root / "street.jpg").write_bytes(b"image")
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "street.jpg", "width": 100, "height": 100}],
                "categories": [
                    {"id": 1, "name": "car"},
                    {"id": 2, "name": "person"},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 60, 60]},
                    {"id": 2, "image_id": 1, "category_id": 1, "bbox": [12, 12, 55, 55]},
                    {"id": 3, "image_id": 1, "category_id": 2, "bbox": [10, 10, 60, 60]},
                ],
            }
        ),
        encoding="utf-8",
    )
    return annotation_path, image_root


def test_import_copies_images_skips_categories_and_removes_overlaps(tmp_path: Path) -> None:
    annotation_path, image_root = write_coco_fixture(tmp_path)
    result = CocoImporter().import_dataset(
        annotation_path,
        image_root,
        tmp_path / "project",
        remove_overlaps=True,
        overlap_iou_threshold=0.5,
        containment_threshold=0.8,
    )
    report = result.report
    assert report.images_imported == 1
    assert report.annotations_imported == 2
    assert report.unsupported_categories == 1
    assert report.overlapping_removed == 1
    document = result.documents[0]
    assert document.image_path.is_file()
    assert len(document.annotations) == 1
    assert document.annotations[0].class_name == "car"


def test_cleaned_documents_export_to_new_coco_dataset(tmp_path: Path) -> None:
    annotation_path, image_root = write_coco_fixture(tmp_path)
    imported = CocoImporter().import_dataset(
        annotation_path, image_root, tmp_path / "project", remove_overlaps=True
    )
    output = tmp_path / "cleaned"
    result = CocoExporter().export(list(imported.documents), output)
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert len(payload["images"]) == 1
    assert len(payload["annotations"]) == 1
    assert (output / "images" / "street.jpg").is_file()


def test_coco_import_resilience_edge_cases(tmp_path: Path) -> None:
    """Test leading slashes, missing dimensions fallback, aliases, and border boxes."""
    from PIL import Image

    image_root = tmp_path / "edge_images"
    image_root.mkdir()
    test_img = image_root / "sample.jpg"
    Image.new("RGB", (200, 150), color=(100, 150, 200)).save(test_img)

    annotation_path = tmp_path / "edge_coco.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 10,
                        "file_name": "/subfolder/sample.jpg",  # Leading slash and subfolder
                        "width": 0,  # Missing/zero width -> should fall back to PIL dimensions
                        "height": None,
                    }
                ],
                "categories": [
                    {"id": 101, "name": "MotorBike"},  # Alias for motorcycle
                    {"id": 102, "name": "TRUCK"},  # Uppercase
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 10,
                        "category_id": 101,
                        "bbox": [0, 0, 200, 150],  # Full image border bounding box
                    },
                    {
                        "id": 2,
                        "image_id": 10,
                        "category_id": 102,
                        "bbox": [10, 10, 50, 50],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    importer = CocoImporter()
    result = importer.import_dataset(annotation_path, image_root, tmp_path / "edge_project")
    assert result.report.images_imported == 1
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.image_width == 200
    assert doc.image_height == 150
    assert len(doc.annotations) == 2
    classes = {ann.class_name for ann in doc.annotations}
    assert classes == {"motorcycle", "truck"}

