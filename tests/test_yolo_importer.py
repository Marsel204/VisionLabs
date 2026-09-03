import zipfile
from pathlib import Path
import cv2
import numpy as np
import pytest
import yaml

from app.export.exporters import RoboflowExporter, YoloExporter
from app.services.dataset.yolo_importer import (
    YoloImportError,
    YoloImporter,
)


def create_dummy_image(path: Path, width: int = 100, height: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_yolo_importer_roboflow_layout(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "roboflow_dataset"
    dataset_dir.mkdir()

    # Create train, val, test images and labels
    train_img = dataset_dir / "train" / "images" / "img1.jpg"
    train_lbl = dataset_dir / "train" / "labels" / "img1.txt"
    create_dummy_image(train_img, 200, 200)
    train_lbl.parent.mkdir(parents=True, exist_ok=True)
    # Box 1: class 0 (motorcycle), center=(0.5, 0.5), w=0.4, h=0.4
    # Box 2: class 0 (overlapping motorcycle duplicate)
    # Box 3: class 4 (person -> unsupported)
    train_lbl.write_text(
        "0 0.5 0.5 0.4 0.4\n"
        "0 0.51 0.51 0.39 0.39\n"
        "4 0.1 0.1 0.1 0.1\n",
        encoding="utf-8",
    )

    val_img = dataset_dir / "valid" / "images" / "img2.jpg"
    val_lbl = dataset_dir / "valid" / "labels" / "img2.txt"
    create_dummy_image(val_img, 300, 150)
    val_lbl.parent.mkdir(parents=True, exist_ok=True)
    # Box: class 1 (car)
    val_lbl.write_text("1 0.3 0.4 0.2 0.3\n", encoding="utf-8")

    # Background image with no label file
    test_img = dataset_dir / "test" / "images" / "img3.jpg"
    create_dummy_image(test_img, 100, 100)

    yaml_path = dataset_dir / "data.yaml"
    yaml_content = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 5,
        "names": ["MotorBike", "Car", "Bus", "Truck", "Person"],
    }
    yaml_path.write_text(yaml.dump(yaml_content), encoding="utf-8")

    destination = tmp_path / "project_dest"
    importer = YoloImporter()
    result = importer.import_dataset(
        yaml_path,
        destination,
        remove_overlaps=True,
        overlap_iou_threshold=0.5,
    )

    assert result.project_root == destination
    assert len(result.documents) == 3
    report = result.report
    assert report.images_imported == 3
    assert report.images_found == 3
    assert report.unsupported_categories == 1
    assert report.overlapping_removed == 1

    # Check imported documents
    doc_map = {doc.image_path.name: doc for doc in result.documents}
    assert "img1.jpg" in doc_map
    doc1 = doc_map["img1.jpg"]
    assert doc1.image_width == 200
    assert doc1.image_height == 200
    assert len(doc1.annotations) == 1
    assert doc1.annotations[0].class_name == "motorcycle"

    doc2 = doc_map["img2.jpg"]
    assert doc2.image_width == 300
    assert doc2.image_height == 150
    assert len(doc2.annotations) == 1
    assert doc2.annotations[0].class_name == "car"

    doc3 = doc_map["img3.jpg"]
    assert len(doc3.annotations) == 0


def test_yolo_importer_dict_names_and_ultralytics_layout(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ultralytics_dataset"
    dataset_dir.mkdir()

    img_path = dataset_dir / "images" / "train" / "street.png"
    lbl_path = dataset_dir / "labels" / "train" / "street.txt"
    create_dummy_image(img_path, 640, 480)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    # class 2 (bus) and class 3 (truck)
    lbl_path.write_text(
        "2 0.25 0.3 0.2 0.4\n"
        "3 0.7 0.6 0.3 0.5\n",
        encoding="utf-8",
    )

    yaml_path = dataset_dir / "data.yaml"
    yaml_content = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/train",
        "names": {
            0: "motorcycle",
            1: "car",
            2: "bus",
            3: "truck",
        },
    }
    yaml_path.write_text(yaml.dump(yaml_content), encoding="utf-8")

    importer = YoloImporter()
    dest = tmp_path / "yolo_project"
    result = importer.import_dataset(yaml_path, dest)

    assert result.report.images_imported == 1
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert len(doc.annotations) == 2
    classes = {ann.class_name for ann in doc.annotations}
    assert classes == {"bus", "truck"}


def test_yolo_importer_errors_on_missing_or_empty_yaml(tmp_path: Path) -> None:
    importer = YoloImporter()
    with pytest.raises(YoloImportError):
        importer.import_dataset(tmp_path / "nonexistent.yaml", tmp_path / "dest")

    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("not a dict\n", encoding="utf-8")
    with pytest.raises(YoloImportError):
        importer.import_dataset(empty_yaml, tmp_path / "dest")


def test_colab_zip_exporter_produces_valid_archive(tmp_path: Path) -> None:
    # 1. Create source dataset and import it
    dataset_dir = tmp_path / "src_ds"
    img_path = dataset_dir / "images" / "car1.jpg"
    lbl_path = dataset_dir / "labels" / "car1.txt"
    create_dummy_image(img_path, 100, 100)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(yaml.dump({"names": ["car"], "train": "images"}), encoding="utf-8")

    imported = YoloImporter().import_dataset(yaml_path, tmp_path / "imported")
    assert len(imported.documents) == 1

    # 2. Export using RoboflowExporter (which creates a Colab-compatible ZIP)
    zip_dest = tmp_path / "exported_colab"
    archive_path = RoboflowExporter().export(list(imported.documents), zip_dest)
    assert archive_path.is_file()
    assert archive_path.suffix == ".zip"

    # 3. Verify ZIP contents: data.yaml must exist and have valid entries
    with zipfile.ZipFile(archive_path, "r") as zf:
        namelist = zf.namelist()
        assert "data.yaml" in namelist
        assert "dataset.yaml" in namelist
        assert "images/car1.jpg" in namelist
        assert "labels/car1.txt" in namelist

        data_yaml_str = zf.read("data.yaml").decode("utf-8")
        parsed_yaml = yaml.safe_load(data_yaml_str)
        assert parsed_yaml["path"] == "."
        assert "train" in parsed_yaml
        assert parsed_yaml["names"] == ["motorcycle", "car", "bus", "truck"]
