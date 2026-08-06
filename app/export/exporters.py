"""Deterministic dataset exporters for supported annotation formats."""

from __future__ import annotations

import json
import logging
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

from app.services.annotation.domain import TARGET_CLASSES, AnnotationDocument

LOGGER = logging.getLogger(__name__)
CLASS_ORDER = ("motorcycle", "car", "bus", "truck")


class ExportError(RuntimeError):
    """Raised when an export cannot be generated safely."""


class DatasetExporter(ABC):
    """Common exporter contract."""

    @abstractmethod
    def export(self, documents: list[AnnotationDocument], destination: Path) -> Path:
        """Export documents and return the generated artifact path."""


class YoloExporter(DatasetExporter):
    """Export YOLO detection labels and a dataset YAML file."""

    def __init__(self, variant: str = "generic") -> None:
        self.variant = variant

    def export(self, documents: list[AnnotationDocument], destination: Path) -> Path:
        validate_documents(documents)
        destination.mkdir(parents=True, exist_ok=True)
        images = destination / "images"
        labels = destination / "labels"
        images.mkdir(exist_ok=True)
        labels.mkdir(exist_ok=True)
        for document in documents:
            image_target = images / document.image_path.name
            label_target = labels / f"{document.image_path.stem}.txt"
            image_target.write_bytes(document.image_path.read_bytes())
            lines = []
            for annotation in document.annotations:
                center_x, center_y, width, height = annotation.box.to_yolo()
                lines.append(
                    f"{CLASS_ORDER.index(annotation.class_name)} {center_x:.6f} "
                    f"{center_y:.6f} {width:.6f} {height:.6f}"
                )
            label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        yaml_path = destination / "dataset.yaml"
        yaml_path.write_text(
            f"path: {destination.resolve()}\n"
            f"train: images\nval: images\nnames: {list(CLASS_ORDER)}\n",
            encoding="utf-8",
        )
        LOGGER.info(
            "exported %d documents to %s at %s",
            len(documents),
            self.variant,
            destination,
        )
        return yaml_path


class CocoExporter(DatasetExporter):
    """Export COCO detection JSON and source images."""

    def export(self, documents: list[AnnotationDocument], destination: Path) -> Path:
        validate_documents(documents)
        destination.mkdir(parents=True, exist_ok=True)
        payload = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": index + 1, "name": name} for index, name in enumerate(CLASS_ORDER)
            ],
        }
        annotation_id = 1
        for image_id, document in enumerate(documents, start=1):
            payload["images"].append(
                {
                    "id": image_id,
                    "file_name": document.image_path.name,
                    "width": document.image_width,
                    "height": document.image_height,
                }
            )
            for annotation in document.annotations:
                box = annotation.box
                payload["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": CLASS_ORDER.index(annotation.class_name) + 1,
                        "bbox": [
                            box.left * document.image_width,
                            box.top * document.image_height,
                            box.width * document.image_width,
                            box.height * document.image_height,
                        ],
                        "area": box.area * document.image_width * document.image_height,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        image_destination = destination / "images"
        image_destination.mkdir(exist_ok=True)
        for document in documents:
            (image_destination / document.image_path.name).write_bytes(
                document.image_path.read_bytes()
            )
        result = destination / "annotations.json"
        result.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("exported %d documents to COCO at %s", len(documents), result)
        return result


class PascalVocExporter(DatasetExporter):
    """Export one Pascal VOC XML file per image."""

    def export(self, documents: list[AnnotationDocument], destination: Path) -> Path:
        validate_documents(documents)
        destination.mkdir(parents=True, exist_ok=True)
        for document in documents:
            root = Element("annotation")
            SubElement(root, "filename").text = document.image_path.name
            size = SubElement(root, "size")
            SubElement(size, "width").text = str(document.image_width)
            SubElement(size, "height").text = str(document.image_height)
            SubElement(size, "depth").text = "3"
            for annotation in document.annotations:
                obj = SubElement(root, "object")
                SubElement(obj, "name").text = annotation.class_name
                box = SubElement(obj, "bndbox")
                for name, value in (
                    ("xmin", annotation.box.left * document.image_width),
                    ("ymin", annotation.box.top * document.image_height),
                    ("xmax", annotation.box.right * document.image_width),
                    ("ymax", annotation.box.bottom * document.image_height),
                ):
                    SubElement(box, name).text = str(round(value))
            ElementTree(root).write(
                destination / f"{document.image_path.stem}.xml",
                encoding="utf-8",
                xml_declaration=True,
            )
        return destination


class CvatExporter(DatasetExporter):
    """Export CVAT image-format XML with one box per annotation."""

    def export(self, documents: list[AnnotationDocument], destination: Path) -> Path:
        validate_documents(documents)
        destination.mkdir(parents=True, exist_ok=True)
        root = Element("annotations")
        SubElement(root, "version").text = "1.1"
        meta = SubElement(root, "meta")
        SubElement(meta, "task")
        for image_id, document in enumerate(documents):
            image = SubElement(
                root,
                "image",
                {
                    "id": str(image_id),
                    "name": document.image_path.name,
                    "width": str(document.image_width),
                    "height": str(document.image_height),
                },
            )
            for annotation in document.annotations:
                box = annotation.box
                SubElement(
                    image,
                    "box",
                    {
                        "label": annotation.class_name,
                        "occluded": "0",
                        "xtl": str(box.left * document.image_width),
                        "ytl": str(box.top * document.image_height),
                        "xbr": str(box.right * document.image_width),
                        "ybr": str(box.bottom * document.image_height),
                    },
                )
        result = destination / "annotations.xml"
        ElementTree(root).write(result, encoding="utf-8", xml_declaration=True)
        return result


class RoboflowExporter(YoloExporter):
    """Export Roboflow-compatible YOLO directory layout as a ZIP archive."""

    def export(self, documents: list[AnnotationDocument], destination: Path) -> Path:
        staging = destination.with_name(f"{destination.name}-staging")
        super().export(documents, staging)
        destination.parent.mkdir(parents=True, exist_ok=True)
        archive = destination.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            for path in staging.rglob("*"):
                if path.is_file():
                    output.write(path, path.relative_to(staging))
        return archive


def validate_documents(documents: list[AnnotationDocument]) -> None:
    """Reject unsupported classes before writing a partially valid export."""
    for document in documents:
        invalid = {item.class_name for item in document.annotations} - TARGET_CLASSES
        if invalid:
            raise ExportError(f"unsupported classes in {document.image_path}: {sorted(invalid)}")
