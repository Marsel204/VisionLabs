"""Deterministic dataset exporters for supported annotation formats."""

from __future__ import annotations

import json
import logging
import random
import shutil
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

from app.services.annotation.domain import TARGET_CLASSES, AnnotationDocument

LOGGER = logging.getLogger(__name__)
CLASS_ORDER = ("motorcycle", "car", "bus", "truck")
SPLIT_NAMES = ("train", "val", "test")


class ExportError(RuntimeError):
    """Raised when an export cannot be generated safely."""


class DatasetExporter(ABC):
    """Common exporter contract."""

    @abstractmethod
    def export(self, documents: list[AnnotationDocument], destination: Path) -> Path:
        """Export documents and return the generated artifact path."""


def split_documents(
    documents: list[AnnotationDocument],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[AnnotationDocument]]:
    """Assign documents to reproducible train, validation, and test splits."""
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0.0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split ratios must be non-negative and sum to 1")
    shuffled = sorted(documents, key=lambda item: str(item.image_path))
    random.Random(seed).shuffle(shuffled)
    train_end = round(len(shuffled) * train_ratio)
    val_end = train_end + round(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


class YoloExporter(DatasetExporter):
    """Export YOLO detection labels and a dataset YAML file."""

    def __init__(
        self,
        variant: str = "generic",
        splits: dict[str, list[AnnotationDocument]] | None = None,
    ) -> None:
        self.variant = variant
        self.splits = splits

    def export(self, documents: list[AnnotationDocument], destination: Path) -> Path:
        validate_documents(documents)
        destination.mkdir(parents=True, exist_ok=True)
        split_documents_map = self.splits or {"": documents}
        metadata: dict[str, list[dict[str, object]]] = {}
        import concurrent.futures
        import os

        max_workers = min(32, max(4, (os.cpu_count() or 4) * 2))

        export_tasks = []
        for split, split_documents_list in split_documents_map.items():
            images = destination / "images" / split if split else destination / "images"
            labels = destination / "labels" / split if split else destination / "labels"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            for document in split_documents_list:
                export_tasks.append((split, document, images, labels))

        def _write_single_yolo_doc(
            task: tuple[str, AnnotationDocument, Path, Path],
        ) -> tuple[str, list[dict[str, object]]]:
            split, document, images, labels = task
            image_target = images / document.image_path.name
            label_target = labels / f"{document.image_path.stem}.txt"
            if document.image_path.resolve() != image_target.resolve():
                shutil.copyfile(document.image_path, image_target)
            lines = []
            for annotation in document.annotations:
                center_x, center_y, width, height = annotation.box.to_yolo()
                lines.append(
                    f"{CLASS_ORDER.index(annotation.class_name)} {center_x:.6f} "
                    f"{center_y:.6f} {width:.6f} {height:.6f}"
                )
            label_target.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            metadata_key = f"{split}/{document.image_path.name}" if split else document.image_path.name
            doc_meta = [
                {
                    "class_name": annotation.class_name,
                    "occluded": annotation.occluded,
                    "truncated": annotation.truncated,
                }
                for annotation in document.annotations
            ]
            return metadata_key, doc_meta

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            meta_results = pool.map(_write_single_yolo_doc, export_tasks)

        for m_key, m_val in meta_results:
            metadata[m_key] = m_val

        (destination / "annotation_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        yaml_path = destination / "dataset.yaml"
        data_yaml_path = destination / "data.yaml"
        if self.splits:
            split_paths = "\n".join(
                f"{name}: images/{name}" for name in SPLIT_NAMES
            )
        else:
            split_paths = "train: images\nval: images"
        yaml_content = (
            f"path: {destination.resolve()}\n"
            f"{split_paths}\n"
            f"names: {list(CLASS_ORDER)}\n"
        )
        yaml_path.write_text(yaml_content, encoding="utf-8")
        # Also write data.yaml with relative path for seamless Google Colab/Roboflow portability
        colab_yaml_content = (
            "path: .\n"
            f"{split_paths}\n"
            f"names: {list(CLASS_ORDER)}\n"
        )
        data_yaml_path.write_text(colab_yaml_content, encoding="utf-8")
        LOGGER.info(
            "exported %d documents to %s at %s",
            len(documents),
            self.variant,
            destination,
        )
        return yaml_path


class CocoExporter(DatasetExporter):
    """Export COCO detection JSON and source images."""

    def __init__(self, splits: dict[str, list[AnnotationDocument]] | None = None) -> None:
        self.splits = splits

    def export(self, documents: list[AnnotationDocument], destination: Path) -> Path:
        validate_documents(documents)
        if self.splits:
            for split, split_documents_list in self.splits.items():
                CocoExporter().export(split_documents_list, destination / split)
            return destination
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
                        "occluded": annotation.occluded,
                        "truncated": annotation.truncated,
                    }
                )
                annotation_id += 1
        image_destination = destination / "images"
        image_destination.mkdir(exist_ok=True)

        import concurrent.futures
        import os

        max_workers = min(32, max(4, (os.cpu_count() or 4) * 2))

        def _copy_coco_img(doc: AnnotationDocument) -> None:
            target = image_destination / doc.image_path.name
            if doc.image_path.resolve() != target.resolve():
                shutil.copyfile(doc.image_path, target)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_copy_coco_img, documents))

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
                SubElement(obj, "occluded").text = str(int(annotation.occluded))
                SubElement(obj, "truncated").text = str(int(annotation.truncated))
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
                        "occluded": str(int(annotation.occluded)),
                        "truncated": str(int(annotation.truncated)),
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
        shutil.rmtree(staging, ignore_errors=True)
        return archive


def validate_documents(documents: list[AnnotationDocument]) -> None:
    """Reject unsupported classes before writing a partially valid export."""
    for document in documents:
        invalid = {item.class_name for item in document.annotations} - TARGET_CLASSES
        if invalid:
            raise ExportError(f"unsupported classes in {document.image_path}: {sorted(invalid)}")
