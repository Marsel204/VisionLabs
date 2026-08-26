"""COCO detection dataset import for annotation cleanup projects."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.services.annotation.domain import (
    TARGET_CLASSES,
    Annotation,
    AnnotationDocument,
    AnnotationSource,
    BoundingBox,
)
from app.services.fusion.overlap import remove_overlapping_annotations

LOGGER = logging.getLogger(__name__)


class CocoImportError(ValueError):
    """Raised when a COCO dataset cannot be imported safely."""


@dataclass(frozen=True, slots=True)
class CocoImportReport:
    """Counts and warnings produced during one COCO import."""

    images_found: int = 0
    images_imported: int = 0
    annotations_found: int = 0
    annotations_imported: int = 0
    unsupported_categories: int = 0
    missing_images: int = 0
    invalid_annotations: int = 0
    overlapping_removed: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CocoImportResult:
    """Copied imported documents and their import report."""

    project_root: Path
    documents: tuple[AnnotationDocument, ...]
    report: CocoImportReport


class CocoImporter:
    """Import COCO bounding-box annotations into a copied project dataset."""

    def import_dataset(
        self,
        annotation_file: Path,
        image_root: Path,
        destination: Path,
        *,
        remove_overlaps: bool = True,
        overlap_iou_threshold: float = 0.50,
        containment_threshold: float = 0.80,
    ) -> CocoImportResult:
        """Copy images and convert supported COCO boxes into annotation documents."""
        payload = self._read_payload(annotation_file)
        if not image_root.is_dir():
            raise CocoImportError(f"image root is not a directory: {image_root}")
        images = self._required_list(payload, "images")
        annotations = self._required_list(payload, "annotations")
        categories = self._required_list(payload, "categories")

        CLASS_ALIASES = {
            "motorbike": "motorcycle",
            "motorcycles": "motorcycle",
            "motorbikes": "motorcycle",
            "cars": "car",
            "automobile": "car",
            "automobiles": "car",
            "auto": "car",
            "autos": "car",
            "suv": "car",
            "sedan": "car",
            "van": "car",
            "vans": "car",
            "trucks": "truck",
            "pickup": "truck",
            "buses": "bus",
        }
        category_names = {}
        for item in categories:
            if isinstance(item, dict) and "id" in item and "name" in item:
                raw_name = str(item["name"]).strip().lower().replace("-", "").replace("_", "")
                norm_name = CLASS_ALIASES.get(raw_name, str(item["name"]).strip().lower())
                category_names[int(item["id"])] = norm_name

        supported = {
            category_id: name
            for category_id, name in category_names.items()
            if name in TARGET_CLASSES
        }
        by_image: dict[int, list[dict[str, Any]]] = {}
        for item in annotations:
            by_image.setdefault(int(item.get("image_id", -1)), []).append(item)

        project_images = destination / "images"
        project_images.mkdir(parents=True, exist_ok=True)
        documents: list[AnnotationDocument] = []
        warnings: list[str] = []
        counts = {
            "unsupported_categories": 0,
            "missing_images": 0,
            "invalid_annotations": 0,
            "annotations_imported": 0,
            "overlapping_removed": 0,
        }
        for image_record in images:
            try:
                image_id = int(image_record["id"])
                relative_name = Path(str(image_record["file_name"]))
                source_path = self._safe_source_path(image_root, relative_name)
                if not source_path.is_file():
                    counts["missing_images"] += 1
                    warnings.append(f"missing image: {relative_name}")
                    continue
                target_path = project_images / relative_name.name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if source_path.resolve() != target_path.resolve():
                        shutil.copy2(source_path, target_path)
                except OSError as err:
                    counts["missing_images"] += 1
                    warnings.append(f"could not copy {source_path.name}: {err}")
                    continue

                try:
                    image_width = int(float(image_record.get("width", 0) or 0))
                    image_height = int(float(image_record.get("height", 0) or 0))
                except (ValueError, TypeError):
                    image_width, image_height = 0, 0

                if image_width <= 0 or image_height <= 0:
                    try:
                        with Image.open(source_path) as im:
                            image_width, image_height = im.width, im.height
                    except Exception:
                        image_width, image_height = 640, 480

                imported: list[Annotation] = []
                for record in by_image.get(image_id, []):
                    category_id = int(record.get("category_id", -1))
                    if category_id not in supported:
                        counts["unsupported_categories"] += 1
                        continue
                    try:
                        box = self._box(record.get("bbox"), image_width, image_height)
                        imported.append(
                            Annotation(
                                class_name=supported[category_id],
                                box=box,
                                confidence=None,
                                source=AnnotationSource.HUMAN,
                                occluded=record.get("occluded", False)
                                in (True, 1, "1", "true"),
                                truncated=record.get("truncated", False)
                                in (True, 1, "1", "true"),
                            )
                        )
                    except (KeyError, TypeError, ValueError) as error:
                        counts["invalid_annotations"] += 1
                        warnings.append(f"invalid annotation {record.get('id', '?')}: {error}")
                counts["annotations_imported"] += len(imported)
                if remove_overlaps:
                    kept, removed = remove_overlapping_annotations(
                        imported,
                        overlap_iou_threshold,
                        containment_threshold,
                        same_class_only=True,
                    )
                    counts["overlapping_removed"] += removed
                    imported = list(kept)
                documents.append(
                    AnnotationDocument(
                        target_path,
                        image_width,
                        image_height,
                        tuple(imported),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                image_id = image_record.get("id", -1)
                related = by_image.get(int(image_id), []) if isinstance(image_id, int) else []
                counts["invalid_annotations"] += len(related)
                warnings.append(f"invalid image record {image_record.get('id', '?')}: {error}")

        report = CocoImportReport(
            images_found=len(images),
            images_imported=len(documents),
            annotations_found=len(annotations),
            warnings=tuple(warnings),
            **counts,
        )
        LOGGER.info(
            "COCO import completed: %d/%d images, %d annotations, %d overlaps removed",
            report.images_imported,
            report.images_found,
            report.annotations_imported,
            report.overlapping_removed,
        )
        return CocoImportResult(destination, tuple(documents), report)

    @staticmethod
    def _read_payload(annotation_file: Path) -> dict[str, Any]:
        try:
            payload = json.loads(annotation_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CocoImportError(f"could not read COCO annotations: {error}") from error
        if not isinstance(payload, dict):
            raise CocoImportError("COCO root must be an object")
        return payload

    @staticmethod
    def _required_list(payload: dict[str, Any], name: str) -> list[dict[str, Any]]:
        value = payload.get(name)
        if not isinstance(value, list):
            raise CocoImportError(f"COCO field '{name}' must be a list")
        return [item for item in value if isinstance(item, dict)]

    @staticmethod
    def _safe_source_path(image_root: Path, relative_name: Path | str) -> Path:
        root = image_root.resolve()
        clean_rel_str = str(relative_name).replace("\\", "/").lstrip("/")
        clean_rel = Path(clean_rel_str)

        # 1. Direct relative candidate
        candidate = (root / clean_rel).resolve()
        if candidate.is_file() and (candidate == root or root in candidate.parents):
            return candidate

        # 2. Direct basename in image_root
        candidate_base = (root / clean_rel.name).resolve()
        if candidate_base.is_file() and (candidate_base == root or root in candidate_base.parents):
            return candidate_base

        # 3. Check 'images' subfolder inside image_root
        candidate_images = (root / "images" / clean_rel.name).resolve()
        if candidate_images.is_file() and root in candidate_images.parents:
            return candidate_images

        # 4. Check if relative_name itself was an existing absolute path inside image_root
        try:
            abs_cand = Path(str(relative_name)).resolve()
            if abs_cand.is_file() and (abs_cand == root or root in abs_cand.parents):
                return abs_cand
        except Exception:
            pass

        return candidate

    @staticmethod
    def _box(value: object, width: int, height: int) -> BoundingBox:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("bbox must contain [x, y, width, height]")
        left, top, box_width, box_height = (float(item) for item in value)
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image dimensions: {width}x{height}")
        right = left + box_width
        bottom = top + box_height
        left = max(0.0, min(left, float(width)))
        top = max(0.0, min(top, float(height)))
        right = max(0.0, min(right, float(width)))
        bottom = max(0.0, min(bottom, float(height)))
        if left >= right or top >= bottom:
            raise ValueError("bbox must have positive area")
        # Ensure clamped normalized coordinates
        n_left = max(0.0, min(1.0, left / width))
        n_top = max(0.0, min(1.0, top / height))
        n_right = max(0.0, min(1.0, right / width))
        n_bottom = max(0.0, min(1.0, bottom / height))
        if n_left >= n_right or n_top >= n_bottom:
            raise ValueError("normalized bbox must have positive area")
        return BoundingBox(n_left, n_top, n_right, n_bottom)
