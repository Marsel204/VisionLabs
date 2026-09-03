"""YOLO detection dataset import for annotation cleanup projects."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

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


class YoloImportError(ValueError):
    """Raised when a YOLO dataset cannot be imported safely."""


@dataclass(frozen=True, slots=True)
class YoloImportReport:
    """Counts and warnings produced during one YOLO import."""

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
class YoloImportResult:
    """Copied imported documents and their import report."""

    project_root: Path
    documents: tuple[AnnotationDocument, ...]
    report: YoloImportReport


class YoloImporter:
    """Import YOLO detection annotations and images into a copied project dataset."""

    def import_dataset(
        self,
        yaml_file: Path,
        destination: Path,
        *,
        remove_overlaps: bool = True,
        overlap_iou_threshold: float = 0.50,
        containment_threshold: float = 0.80,
    ) -> YoloImportResult:
        """Parse data.yaml, copy images, and convert YOLO annotations into AnnotationDocuments."""
        if not yaml_file.is_file():
            raise YoloImportError(f"YOLO YAML file does not exist: {yaml_file}")

        payload = self._read_yaml(yaml_file)
        category_map = self._parse_categories(payload)
        supported_categories = {
            cat_id: name for cat_id, name in category_map.items() if name in TARGET_CLASSES
        }

        base_dir = self._resolve_base_dir(yaml_file, payload)
        image_paths = self._discover_images(payload, base_dir, yaml_file.parent)

        if not image_paths:
            raise YoloImportError(f"No images found for dataset defined in {yaml_file}")

        project_images = destination / "images"
        project_images.mkdir(parents=True, exist_ok=True)

        documents: list[AnnotationDocument] = []
        warnings: list[str] = []
        counts = {
            "unsupported_categories": 0,
            "missing_images": 0,
            "invalid_annotations": 0,
            "annotations_found": 0,
            "annotations_imported": 0,
            "overlapping_removed": 0,
        }

        # Build unique target paths deterministically
        tasks: list[tuple[Path, Path, Path | None]] = []
        used_target_names: set[str] = set()
        for image_src in image_paths:
            if not image_src.is_file():
                counts["missing_images"] += 1
                warnings.append(f"missing image file: {image_src}")
                continue

            target_name = image_src.name
            if target_name in used_target_names:
                stem = image_src.stem
                suffix = image_src.suffix
                counter = 2
                while f"{stem}_{counter}{suffix}" in used_target_names:
                    counter += 1
                target_name = f"{stem}_{counter}{suffix}"
            used_target_names.add(target_name)
            target_path = project_images / target_name
            label_path = self._find_label_file(image_src, base_dir, yaml_file.parent)
            tasks.append((image_src, target_path, label_path))

        def _process_item(
            item: tuple[Path, Path, Path | None],
        ) -> tuple[AnnotationDocument | None, dict[str, int], list[str]]:
            image_src, target_path, label_path = item
            sub_counts = {
                "unsupported_categories": 0,
                "missing_images": 0,
                "invalid_annotations": 0,
                "annotations_found": 0,
                "annotations_imported": 0,
                "overlapping_removed": 0,
            }
            sub_warnings: list[str] = []

            try:
                if image_src.resolve() != target_path.resolve():
                    shutil.copy2(image_src, target_path)
            except OSError as err:
                sub_counts["missing_images"] += 1
                sub_warnings.append(f"could not copy {image_src.name}: {err}")
                return None, sub_counts, sub_warnings

            # Fast dimension read via PIL size (reads header only)
            image_width, image_height = 0, 0
            try:
                with Image.open(image_src) as im:
                    image_width, image_height = im.size
            except Exception:
                image_width, image_height = 1920, 1080

            if image_width <= 0 or image_height <= 0:
                image_width, image_height = 1920, 1080

            raw_annotations: list[Annotation] = []
            if label_path and label_path.is_file():
                try:
                    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
                except Exception as err:
                    sub_warnings.append(f"could not read label file {label_path}: {err}")
                    lines = []

                for line_idx, line in enumerate(lines, start=1):
                    line_str = line.strip()
                    if not line_str or line_str.startswith("#"):
                        continue
                    sub_counts["annotations_found"] += 1
                    parts = line_str.split()
                    if len(parts) < 5:
                        sub_counts["invalid_annotations"] += 1
                        sub_warnings.append(
                            f"invalid YOLO line in {label_path.name}:{line_idx} -> '{line_str}'"
                        )
                        continue

                    try:
                        class_id = int(float(parts[0]))
                    except ValueError:
                        sub_counts["invalid_annotations"] += 1
                        sub_warnings.append(f"invalid class ID in {label_path.name}:{line_idx}")
                        continue

                    if class_id not in supported_categories:
                        sub_counts["unsupported_categories"] += 1
                        continue

                    try:
                        cx, cy, bw, bh = (float(v) for v in parts[1:5])
                        conf = float(parts[5]) if len(parts) >= 6 else None
                        box = self._yolo_to_bbox(cx, cy, bw, bh)
                        raw_annotations.append(
                            Annotation(
                                class_name=supported_categories[class_id],
                                box=box,
                                confidence=conf if conf is not None and 0.0 <= conf <= 1.0 else None,
                                source=AnnotationSource.HUMAN,
                            )
                        )
                    except (ValueError, TypeError) as err:
                        sub_counts["invalid_annotations"] += 1
                        sub_warnings.append(
                            f"invalid coordinates in {label_path.name}:{line_idx} -> {err}"
                        )

            sub_counts["annotations_imported"] += len(raw_annotations)

            if remove_overlaps and raw_annotations:
                kept, removed = remove_overlapping_annotations(
                    raw_annotations,
                    overlap_iou_threshold,
                    containment_threshold,
                    same_class_only=True,
                )
                sub_counts["overlapping_removed"] += removed
                raw_annotations = list(kept)

            doc = AnnotationDocument(
                target_path,
                image_width,
                image_height,
                tuple(raw_annotations),
            )
            return doc, sub_counts, sub_warnings

        import concurrent.futures
        import os

        max_workers = min(32, max(4, (os.cpu_count() or 4) * 2))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = pool.map(_process_item, tasks)

        for doc, sub_counts, sub_warnings in results:
            if doc is not None:
                documents.append(doc)
            for k, v in sub_counts.items():
                counts[k] += v
            warnings.extend(sub_warnings)

        report = YoloImportReport(
            images_found=len(image_paths),
            images_imported=len(documents),
            warnings=tuple(warnings),
            **counts,
        )

        LOGGER.info(
            "YOLO import completed: %d/%d images, %d annotations, %d overlaps removed",
            report.images_imported,
            report.images_found,
            report.annotations_imported,
            report.overlapping_removed,
        )
        return YoloImportResult(destination, tuple(documents), report)

    @staticmethod
    def _read_yaml(yaml_file: Path) -> dict[str, Any]:
        try:
            content = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        except Exception as error:
            raise YoloImportError(f"could not parse YAML file: {error}") from error
        if not isinstance(content, dict):
            raise YoloImportError("YAML root must be a dictionary/mapping")
        return content

    @staticmethod
    def _parse_categories(payload: dict[str, Any]) -> dict[int, str]:
        """Extract and normalize category mappings from names / nc fields."""
        names = payload.get("names")
        raw_map: dict[int, str] = {}

        if isinstance(names, list):
            for idx, name in enumerate(names):
                raw_map[idx] = str(name)
        elif isinstance(names, dict):
            for k, v in names.items():
                try:
                    raw_map[int(k)] = str(v)
                except ValueError:
                    pass

        normalized: dict[int, str] = {}
        for cat_id, raw_name in raw_map.items():
            clean = raw_name.strip().lower().replace("-", "").replace("_", "")
            norm_name = CLASS_ALIASES.get(clean, raw_name.strip().lower())
            normalized[cat_id] = norm_name

        return normalized

    @staticmethod
    def _resolve_base_dir(yaml_file: Path, payload: dict[str, Any]) -> Path:
        """Determine the root path from the 'path' key or yaml parent directory."""
        yaml_dir = yaml_file.parent.resolve()
        path_val = payload.get("path")
        if path_val:
            candidate = Path(str(path_val))
            if candidate.is_absolute() and candidate.is_dir():
                return candidate.resolve()
            rel_candidate = (yaml_dir / candidate).resolve()
            if rel_candidate.is_dir():
                return rel_candidate
        return yaml_dir

    def _discover_images(
        self, payload: dict[str, Any], base_dir: Path, yaml_dir: Path
    ) -> list[Path]:
        """Find all image paths referenced in splits or located in dataset folders."""
        image_paths: list[Path] = []
        seen: set[Path] = set()

        # Check standard split keys
        split_keys = ("train", "val", "valid", "test", "images")
        candidates = []
        for key in split_keys:
            val = payload.get(key)
            if val is not None:
                if isinstance(val, (list, tuple)):
                    candidates.extend(val)
                elif isinstance(val, str):
                    candidates.append(val)

        if not candidates:
            # Fall back to base_dir or base_dir/images
            candidates = [str(base_dir)]

        for entry in candidates:
            p_str = str(entry).strip()
            # Try relative to base_dir, then relative to yaml_dir, then absolute
            resolved: Path | None = None
            for root in (base_dir, yaml_dir):
                cand = (root / p_str).resolve()
                if cand.exists():
                    resolved = cand
                    break
            if resolved is None and Path(p_str).is_absolute() and Path(p_str).exists():
                resolved = Path(p_str).resolve()

            if resolved is None:
                continue

            if resolved.is_file():
                if resolved.suffix.lower() in IMAGE_SUFFIXES:
                    if resolved not in seen:
                        seen.add(resolved)
                        image_paths.append(resolved)
                elif resolved.suffix.lower() == ".txt":
                    # Text file containing image paths
                    try:
                        for line in resolved.read_text(encoding="utf-8").splitlines():
                            img_line = line.strip()
                            if not img_line:
                                continue
                            cand_img = self._resolve_single_image(img_line, resolved.parent, base_dir)
                            if cand_img and cand_img not in seen:
                                seen.add(cand_img)
                                image_paths.append(cand_img)
                    except Exception as err:
                        LOGGER.warning("Could not read text list %s: %s", resolved, err)
            elif resolved.is_dir():
                # Directory of images
                for item in resolved.rglob("*"):
                    if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES:
                        norm = item.resolve()
                        if norm not in seen:
                            seen.add(norm)
                            image_paths.append(norm)

        return sorted(image_paths)

    @staticmethod
    def _resolve_single_image(path_str: str, *roots: Path) -> Path | None:
        p = Path(path_str)
        if p.is_absolute() and p.is_file():
            return p.resolve()
        for root in roots:
            cand = (root / p).resolve()
            if cand.is_file():
                return cand
        return None

    @staticmethod
    def _find_label_file(image_path: Path, base_dir: Path, yaml_dir: Path) -> Path | None:
        """Find the matching .txt label file for a YOLO image."""
        stem = image_path.stem
        # 1. Check path replacing /images/ with /labels/
        parts = list(image_path.parts)
        for idx in reversed(range(len(parts))):
            if parts[idx].lower() == "images":
                label_parts = list(parts)
                label_parts[idx] = "labels"
                cand = Path(*label_parts).with_suffix(".txt")
                if cand.is_file():
                    return cand

        # 2. Check sibling labels directory: parent.parent / labels / parent.name / stem.txt or parent.parent / labels / stem.txt
        for parent_cand in (image_path.parent, image_path.parent.parent):
            cand1 = parent_cand / "labels" / f"{stem}.txt"
            if cand1.is_file():
                return cand1
            cand2 = parent_cand / "labels" / image_path.parent.name / f"{stem}.txt"
            if cand2.is_file():
                return cand2

        # 3. Check exact sibling file with .txt extension
        same_dir_txt = image_path.with_suffix(".txt")
        if same_dir_txt.is_file():
            return same_dir_txt

        # 4. Search in labels/ subdirectories of base_dir or yaml_dir
        for root in (base_dir, yaml_dir):
            labels_dir = root / "labels"
            if labels_dir.is_dir():
                cand_match = list(labels_dir.rglob(f"{stem}.txt"))
                if cand_match:
                    return cand_match[0]

        return None

    @staticmethod
    def _yolo_to_bbox(cx: float, cy: float, bw: float, bh: float) -> BoundingBox:
        """Convert normalized YOLO cx, cy, w, h to normalized BoundingBox xyxy."""
        if bw <= 0 or bh <= 0:
            raise ValueError(f"box dimensions must be positive: {bw}x{bh}")

        left = max(0.0, min(1.0, cx - bw / 2.0))
        top = max(0.0, min(1.0, cy - bh / 2.0))
        right = max(0.0, min(1.0, cx + bw / 2.0))
        bottom = max(0.0, min(1.0, cy + bh / 2.0))

        if left >= right or top >= bottom:
            raise ValueError("computed bounding box coordinates are inverted or zero area")

        return BoundingBox(left, top, right, bottom)
