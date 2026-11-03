"""Import existing COCO bounding-box annotations into a dataset in place.

Unlike the legacy cleanup importer in :mod:`app.services.dataset.coco_importer`, nothing is
copied, class names are kept, and overlapping boxes are never dropped: a box that overlaps
another is usually an occluded object, which is a valid label and exactly what review looks for.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from app.services.annotation.class_names import normalize_class_name
from app.services.dataset.repository import (
    AnnotationRepository,
    RepositoryError,
    RevisionConflict,
)

LOGGER = logging.getLogger(__name__)

IMPORT_STATUS = "unreviewed"  # Imported labels still need a person to approve each image.
MAX_FAILURES_KEPT = 20
_SKIPPED_DIRS = frozenset({"exports", "labels", "__pycache__", "node_modules"})
_MIN_BOX_PIXELS = 0.5


class CocoFormatError(ValueError):
    """Raised when a file is not a usable COCO detection document."""


@dataclass(frozen=True, slots=True)
class CocoSource:
    """A COCO annotation file found next to a dataset's images."""

    path: Path
    image_root: Path
    images: int
    annotations: int
    categories: tuple[str, ...]
    found_images: int


@dataclass(slots=True)
class ImportCounts:
    """Running totals reported while an import job works through a file."""

    total_images: int = 0
    processed: int = 0
    imported: int = 0
    skipped_existing: int = 0
    skipped_conflict: int = 0
    missing_images: int = 0
    boxes: int = 0
    invalid_boxes: int = 0
    orphan_boxes: int = 0
    classes_added: list[str] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)

    def fail(self, image: str, error: str) -> None:
        if len(self.failures) < MAX_FAILURES_KEPT:
            self.failures.append({"image": image, "error": error})


def looks_like_coco_name(name: str) -> bool:
    lowered = name.lower()
    if not lowered.endswith(".json"):
        return False
    return (
        lowered in {"annotations.json", "_annotations.coco.json"}
        or lowered.startswith("instances")
        or lowered.endswith(".coco.json")
        or "coco" in lowered
    )


def find_coco_files(root: Path, max_depth: int = 3) -> list[Path]:
    """Return candidate COCO files under ``root`` without descending into hidden or output dirs."""
    root = root.resolve()
    found: list[Path] = []
    for directory, subdirs, files in os.walk(root):
        depth = len(Path(directory).relative_to(root).parts)
        subdirs[:] = sorted(
            d for d in subdirs
            if not d.startswith(".") and d not in _SKIPPED_DIRS and depth < max_depth
        )
        found.extend(Path(directory) / name for name in sorted(files) if looks_like_coco_name(name))
    return found


def read_coco(path: Path) -> dict[str, Any]:
    """Load and shape-check a COCO document."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise CocoFormatError(f"cannot read {path.name}: {err}") from err
    if not isinstance(payload, dict) or not all(
        isinstance(payload.get(key), list) for key in ("images", "annotations", "categories")
    ):
        raise CocoFormatError(f"{path.name} is not a COCO detection file")
    return payload


def default_image_root(annotation_file: Path) -> Path:
    """Images usually sit in an ``images`` folder beside the file, or in the same folder."""
    sibling = annotation_file.parent / "images"
    return sibling if sibling.is_dir() else annotation_file.parent


def _category_names(payload: dict[str, Any]) -> dict[int, str]:
    names: dict[int, str] = {}
    for item in payload["categories"]:
        if isinstance(item, dict) and "id" in item and str(item.get("name", "")).strip():
            names[int(item["id"])] = str(item["name"]).strip()
    return names


def _file_name(record: Any) -> str | None:
    value = record.get("file_name") if isinstance(record, dict) else None
    return str(value) if value else None


_SUMMARY_CACHE: dict[tuple[str, int, int], CocoSource] = {}


def describe_source(path: Path) -> CocoSource:
    """Summarize a COCO file. Parsed results are cached by file identity."""
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _SUMMARY_CACHE.get(key)
    if cached is not None:
        return cached
    payload = read_coco(path)
    root = default_image_root(path)
    found = sum(
        1 for record in payload["images"]
        if (name := _file_name(record)) and (root / name).is_file()
    )
    summary = CocoSource(
        path=path,
        image_root=root,
        images=len(payload["images"]),
        annotations=len(payload["annotations"]),
        categories=tuple(_category_names(payload).values()),
        found_images=found,
    )
    if len(_SUMMARY_CACHE) > 64:
        _SUMMARY_CACHE.clear()
    _SUMMARY_CACHE[key] = summary
    return summary


class ClassResolver:
    """Map annotation-file category names onto the dataset's class registry.

    An exact (case-insensitive) match wins, then the shared vehicle aliases. A name that is
    still unknown is registered as a new class so no label is lost.
    """

    def __init__(self, repo: AnnotationRepository) -> None:
        self._repo = repo
        self._registered = {name.casefold(): name for name in repo.classes()}
        self.added: list[str] = []

    def resolve(self, raw: str) -> str:
        key = raw.strip().casefold()
        if key in self._registered:
            return self._registered[key]
        alias = normalize_class_name(raw).strip().casefold()
        if alias in self._registered:
            return self._registered[alias]
        classes = self._repo.register_classes([raw.strip()])
        self._registered = {name.casefold(): name for name in classes}
        name = self._registered[key]
        self.added.append(name)
        return name


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _polygon(
    segmentation: Any, scale: tuple[float, float], size: tuple[int, int]
) -> list[list[float]] | None:
    """Normalized points of the largest polygon in a COCO ``segmentation`` field."""
    if not isinstance(segmentation, list):
        return None
    candidates = [
        ring for ring in segmentation
        if isinstance(ring, list) and len(ring) >= 6 and len(ring) % 2 == 0
    ]
    if not candidates:
        return None
    ring = max(candidates, key=len)
    try:
        return [
            [
                _clamp(float(ring[i]) * scale[0] / size[0], 0.0, 1.0),
                _clamp(float(ring[i + 1]) * scale[1] / size[1], 0.0, 1.0),
            ]
            for i in range(0, len(ring), 2)
        ]
    except (TypeError, ValueError):
        return None


def build_boxes(
    annotations: list[dict[str, Any]],
    categories: dict[int, str],
    resolve: Callable[[str], str],
    size: tuple[int, int],
    json_size: tuple[int, int] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Convert COCO annotations of one image to repository box dicts.

    ``size`` is the real image size; when the file declares a different size the coordinates
    are rescaled. ``class_id`` is left out on purpose: the repository derives it from the
    registry, which ``resolve`` may extend while this runs. Returns ``(boxes, invalid_count)``.
    """
    width, height = size
    scale = (1.0, 1.0)
    if json_size and all(json_size):
        scale = (width / json_size[0], height / json_size[1])
    boxes: list[dict[str, Any]] = []
    invalid = 0
    used: set[str] = set()
    for index, item in enumerate(annotations):
        try:
            x, y, w, h = (float(v) for v in item["bbox"])
            name = categories[int(item["category_id"])]
        except (KeyError, TypeError, ValueError):
            invalid += 1
            continue
        left = _clamp(x * scale[0], 0.0, width)
        top = _clamp(y * scale[1], 0.0, height)
        right = _clamp((x + w) * scale[0], 0.0, width)
        bottom = _clamp((y + h) * scale[1], 0.0, height)
        values = (left, top, right, bottom)
        if (
            not all(math.isfinite(v) for v in values)
            or right - left < _MIN_BOX_PIXELS
            or bottom - top < _MIN_BOX_PIXELS
        ):
            invalid += 1
            continue
        class_name = resolve(name)
        identifier = f"coco-{item['id']}" if item.get("id") is not None else f"coco-{index}"
        if identifier in used:
            identifier = f"{identifier}-{index}"
        used.add(identifier)
        attributes = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
        box: dict[str, Any] = {
            "id": identifier,
            "class_name": class_name,
            "confidence": 1.0,
            "source": "human",
            "norm_left": left / width,
            "norm_top": top / height,
            "norm_right": right / width,
            "norm_bottom": bottom / height,
            "occluded": bool(item.get("occluded", attributes.get("occluded", False))),
            "truncated": bool(item.get("truncated", attributes.get("truncated", False))),
        }
        polygon = _polygon(item.get("segmentation"), scale, size)
        if polygon:
            box["polygon_normalized"] = polygon
        boxes.append(box)
    return boxes, invalid


def iter_image_records(
    payload: dict[str, Any],
    annotation_file: Path,
    locate_fallback: Callable[[str], Path | None] | None = None,
) -> Iterator[tuple[dict[str, Any], Path | None]]:
    """Yield each image record with its file on disk (``None`` when it cannot be found)."""
    root = default_image_root(annotation_file)
    for record in payload["images"]:
        name = _file_name(record)
        located: Path | None = None
        if name:
            for base in (root, annotation_file.parent):
                candidate = base / name
                if candidate.is_file():
                    located = candidate
                    break
            if located is None and locate_fallback is not None:
                located = locate_fallback(Path(name).name)
        yield record, located


def import_coco(
    repo: AnnotationRepository,
    annotation_file: Path,
    *,
    overwrite: bool = False,
    counts: ImportCounts | None = None,
    progress: Callable[[ImportCounts, str], None] | None = None,
    on_saved: Callable[[Path, dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    locate_fallback: Callable[[str], Path | None] | None = None,
    count_total: bool = True,
) -> ImportCounts:
    """Save every image's COCO boxes as its annotation document, in place.

    Images that already have labels are left alone unless ``overwrite`` is set, and an image
    edited by someone else while the import runs is skipped rather than overwritten. A caller
    that already knows the grand total over several files passes ``count_total=False``.
    """
    counts = counts if counts is not None else ImportCounts()
    payload = read_coco(annotation_file)
    categories = _category_names(payload)
    by_image: dict[int, list[dict[str, Any]]] = {}
    known_ids = {int(r["id"]) for r in payload["images"] if isinstance(r, dict) and "id" in r}
    for item in payload["annotations"]:
        try:
            image_id = int(item["image_id"])
        except (KeyError, TypeError, ValueError):
            counts.orphan_boxes += 1
            continue
        if image_id in known_ids:
            by_image.setdefault(image_id, []).append(item)
        else:
            counts.orphan_boxes += 1
    resolver = ClassResolver(repo)
    if count_total:
        counts.total_images += len(payload["images"])

    for record, image_path in iter_image_records(payload, annotation_file, locate_fallback):
        if cancelled():
            break
        counts.processed += 1
        label = _file_name(record) or "?"
        counts.classes_added = list(resolver.added)
        if progress:
            progress(counts, label)
        if image_path is None:
            counts.missing_images += 1
            continue
        try:
            existing, kind = repo.resolve(image_path)
            revision = 0
            if existing is not None:
                if not overwrite:
                    counts.skipped_existing += 1
                    continue
                revision = int(repo.read(image_path).get("revision", 0)) if kind == "json" else 0
            with Image.open(image_path) as source:
                size = source.size
            declared = (int(record.get("width") or 0), int(record.get("height") or 0))
            boxes, invalid = build_boxes(
                by_image.get(int(record["id"]), []) if "id" in record else [],
                categories,
                resolver.resolve,
                size,
                declared if all(declared) else None,
            )
            counts.invalid_boxes += invalid
            with repo.lock, repo._write_lock():
                saved = repo.save(
                    image_path, boxes, status=IMPORT_STATUS, expected_revision=revision
                )
                counts.imported += 1
                counts.boxes += len(saved["boxes"])
                if on_saved:
                    on_saved(image_path, saved)
        except RevisionConflict:
            counts.skipped_conflict += 1
        except (RepositoryError, OSError, ValueError) as err:
            LOGGER.warning("Cannot import annotations for %s: %s", label, err)
            counts.fail(label, str(err))
    counts.classes_added = list(resolver.added)
    return counts
