"""Canonical local annotations, class registry and derived training labels.

JSON is authoritative; SQLite is a rebuildable image/review cache. Legacy labels
are read on demand and become canonical on the first save. Each key retains the
relative image path and extension, so equal basenames and stems cannot collide.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import yaml
from PIL import Image

from app.services.annotation.domain import (
    Annotation,
    AnnotationDocument,
    AnnotationSource,
    BoundingBox,
    ReviewStatus,
)
from app.services.dataset.index import IMAGE_SUFFIXES

DEFAULT_CLASSES = ("motorcycle", "car", "bus", "truck", "minivan", "person")
_LOCKS: dict[Path, Any] = {}
_LOCKS_GUARD = threading.Lock()
_WRITE_STATE = threading.local()


class RepositoryError(ValueError):
    """An input or persisted document cannot be interpreted safely."""


class RevisionConflict(RepositoryError):
    """An edit targets a document revision that has already changed."""


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class AnnotationRepository:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        with _LOCKS_GUARD:
            self.lock = _LOCKS.setdefault(self.root, threading.RLock())
        self._stem_counts: dict[str, int] | None = None

    @contextmanager
    def _write_lock(self):
        """Serialize writers in separate Qt/API processes as well as threads."""
        held = getattr(_WRITE_STATE, "held", None)
        if held is None:
            held = _WRITE_STATE.held = set()
        if self.root in held:
            yield
            return
        path = self.root / ".visionlab" / "repository.lock"
        if not path.resolve().is_relative_to(self.root):
            raise RepositoryError("repository lock escapes dataset")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as stream:
            if os.name == "nt":
                import msvcrt

                stream.write(b"0")
                stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            held.add(self.root)
            try:
                yield
            finally:
                held.remove(self.root)
                if os.name == "nt":
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def image_path(self, name: str | Path) -> Path:
        candidate = Path(name)
        path = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
        if not path.is_relative_to(self.root) or path.suffix.lower() not in IMAGE_SUFFIXES:
            raise RepositoryError("image must be inside the dataset and have a supported format")
        if not path.is_file() and not candidate.is_absolute() and len(candidate.parts) == 1:
            matches = [
                p
                for p in self.root.rglob(candidate.name)
                if p.is_file()
                and p.resolve().is_relative_to(self.root)
                and not any(
                    part.startswith(".") or part in {"exports", "labels"}
                    for part in p.relative_to(self.root).parts[:-1]
                )
            ]
            if len(matches) > 1:
                raise RepositoryError("ambiguous image basename; use the dataset-relative path")
            if matches:
                path = matches[0].resolve()
        if not path.is_file():
            raise FileNotFoundError(name)
        return path

    def canonical_path(self, image: Path) -> Path:
        relative = self.image_path(image).relative_to(self.root)
        path = (
            self.root / ".visionlab" / "annotations" / relative.parent / (relative.name + ".json")
        )
        if not path.resolve().is_relative_to(self.root):
            raise RepositoryError("annotation storage escapes dataset")
        return path

    def classes(self) -> list[str]:
        with self.lock:
            for name in ("data.yaml", "dataset.yaml"):
                path = self.root / name
                if path.is_file():
                    try:
                        names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(
                            "names", []
                        )
                        if isinstance(names, dict):
                            keys = sorted(names, key=lambda k: int(k))
                            if [int(key) for key in keys] != list(range(len(keys))):
                                raise RepositoryError("class IDs must be contiguous from zero")
                            names = [names[key] for key in keys]
                        if names:
                            result = [str(n).strip() for n in names]
                            if any(not n for n in result) or len(
                                {n.casefold() for n in result}
                            ) != len(result):
                                raise RepositoryError(
                                    "dataset classes must be unique nonempty names"
                                )
                            return result
                    except (
                        OSError,
                        yaml.YAMLError,
                        TypeError,
                        KeyError,
                        ValueError,
                        AttributeError,
                    ) as err:
                        raise RepositoryError(f"cannot read class registry: {err}") from err
            return list(DEFAULT_CLASSES)

    def register_classes(self, names: list[str]) -> list[str]:
        with self.lock, self._write_lock():
            classes = self.classes()
            for name in names:
                name = name.strip()
                if not name:
                    raise RepositoryError("class name cannot be empty")
                if name.casefold() not in {c.casefold() for c in classes}:
                    classes.append(name)
            path = self.root / "data.yaml"
            data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
            data = data or {}
            data["names"] = classes
            atomic_text(path, yaml.safe_dump(data, sort_keys=False))
            return classes

    def resolve(self, image: Path) -> tuple[Path | None, str]:
        image = self.image_path(image)
        canonical = self.canonical_path(image)
        if canonical.is_file():
            return canonical, "json"
        relative = image.relative_to(self.root)
        parts = list(relative.parts)
        mirrored = None
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            mirrored = self.root.joinpath(*parts).with_suffix(".json")
        candidates = [mirrored, mirrored.with_suffix(".txt")] if mirrored else []
        nested = self.root / "labels" / relative
        candidates += [nested.with_suffix(".json"), nested.with_suffix(".txt")]
        candidates += [image.with_suffix(".json"), image.with_suffix(".txt")]
        candidates += [
            self.root / "labels" / f"{image.stem}.json",
            self.root / "labels" / f"{image.stem}.txt",
        ]
        for path in candidates:
            if path.is_file():
                if not path.resolve().is_relative_to(self.root):
                    raise RepositoryError("label path escapes dataset")
                # A legacy stem-only path cannot identify two images reliably.
                if self._stem_counts is None:
                    self._stem_counts = {}
                    for peer in self.root.rglob("*"):
                        if (
                            peer.suffix.lower() in IMAGE_SUFFIXES
                            and peer.is_file()
                            and peer.resolve().is_relative_to(self.root)
                            and not any(
                                part in {"exports", "labels"} or part.startswith(".")
                                for part in peer.relative_to(self.root).parts[:-1]
                            )
                        ):
                            self._stem_counts[peer.stem] = self._stem_counts.get(peer.stem, 0) + 1
                same_parent = sum(
                    p.suffix.lower() in IMAGE_SUFFIXES for p in image.parent.glob(image.stem + ".*")
                )
                if (
                    self._stem_counts.get(image.stem, 0) > 1 and path.parent == self.root / "labels"
                ) or same_parent > 1:
                    raise RepositoryError(
                        f"ambiguous legacy labels for {relative}; migrate each image explicitly"
                    )
                return path, "json" if path.suffix == ".json" else "yolo"
        return None, ""

    def read(self, image: Path) -> dict[str, Any]:
        with self.lock:
            image = self.image_path(image)
            with Image.open(image) as source:
                width, height = source.size
            path, kind = self.resolve(image)
            data: dict[str, Any] = {"boxes": [], "revision": 0, "status": "unreviewed"}
            if path and kind == "json":
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    data.update(loaded)
                except (OSError, ValueError) as err:
                    raise RepositoryError(f"unreadable annotation document: {path}") from err
                if "status" not in loaded:
                    data["status"] = "ai_labeled" if data.get("auto_labeled") else "reviewed"
            elif path:
                classes = self.classes()
                for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                    if not line.strip():
                        continue
                    values = line.split()
                    cls = int(values[0])
                    if not 0 <= cls < len(classes):
                        raise RepositoryError("YOLO class ID is outside the dataset registry")
                    polygon = None
                    if len(values) == 5:
                        xc, yc, w, h = map(float, values[1:])
                        left, top, right, bottom = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
                    elif len(values) >= 7 and len(values) % 2 == 1:
                        coords = list(map(float, values[1:]))
                        polygon = list(zip(coords[::2], coords[1::2]))
                        left, right = min(coords[::2]), max(coords[::2])
                        top, bottom = min(coords[1::2]), max(coords[1::2])
                    else:
                        raise RepositoryError("invalid YOLO annotation row")
                    data["boxes"].append(
                        {
                            "id": f"box-{i + 1}",
                            "class_name": classes[cls],
                            "class_id": cls,
                            "confidence": 1.0,
                            "source": "human",
                            "norm_left": max(0.0, left),
                            "norm_top": max(0.0, top),
                            "norm_right": min(1.0, right),
                            "norm_bottom": min(1.0, bottom),
                            "polygon_normalized": polygon,
                        }
                    )
                data["status"] = "reviewed"
            stat = image.stat()
            if (
                data.get("image_modified_ns", stat.st_mtime_ns) != stat.st_mtime_ns
                or data.get("image_size_bytes", stat.st_size) != stat.st_size
            ):
                data["status"] = "unreviewed"
            data["boxes"] = self.validate_boxes(data["boxes"], width, height)
            data.update(
                image_name=str(image.relative_to(self.root)),
                width=width,
                height=height,
                dataset_id=str(self.root),
            )
            return data

    def validate_boxes(self, boxes: list[dict], width: int, height: int) -> list[dict]:
        classes = self.classes()
        result = []
        ids = set()
        for value in boxes:
            value = dict(value)
            name = value.get("class_name", "").strip()
            cls = value.get("class_id")
            if name not in classes or (cls is not None and cls != classes.index(name)):
                raise RepositoryError(
                    "annotation class name and ID must match the dataset registry"
                )
            b = BoundingBox(
                *(float(value[k]) for k in ("norm_left", "norm_top", "norm_right", "norm_bottom"))
            )
            confidence = float(value.get("confidence", 1.0))
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise RepositoryError("confidence must be finite and between zero and one")
            identifier = str(value.get("id") or uuid4())
            if identifier in ids:
                raise RepositoryError("annotation IDs must be unique")
            ids.add(identifier)
            source = value.get("source", "human")
            if source == "manual":
                source = "human"
            if source not in {str(s) for s in AnnotationSource}:
                raise RepositoryError("unknown annotation source")
            value["source"] = source
            try:
                ReviewStatus(value.get("review_status", "accepted"))
            except ValueError as err:
                raise RepositoryError("unknown annotation review status") from err
            polygon = value.get("polygon_normalized")
            if polygon is not None:
                if len(polygon) < 3 or any(
                    len(pt) != 2 or any(not math.isfinite(c) or not 0 <= c <= 1 for c in pt)
                    for pt in polygon
                ):
                    raise RepositoryError("polygon needs at least three normalized finite points")
                value["polygon_pixels"] = [[x * width, y * height] for x, y in polygon]
            value.update(
                id=identifier,
                class_name=name,
                class_id=classes.index(name),
                confidence=confidence,
                x=b.left * width,
                y=b.top * height,
                width=b.width * width,
                height=b.height * height,
            )
            result.append(value)
        return result

    def save(
        self,
        image: Path,
        boxes: list[dict],
        *,
        status: str = "reviewed",
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        with self.lock, self._write_lock():
            image = self.image_path(image)
            with Image.open(image) as source:
                width, height = source.size
            path = self.canonical_path(image)
            current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            revision = current.get("revision", 0)
            if expected_revision is not None and expected_revision != revision:
                raise RevisionConflict("document changed; reload before saving")
            stat = image.stat()
            data = {
                "image_name": str(image.relative_to(self.root)),
                "dataset_id": str(self.root),
                "image_modified_ns": stat.st_mtime_ns,
                "image_size_bytes": stat.st_size,
                "boxes": self.validate_boxes(boxes, width, height),
                "status": status,
                "revision": revision + 1,
            }
            relative = image.relative_to(self.root)
            parts = list(relative.parts)
            if "images" in parts:
                parts[parts.index("images")] = "labels"
                projection = self.root.joinpath(*parts).with_suffix(".txt")
            else:
                projection = image.with_suffix(".txt")
            peers = [
                p
                for p in image.parent.glob(image.stem + ".*")
                if p.suffix.lower() in IMAGE_SUFFIXES
            ]
            if len(peers) == 1:
                if not projection.resolve().is_relative_to(self.root):
                    raise RepositoryError("training label path escapes dataset")
            atomic_text(path, json.dumps(data, indent=2))
            # A derived-label failure must not misreport an already committed edit.
            targets = [path.with_suffix(".txt")] + ([projection] if len(peers) == 1 else [])
            warnings = []
            for target in targets:
                try:
                    atomic_text(target, self.yolo(data["boxes"]))
                except OSError as err:
                    warnings.append(f"Cannot refresh training labels at {target}: {err}")
            if warnings:
                data["projection_warning"] = "; ".join(warnings)
            return data

    @staticmethod
    def yolo(boxes: list[dict]) -> str:
        return "".join(
            f"{b['class_id']} {(b['norm_left'] + b['norm_right']) / 2:.6f} "
            f"{(b['norm_top'] + b['norm_bottom']) / 2:.6f} "
            f"{b['norm_right'] - b['norm_left']:.6f} {b['norm_bottom'] - b['norm_top']:.6f}\n"
            for b in boxes
        )

    def document(self, image: Path, *, data: dict[str, Any] | None = None) -> AnnotationDocument:
        if data is None:
            data = self.read(image)
        annotations = []
        for b in data["boxes"]:
            try:
                identifier = UUID(b["id"])
            except ValueError:
                identifier = uuid5(NAMESPACE_URL, f"{self.root}/{data['image_name']}#{b['id']}")
            annotations.append(
                Annotation(
                    class_name=b["class_name"],
                    box=BoundingBox(
                        *(b[k] for k in ("norm_left", "norm_top", "norm_right", "norm_bottom"))
                    ),
                    confidence=b["confidence"],
                    source=AnnotationSource(b.get("source", "human")),
                    review_status=ReviewStatus(b.get("review_status", "accepted")),
                    occluded=b.get("occluded", False),
                    truncated=b.get("truncated", False),
                    annotation_id=identifier,
                    polygon_normalized=tuple(tuple(p) for p in b["polygon_normalized"])
                    if b.get("polygon_normalized")
                    else None,
                )
            )
        return AnnotationDocument(image, data["width"], data["height"], tuple(annotations))

    def save_document(
        self, document: AnnotationDocument, *, expected_revision: int | None = None
    ) -> dict[str, Any]:
        classes = self.register_classes([a.class_name for a in document.annotations])
        boxes = [
            {
                "id": str(a.annotation_id),
                "class_name": next(c for c in classes if c.casefold() == a.class_name.casefold()),
                "class_id": next(
                    i for i, c in enumerate(classes) if c.casefold() == a.class_name.casefold()
                ),
                "confidence": a.confidence if a.confidence is not None else 1.0,
                "source": str(a.source),
                "review_status": str(a.review_status),
                "occluded": a.occluded,
                "truncated": a.truncated,
                "norm_left": a.box.left,
                "norm_top": a.box.top,
                "norm_right": a.box.right,
                "norm_bottom": a.box.bottom,
                "polygon_normalized": a.polygon_normalized,
            }
            for a in document.annotations
        ]
        return self.save(document.image_path, boxes, expected_revision=expected_revision)
