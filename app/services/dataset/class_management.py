"""Backed-up class registry edits with annotation remapping and write rollback."""

from __future__ import annotations

import json
import math
from pathlib import Path
from uuid import uuid4

import yaml

from app.services.dataset.index import IMAGE_SUFFIXES
from app.services.dataset.repository import AnnotationRepository, RepositoryError, atomic_text


def _annotation_paths(repo: AnnotationRepository) -> set[Path]:
    paths = set()
    images = []
    for path in repo.root.rglob("*"):
        relative = path.relative_to(repo.root)
        canonical = relative.parts[:2] == (".visionlab", "annotations")
        if not canonical and any(
            part.startswith(".") or part == "exports" for part in relative.parts[:-1]
        ):
            continue
        if not path.is_file():
            continue
        if canonical or "labels" in relative.parts[:-1]:
            if path.suffix in {".json", ".txt"} and path.name != "classes.txt":
                paths.add(path)
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
    for image in images:
        for suffix in (".json", ".txt"):
            candidate = image.with_suffix(suffix)
            if candidate.is_file():
                paths.add(candidate)
    for path in paths:
        if not path.resolve().is_relative_to(repo.root):
            raise RepositoryError("Annotation path escapes dataset")
    return paths


def change_class(
    repo: AnnotationRepository,
    action: str,
    name: str,
    target: str | None,
    expected_classes: list[str],
    *,
    delete_annotations: bool = False,
) -> dict:
    with repo.lock, repo._write_lock():
        classes = repo.classes()
        if classes != expected_classes:
            raise RepositoryError("Daftar kelas berubah. Muat ulang sebelum mencoba lagi.")
        if name not in classes or action not in {"rename", "delete"}:
            raise RepositoryError("Kelas atau tindakan tidak valid.")
        target = target.strip() if target is not None else None
        if action == "rename":
            if not target or any(
                item.casefold() == target.casefold() for item in classes if item != name
            ):
                raise RepositoryError("Nama kelas harus diisi dan tidak boleh duplikat.")
            updated = [target if item == name else item for item in classes]
        else:
            if target is not None and (target not in classes or target == name):
                raise RepositoryError("Pilih kelas pengganti yang berbeda.")
            if target is not None and delete_annotations:
                raise RepositoryError("Pilih pemindahan atau penghapusan anotasi, bukan keduanya.")
            updated = [item for item in classes if item != name]

        def mapped(old_name):
            if old_name != name:
                return old_name
            if action == "rename" or target is not None:
                return target
            if not delete_annotations:
                raise RepositoryError(
                    "Kelas masih digunakan. Pilih kelas pengganti atau hapus anotasinya."
                )
            return None

        changes = {}
        for path in sorted(_annotation_paths(repo)):
            original = path.read_bytes().decode("utf-8")
            if path.suffix == ".json":
                try:
                    document = json.loads(original)
                    if not isinstance(document, dict) or not isinstance(
                        document.get("boxes"), list
                    ):
                        raise ValueError("expected an annotation document with boxes")
                    boxes = []
                    for box in document["boxes"]:
                        old_name = box["class_name"]
                        if old_name not in classes or box.get(
                            "class_id", classes.index(old_name)
                        ) != classes.index(old_name):
                            raise ValueError("class name and ID disagree")
                        new_name = mapped(old_name)
                        if new_name is not None:
                            boxes.append(
                                {**box, "class_name": new_name, "class_id": updated.index(new_name)}
                            )
                    if boxes == document["boxes"]:
                        continue
                    document["boxes"] = boxes
                    document["revision"] = document.get("revision", 0) + 1
                    value = json.dumps(document, indent=2)
                except (KeyError, TypeError, ValueError) as error:
                    if isinstance(error, RepositoryError):
                        raise
                    raise RepositoryError(f"Anotasi tidak valid: {path}: {error}") from error
            else:
                lines = []
                for line in original.splitlines(keepends=True):
                    values = line.split()
                    if not values:
                        lines.append(line)
                        continue
                    try:
                        class_id = int(values[0])
                        if not 0 <= class_id < len(classes):
                            raise ValueError("class ID outside registry")
                        if len(values) != 5 and not (len(values) >= 7 and len(values) % 2 == 1):
                            raise ValueError("invalid YOLO row")
                        if not all(math.isfinite(float(value)) for value in values[1:]):
                            raise ValueError("non-finite coordinate")
                    except ValueError as error:
                        raise RepositoryError(f"Label YOLO tidak valid: {path}: {error}") from error
                    new_name = mapped(classes[class_id])
                    if new_name is not None:
                        new_id = updated.index(new_name)
                        lines.append(
                            line if new_id == class_id else line.replace(values[0], str(new_id), 1)
                        )
                value = "".join(lines)
            if value != original:
                changes[path] = value

        registry = repo.root / "data.yaml"
        for path in (repo.root / "dataset.yaml", registry):
            if path != registry and not path.is_file():
                continue
            if not path.resolve().is_relative_to(repo.root):
                raise RepositoryError("Class registry path escapes dataset")
            data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
            data = data or {}
            if not isinstance(data, dict):
                raise RepositoryError(f"Registry kelas tidak valid: {path}")
            data["names"] = updated
            if "nc" in data:
                data["nc"] = len(updated)
            changes[path] = yaml.safe_dump(data, sort_keys=False)

        backup = repo.root / ".visionlab" / "class-backups" / str(uuid4())
        if not backup.resolve().is_relative_to(repo.root):
            raise RepositoryError("Backup path escapes dataset")
        originals = {
            path: path.read_bytes().decode("utf-8") if path.exists() else None for path in changes
        }
        manifest = {"before": classes, "after": updated, "created": []}
        for path, original in originals.items():
            relative = str(path.relative_to(repo.root))
            if original is None:
                manifest["created"].append(relative)
            else:
                atomic_text(backup / relative, original)
        atomic_text(backup / "manifest.json", json.dumps(manifest, indent=2))
        written = []
        try:
            for path, value in changes.items():
                written.append(path)
                atomic_text(path, value)
        except OSError as error:
            failures = []
            for path in reversed(written):
                original = originals[path]
                try:
                    if original is None:
                        path.unlink(missing_ok=True)
                    else:
                        atomic_text(path, original)
                except OSError:
                    failures.append(str(path))
            if failures:
                raise RepositoryError(
                    f"Pemulihan belum selesai. Jangan lanjutkan anotasi. Pulihkan backup: {backup}"
                ) from error
            raise
        return {"classes": updated, "backup": str(backup.relative_to(repo.root))}
