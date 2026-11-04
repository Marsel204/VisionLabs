"""Process-local, rebuildable repository caches shared across dataset handles.

Registry entries are checked by path/mtime_ns/size on every access. Image catalogs
check the identity and mtime of every traversed directory before reuse, so ordinary
external additions, removals and renames are visible without a rescan. Explicit
scan/add/delete/switch operations also invalidate them. Filesystems that preserve
or restore directory mtimes require an explicit rescan; file confinement is always
checked at use time. No annotation documents or image dimensions live here.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


@dataclass
class DatasetCache:
    lock: Any = field(default_factory=threading.RLock)
    registries: dict[Path, tuple[tuple[int, int], Any]] = field(default_factory=dict)
    directories: dict[Path, tuple[int, int, int]] | None = None
    stems: dict[str, int] = field(default_factory=dict)
    peers: dict[tuple[Path, str], int] = field(default_factory=dict)

    def invalidate_images(self) -> None:
        with self.lock:
            self.directories = None
            self.stems.clear()
            self.peers.clear()

    def image_counts(self, root: Path, image: Path) -> tuple[int, int]:
        with self.lock:
            if self.directories is None or not self._unchanged():
                self._rebuild(root)
            same_parent = self.peers.get((image.parent, image.stem), 0)
            if image.parent not in self.directories:
                # Direct repository access also permits images in excluded
                # folders. Preserve local ambiguity checks without indexing
                # exports, labels, or hidden trees into the dataset catalog.
                same_parent = sum(
                    p.suffix.lower() in IMAGE_SUFFIXES
                    for p in image.parent.glob(image.stem + ".*")
                )
            return self.stems.get(image.stem, 0), same_parent

    @staticmethod
    def _identity(path: Path) -> tuple[int, int, int]:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_mtime_ns

    def _unchanged(self) -> bool:
        try:
            return all(
                self._identity(path) == identity for path, identity in self.directories.items()
            )
        except OSError:
            return False

    def _rebuild(self, root: Path) -> None:
        directories, stems, peers = {}, {}, {}
        pending = [root]
        while pending:
            parent = pending.pop()
            # Snapshot before enumeration so a concurrent directory mutation is
            # noticed by the next validation.
            directories[parent] = self._identity(parent)
            with os.scandir(parent) as entries:
                for entry in entries:
                    path = parent / entry.name
                    if path.suffix.lower() in IMAGE_SUFFIXES:
                        key = (parent, path.stem)
                        peers[key] = peers.get(key, 0) + 1
                        if path.is_file() and path.resolve().is_relative_to(root):
                            stems[path.stem] = stems.get(path.stem, 0) + 1
                    if (
                        entry.is_dir(follow_symlinks=False)
                        and entry.name not in {"exports", "labels"}
                        and not entry.name.startswith(".")
                    ):
                        pending.append(path)
        self.directories, self.stems, self.peers = directories, stems, peers


_CACHES: dict[Path, DatasetCache] = {}
_GUARD = threading.Lock()


def dataset_cache(root: Path) -> DatasetCache:
    with _GUARD:
        return _CACHES.setdefault(root.resolve(), DatasetCache())


def invalidate_image_catalog(root: Path) -> None:
    dataset_cache(root).invalidate_images()


def invalidate_image_catalog_for_paths(paths: list[Path]) -> None:
    """Index database files may live outside the dataset they describe."""
    with _GUARD:
        affected = [
            cache for root, cache in _CACHES.items()
            if any(path.is_relative_to(root) for path in paths)
        ]
    for cache in affected:
        cache.invalidate_images()
