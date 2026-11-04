"""SQLite-backed dataset index suitable for large image collections."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.services.dataset.cache import (
    IMAGE_SUFFIXES, invalidate_image_catalog, invalidate_image_catalog_for_paths,
)

LOGGER = logging.getLogger(__name__)
EXCLUDE_DIRS = frozenset({"exports", "labels", ".git", ".cache", "__pycache__"})



class DatasetIndex:
    """Persistent image index with incremental upsert and paginated reads."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._lock:
            self._connection = sqlite3.connect(self._database_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA cache_size=-32768")   # 32MB page cache
            self._connection.execute("PRAGMA temp_store=MEMORY")
            self._connection.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS images (
                    path TEXT PRIMARY KEY,
                    modified_ns INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL DEFAULT -1,
                    width INTEGER,
                    height INTEGER,
                    difficulty REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'unreviewed',
                    annotation_count INTEGER NOT NULL DEFAULT -1,
                    class_histogram TEXT
                )"""
            )
            columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(images)")}
            for column, declaration in (("annotation_count", "INTEGER NOT NULL DEFAULT -1"),
                                        ("size_bytes", "INTEGER NOT NULL DEFAULT -1"),
                                        ("class_histogram", "TEXT")):
                if column not in columns:
                    self._connection.execute(f"ALTER TABLE images ADD COLUMN {column} {declaration}")
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS annotation_cache_state ("
                "id INTEGER PRIMARY KEY CHECK(id=1), generation INTEGER NOT NULL)"
            )
            self._connection.execute("INSERT OR IGNORE INTO annotation_cache_state VALUES(1, 0)")
            # Create indexes for fast filtering and sorting
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_images_status ON images(status)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_images_difficulty ON images(difficulty DESC)"
            )
            self._connection.commit()


    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> DatasetIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def scan(self, root: Path, *, refresh_annotations: bool = False) -> int:
        """Index changed images, preserving persisted annotation caches on open.

        Explicit refresh also invalidates label-derived caches and in-flight fills,
        because label files can change independently of their images.
        """
        if not root.is_dir():
            raise NotADirectoryError(root)
        resolved_root = root.resolve()
        invalidate_image_catalog(resolved_root)
        rows = []
        for path in resolved_root.rglob("*"):
            if not (path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
                continue
            try:
                parts = path.relative_to(resolved_root).parts[:-1]
                if any(p in EXCLUDE_DIRS or p.startswith(".") for p in parts):
                    continue
            except Exception:
                pass
            if path.resolve().is_relative_to(resolved_root):
                stat = path.stat()
                rows.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))

        with self._lock:
            # Prune non-existent paths, paths inside excluded directories, or non-resolved duplicates
            all_rows = self._connection.execute("SELECT path FROM images").fetchall()
            prune_paths = []
            seen_resolved = set()
            for row in all_rows:
                p_str = row["path"]
                p = Path(p_str)
                if not p.is_file():
                    prune_paths.append((p_str,))
                    continue
                resolved_str = str(p.resolve())
                if resolved_str in seen_resolved or p_str != resolved_str:
                    prune_paths.append((p_str,))
                    continue
                try:
                    parts = p.resolve().relative_to(resolved_root).parts[:-1]
                    if any(part in EXCLUDE_DIRS or part.startswith(".") for part in parts):
                        prune_paths.append((p_str,))
                        continue
                except ValueError:
                    prune_paths.append((p_str,))
                    continue
                seen_resolved.add(resolved_str)

            if prune_paths:
                self._connection.executemany("DELETE FROM images WHERE path=?", prune_paths)

            if refresh_annotations:
                self._connection.execute("UPDATE images SET annotation_count=-1, class_histogram=NULL")

            cursor = self._connection.executemany(
                "INSERT INTO images(path, modified_ns, size_bytes) VALUES(?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET "
                "width=CASE WHEN (images.modified_ns != excluded.modified_ns OR images.size_bytes != excluded.size_bytes) THEN NULL ELSE images.width END, "
                "height=CASE WHEN (images.modified_ns != excluded.modified_ns OR images.size_bytes != excluded.size_bytes) THEN NULL ELSE images.height END, "
                "status=CASE WHEN (images.modified_ns != excluded.modified_ns OR images.size_bytes != excluded.size_bytes) THEN 'unreviewed' ELSE images.status END, "
                "annotation_count=CASE WHEN (images.modified_ns != excluded.modified_ns OR images.size_bytes != excluded.size_bytes) THEN -1 ELSE images.annotation_count END, "
                "class_histogram=CASE WHEN (images.modified_ns != excluded.modified_ns OR images.size_bytes != excluded.size_bytes) THEN NULL ELSE images.class_histogram END, "
                "difficulty=CASE WHEN (images.modified_ns != excluded.modified_ns OR images.size_bytes != excluded.size_bytes) THEN 0 ELSE images.difficulty END, "
                "modified_ns=excluded.modified_ns, size_bytes=excluded.size_bytes "
                "WHERE images.modified_ns != excluded.modified_ns OR images.size_bytes != excluded.size_bytes",
                rows,
            )
            if refresh_annotations or prune_paths or cursor.rowcount:
                self._bump_annotation_generation()
            self._connection.commit()
            LOGGER.info("indexed %d image paths under %s (pruned %d)", len(rows), resolved_root, len(prune_paths))
            return len(rows)

    def add_paths(self, items: list[tuple[Path, int, int]]) -> int:
        """Atomically upsert known image paths and dimensions; return the index total.

        Unlike a scan, this does not traverse the dataset or prune unrelated rows.
        Callers must supply verified, confined image paths and their dimensions.
        """
        invalidate_image_catalog_for_paths([path.resolve() for path, _, _ in items])
        rows = []
        for path, width, height in items:
            path = path.resolve()
            stat = path.stat()
            rows.append((str(path), stat.st_mtime_ns, stat.st_size, width, height))
        with self._lock, self._connection:
            self._connection.executemany(
                "INSERT INTO images(path, modified_ns, size_bytes, width, height) "
                "VALUES(?, ?, ?, ?, ?) ON CONFLICT(path) DO UPDATE SET "
                "width=excluded.width, height=excluded.height, "
                "status=CASE WHEN (images.modified_ns != excluded.modified_ns OR images.size_bytes != excluded.size_bytes) THEN 'unreviewed' ELSE images.status END, "
                "annotation_count=CASE WHEN (images.modified_ns != excluded.modified_ns OR images.size_bytes != excluded.size_bytes) THEN -1 ELSE images.annotation_count END, "
                "class_histogram=CASE WHEN (images.modified_ns != excluded.modified_ns OR images.size_bytes != excluded.size_bytes) THEN NULL ELSE images.class_histogram END, "
                "difficulty=CASE WHEN (images.modified_ns != excluded.modified_ns OR images.size_bytes != excluded.size_bytes) THEN 0 ELSE images.difficulty END, "
                "modified_ns=excluded.modified_ns, size_bytes=excluded.size_bytes",
                rows,
            )
            self._bump_annotation_generation()
            return int(self._connection.execute("SELECT COUNT(*) FROM images").fetchone()[0])

    def iter_paths(self, page_size: int = 1000) -> Iterator[Path]:
        """Yield indexed image paths in stable pages."""
        if page_size < 1:
            raise ValueError("page_size must be greater than zero")
        offset = 0
        while True:
            with self._lock:
                rows = self._connection.execute(
                    "SELECT path FROM images ORDER BY path LIMIT ? OFFSET ?", (page_size, offset)
                ).fetchall()
            if not rows:
                return
            yield from (Path(row["path"]) for row in rows)
            offset += len(rows)

    def set_metadata(self, path: Path, width: int, height: int) -> None:
        """Persist dimensions discovered by an image loader."""
        with self._lock:
            self._connection.execute(
                "UPDATE images SET width=?, height=? WHERE path=?", (width, height, str(path))
            )
            self._connection.commit()

    def set_metadata_batch(self, items: list[tuple[Path, int, int]]) -> None:
        """Persist dimensions for multiple images in a single atomic transaction."""
        if not items:
            return
        rows = [(w, h, str(p)) for p, w, h in items]
        with self._lock:
            self._connection.executemany(
                "UPDATE images SET width=?, height=? WHERE path=?", rows
            )
            self._connection.commit()

    def set_listing_cache_batch(
        self, metadata: list[tuple[Path, int, int]],
        annotations: list[tuple[Path, str, int, float, dict[str, int]]],
        *, expected_generation: int,
    ) -> bool:
        """Publish a page's read results in one transaction, rejecting stale fills."""
        if not metadata and not annotations:
            return True
        guard = " AND (SELECT generation FROM annotation_cache_state WHERE id=1)=?"
        with self._lock, self._connection:
            dimensions = self._connection.executemany(
                "UPDATE images SET width=?, height=? WHERE path=?" + guard,
                [(w, h, str(path), expected_generation) for path, w, h in metadata],
            )
            counts = self._connection.executemany(
                "UPDATE images SET status=?, annotation_count=?, difficulty=?, class_histogram=? "
                "WHERE path=?" + guard,
                [(status, count, difficulty, json.dumps(histogram), str(path), expected_generation)
                 for path, status, count, difficulty, histogram in annotations],
            )
            return dimensions.rowcount == len(metadata) and counts.rowcount == len(annotations)

    def set_difficulty(self, path: Path, difficulty: float, status: str = "unreviewed") -> None:
        """Persist a validated difficulty score used by the review queue."""
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("difficulty must be between 0 and 1")
        with self._lock:
            self._connection.execute(
                "UPDATE images SET difficulty=?, status=? WHERE path=?", (difficulty, status, str(path))
            )
            self._connection.commit()

    def delete(self, path: Path) -> bool:
        """Remove one image path from the database index."""
        invalidate_image_catalog_for_paths([path.resolve()])
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM images WHERE path=?", (str(path),)
            )
            if cursor.rowcount:
                self._bump_annotation_generation()
            self._connection.commit()
            return cursor.rowcount > 0

    def count(self) -> int:
        """Return total number of indexed images."""
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) FROM images").fetchone()
            return int(row[0]) if row else 0

    def hardest(self, limit: int = 100) -> list[Path]:
        """Return the highest-difficulty images first."""
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        with self._lock:
            rows = self._connection.execute(
                "SELECT path FROM images ORDER BY difficulty DESC, path LIMIT ?", (limit,)
            ).fetchall()
            return [Path(row["path"]) for row in rows]

    def get_image(self, path: Path) -> dict[str, Any] | None:
        """Fetch metadata, status, and difficulty for a single image."""
        with self._lock:
            row = self._connection.execute(
                "SELECT path, width, height, difficulty, status, modified_ns, annotation_count FROM images WHERE path=?",
                (str(path),),
            ).fetchone()
            return dict(row) if row else None

    def find_by_name(self, filename: str) -> dict[str, Any] | None:
        """Find an image record by filename ending or exact path."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT path, width, height, difficulty, status, modified_ns FROM images WHERE path = ? OR path LIKE ?",
                (filename, f"%/{filename}"),
            ).fetchall()
            if len(rows) > 1:
                raise ValueError("ambiguous image basename")
            return dict(rows[0]) if rows else None

    def invalidate_annotations(self) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE images SET annotation_count=-1, class_histogram=NULL, difficulty=0"
            )
            self._bump_annotation_generation()
            self._connection.commit()

    def set_annotation_count(self, path: Path, count: int) -> None:
        """Cache the number of annotations for an image to avoid repeated filesystem probes."""
        with self._lock:
            self._connection.execute(
                "UPDATE images SET annotation_count=?, class_histogram=? WHERE path=?",
                (count, "{}" if count == 0 else None, str(path)),
            )
            self._bump_annotation_generation()
            self._connection.commit()

    def set_annotation_counts_batch(self, items: list[tuple[Path, int]]) -> None:
        """Batch-persist annotation counts in a single locked transaction."""
        if not items:
            return
        rows = [(cnt, "{}" if cnt == 0 else None, str(p)) for p, cnt in items]
        with self._lock:
            self._connection.executemany(
                "UPDATE images SET annotation_count=?, class_histogram=? WHERE path=?", rows
            )
            self._bump_annotation_generation()
            self._connection.commit()

    def set_status_and_count(
        self, path: Path, status: str, count: int, difficulty: float = 0.0,
        *, boxes: list[dict[str, Any]] | None = None,
        expected_generation: int | None = None,
    ) -> bool:
        """Persist read/save results together; absent boxes invalidate a positive histogram."""
        histogram = (
            json.dumps(Counter(box["class_name"] for box in boxes))
            if boxes is not None else ("{}" if count == 0 else None)
        )
        with self._lock:
            query = ("UPDATE images SET status=?, annotation_count=?, difficulty=?, "
                     "class_histogram=? WHERE path=?")
            params = (status, count, difficulty, histogram, str(path))
            if expected_generation is not None:
                query += " AND (SELECT generation FROM annotation_cache_state WHERE id=1)=?"
                params += (expected_generation,)
            cursor = self._connection.execute(query, params)
            if boxes is None:
                self._bump_annotation_generation()
            self._connection.commit()
            return cursor.rowcount > 0

    def _bump_annotation_generation(self) -> None:
        """Called inside the same transaction as an invalidation."""
        self._connection.execute(
            "UPDATE annotation_cache_state SET generation=generation+1 WHERE id=1"
        )

    def annotation_generation(self) -> int:
        """Detect invalidation/rescan races across handles and processes."""
        with self._lock:
            return int(self._connection.execute(
                "SELECT generation FROM annotation_cache_state WHERE id=1"
            ).fetchone()[0])

    def unknown_histograms(self, page_size: int = 1000) -> Iterator[dict[str, Any]]:
        """Yield only rows needing annotation reads, skipping known empty images.

        Use a path cursor because filling a page removes it from this query.
        """
        after = ""
        while True:
            with self._lock:
                rows = self._connection.execute(
                    "SELECT path, difficulty FROM images WHERE class_histogram IS NULL "
                    "AND COALESCE(annotation_count, -1) != 0 AND path > ? ORDER BY path LIMIT ?",
                    (after, page_size),
                ).fetchall()
            if not rows:
                return
            yield from (dict(row) for row in rows)
            after = rows[-1]["path"]

    def class_counts(self) -> dict[str, int]:
        """Aggregate rebuildable per-image histograms entirely in SQLite."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT entry.key AS name, SUM(entry.value) AS count "
                "FROM images, json_each(images.class_histogram) AS entry GROUP BY entry.key"
            ).fetchall()
            return {row["name"]: int(row["count"]) for row in rows}

    def list_records(
        self,
        status: str | None = None,
        order_by: str = "path",
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query indexed images with optional status filtering and sorting."""
        query = "SELECT path, width, height, difficulty, status, modified_ns, annotation_count FROM images"
        params: list[Any] = []
        if status:
            query += " WHERE status=?"
            params.append(status)

        if order_by == "difficulty":
            query += " ORDER BY difficulty DESC, path"
        elif order_by == "modified":
            query += " ORDER BY modified_ns DESC, path"
        else:
            query += " ORDER BY path"

        query += " LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])

        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
            return [dict(r) for r in rows if r is not None and r["path"] is not None]

    def stats(self) -> dict[str, int]:
        """Return counts grouped by review status."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT status, COUNT(*) as count FROM images GROUP BY status"
            ).fetchall()
        result = {"total": 0, "unreviewed": 0, "reviewed": 0, "ai_labeled": 0}
        for r in rows:
            st = r["status"]
            cnt = int(r["count"])
            result[st] = cnt
            result["total"] += cnt
        return result
