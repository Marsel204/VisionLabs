"""SQLite-backed dataset index suitable for large image collections."""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
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
                    width INTEGER,
                    height INTEGER,
                    difficulty REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'unreviewed',
                    annotation_count INTEGER NOT NULL DEFAULT -1
                )"""
            )
            # Migration: add annotation_count column to existing databases
            try:
                self._connection.execute("ALTER TABLE images ADD COLUMN annotation_count INTEGER NOT NULL DEFAULT -1")
                LOGGER.info("Migrated images table: added annotation_count column")
            except Exception:
                pass  # Column already exists — normal case
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

    def scan(self, root: Path) -> int:
        """Incrementally index supported images under ``root``."""
        if not root.is_dir():
            raise NotADirectoryError(root)
        resolved_root = root.resolve()
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
            rows.append((str(path.resolve()), path.stat().st_mtime_ns))

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
                except Exception:
                    pass
                seen_resolved.add(resolved_str)

            if prune_paths:
                self._connection.executemany("DELETE FROM images WHERE path=?", prune_paths)

            self._connection.executemany(
                "INSERT INTO images(path, modified_ns) VALUES(?, ?) "
                "ON CONFLICT(path) DO UPDATE SET modified_ns=excluded.modified_ns",
                rows,
            )
            self._connection.commit()
            LOGGER.info("indexed %d image paths under %s (pruned %d)", len(rows), resolved_root, len(prune_paths))
            return len(rows)

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
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM images WHERE path=?", (str(path),)
            )
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
                "SELECT path, width, height, difficulty, status, modified_ns FROM images WHERE path=?",
                (str(path),),
            ).fetchone()
            return dict(row) if row else None

    def find_by_name(self, filename: str) -> dict[str, Any] | None:
        """Find an image record by filename ending or exact path."""
        with self._lock:
            row = self._connection.execute(
                "SELECT path, width, height, difficulty, status, modified_ns FROM images WHERE path = ? OR path LIKE ? LIMIT 1",
                (filename, f"%/{filename}"),
            ).fetchone()
            return dict(row) if row else None

    def set_annotation_count(self, path: Path, count: int) -> None:
        """Cache the number of annotations for an image to avoid repeated filesystem probes."""
        with self._lock:
            self._connection.execute(
                "UPDATE images SET annotation_count=? WHERE path=?", (count, str(path))
            )
            self._connection.commit()

    def set_annotation_counts_batch(self, items: list[tuple[Path, int]]) -> None:
        """Batch-persist annotation counts in a single locked transaction."""
        if not items:
            return
        rows = [(cnt, str(p)) for p, cnt in items]
        with self._lock:
            self._connection.executemany(
                "UPDATE images SET annotation_count=? WHERE path=?", rows
            )
            self._connection.commit()

    def set_status_and_count(self, path: Path, status: str, count: int, difficulty: float = 0.0) -> None:
        """Atomic update of status, annotation count, and difficulty in a single write."""
        with self._lock:
            self._connection.execute(
                "UPDATE images SET status=?, annotation_count=?, difficulty=? WHERE path=?",
                (status, count, difficulty, str(path)),
            )
            self._connection.commit()

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



