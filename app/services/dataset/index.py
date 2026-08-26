"""SQLite-backed dataset index suitable for large image collections."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


class DatasetIndex:
    """Persistent image index with incremental upsert and paginated reads."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._database_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS images (
                path TEXT PRIMARY KEY,
                modified_ns INTEGER NOT NULL,
                width INTEGER,
                height INTEGER,
                difficulty REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'unreviewed'
            )"""
        )
        self._connection.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._connection.close()

    def __enter__(self) -> DatasetIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def scan(self, root: Path) -> int:
        """Incrementally index supported images under ``root``."""
        if not root.is_dir():
            raise NotADirectoryError(root)
        rows = []
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append((str(path), path.stat().st_mtime_ns))
        self._connection.executemany(
            "INSERT INTO images(path, modified_ns) VALUES(?, ?) "
            "ON CONFLICT(path) DO UPDATE SET modified_ns=excluded.modified_ns",
            rows,
        )
        self._connection.commit()
        LOGGER.info("indexed %d image paths under %s", len(rows), root)
        return len(rows)

    def iter_paths(self, page_size: int = 1000) -> Iterator[Path]:
        """Yield indexed image paths in stable pages."""
        if page_size < 1:
            raise ValueError("page_size must be greater than zero")
        offset = 0
        while True:
            rows = self._connection.execute(
                "SELECT path FROM images ORDER BY path LIMIT ? OFFSET ?", (page_size, offset)
            ).fetchall()
            if not rows:
                return
            yield from (Path(row["path"]) for row in rows)
            offset += len(rows)

    def set_metadata(self, path: Path, width: int, height: int) -> None:
        """Persist dimensions discovered by an image loader."""
        self._connection.execute(
            "UPDATE images SET width=?, height=? WHERE path=?", (width, height, str(path))
        )
        self._connection.commit()

    def set_difficulty(self, path: Path, difficulty: float, status: str = "unreviewed") -> None:
        """Persist a validated difficulty score used by the review queue."""
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("difficulty must be between 0 and 1")
        self._connection.execute(
            "UPDATE images SET difficulty=?, status=? WHERE path=?", (difficulty, status, str(path))
        )
        self._connection.commit()

    def delete(self, path: Path) -> bool:
        """Remove one image path from the database index."""
        cursor = self._connection.execute(
            "DELETE FROM images WHERE path=?", (str(path),)
        )
        self._connection.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Return total number of indexed images."""
        row = self._connection.execute("SELECT COUNT(*) FROM images").fetchone()
        return int(row[0]) if row else 0

    def hardest(self, limit: int = 100) -> list[Path]:
        """Return the highest-difficulty images first."""
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        rows = self._connection.execute(
            "SELECT path FROM images ORDER BY difficulty DESC, path LIMIT ?", (limit,)
        ).fetchall()
        return [Path(row["path"]) for row in rows]

