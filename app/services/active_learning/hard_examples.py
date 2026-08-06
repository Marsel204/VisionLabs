"""Difficulty scoring and review queue management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.services.active_learning.active_learning_models import DifficultyResult
from app.services.dataset.index import DatasetIndex


@dataclass(frozen=True, slots=True)
class DifficultySignals:
    """Signals used to prioritize hard images."""

    low_confidence: float = 0.0
    missed_motorcycles: float = 0.0
    density: float = 0.0
    occlusion: float = 0.0
    duplicate_detections: float = 0.0

    def score(self) -> float:
        """Return weighted difficulty in the normalized range [0, 1]."""
        values = (
            max(0.0, min(1.0, self.low_confidence)),
            max(0.0, min(1.0, self.missed_motorcycles)),
            max(0.0, min(1.0, self.density)),
            max(0.0, min(1.0, self.occlusion)),
            max(0.0, min(1.0, self.duplicate_detections)),
        )
        return min(
            1.0,
            sum(
                value * weight
                for value, weight in zip(values, (0.25, 0.3, 0.15, 0.2, 0.1), strict=True)
            ),
        )


class HardExampleManager:
    """Calculates and persists difficulty scores through the dataset index."""

    def __init__(self, index: DatasetIndex) -> None:
        self._index = index
        self._collections: dict[str, set[Path]] = {}

    def record(self, image: Path, signals: DifficultySignals) -> float:
        """Persist one image's score and return it."""
        score = signals.score()
        self._index.set_difficulty(image, score)
        return score

    def prioritize(self, limit: int = 100) -> list[Path]:
        """Return images ordered by descending difficulty."""
        return self._index.hardest(limit)

    def assign_collections(self, result: DifficultyResult) -> frozenset[str]:
        """Add an image to every collection indicated by its difficulty features."""
        for collection in result.collections:
            self._collections.setdefault(collection, set()).add(result.image_path)
        return result.collections

    def collection(self, name: str) -> tuple[Path, ...]:
        """Return paths in a named hard-example collection."""
        return tuple(sorted(self._collections.get(name, set())))
