"""Stable contracts shared by model adapters and inference services."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from app.services.annotation.domain import AnnotationSource, BoundingBox


@dataclass(frozen=True, slots=True)
class Detection:
    """A model detection before it becomes an annotation."""

    class_name: str
    box: BoundingBox
    confidence: float
    source: AnnotationSource
    mask: np.ndarray | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class GroundingDinoModel(Protocol):
    """Candidate detector contract."""

    def predict(self, images: Sequence[Path], prompt: str) -> list[list[Detection]]: ...


class Sam2Model(Protocol):
    """Mask refinement contract."""

    def segment(self, image: Path, boxes: Sequence[BoundingBox]) -> list[np.ndarray]: ...


class YoloModel(Protocol):
    """Verification detector contract."""

    def predict(self, images: Sequence[Path]) -> list[list[Detection]]: ...


class ModelRuntimeError(RuntimeError):
    """Raised when an optional model runtime cannot be loaded or executed."""
