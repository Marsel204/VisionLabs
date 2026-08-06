"""Data models for crop-assisted annotation sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.services.annotation.domain import AnnotationDocument


@dataclass(frozen=True, slots=True)
class CropRegion:
    """A pixel crop and its non-overlapping ownership region."""

    index: int
    left: int
    top: int
    right: int
    bottom: int
    core_left: int
    core_top: int
    core_right: int
    core_bottom: int
    image_path: Path

    @property
    def width(self) -> int:
        """Return crop width in pixels."""
        return self.right - self.left

    @property
    def height(self) -> int:
        """Return crop height in pixels."""
        return self.bottom - self.top

    def owns_center(self, x: float, y: float) -> bool:
        """Return whether an original-image center belongs to this crop."""
        return self.core_left <= x <= self.core_right and self.core_top <= y <= self.core_bottom


@dataclass(frozen=True, slots=True)
class CropSession:
    """Temporary crop documents and their source-image context."""

    original: AnnotationDocument
    regions: tuple[CropRegion, ...]
    documents: tuple[AnnotationDocument, ...]
    tile_size: int
    overlap: float
