"""Optional-runtime model adapters using injected predictor callables.

The concrete libraries have incompatible loading APIs and are intentionally loaded by
the composition root. These adapters provide one stable application-facing boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from app.models.contracts import Detection, ModelRuntimeError
from app.services.annotation.domain import AnnotationSource, BoundingBox


class GroundingDinoAdapter:
    """Adapter around a configured Grounding DINO 1.5 predictor."""

    def __init__(
        self, predictor: Callable[[Sequence[Path], str], Sequence[Sequence[dict[str, Any]]]]
    ) -> None:
        self._predictor = predictor

    def predict(self, images: Sequence[Path], prompt: str) -> list[list[Detection]]:
        try:
            return [
                [self._detection(item, AnnotationSource.GROUNDING_DINO) for item in batch]
                for batch in self._predictor(images, prompt)
            ]
        except Exception as error:
            raise ModelRuntimeError(f"Grounding DINO inference failed: {error}") from error

    @staticmethod
    def _detection(item: dict[str, Any], source: AnnotationSource) -> Detection:
        return Detection(
            class_name=str(item["class_name"]),
            box=BoundingBox(*map(float, item["box"])),
            confidence=float(item["confidence"]),
            source=source,
        )


class Sam2Adapter:
    """Adapter around a configured SAM2 predictor."""

    def __init__(
        self, predictor: Callable[[Path, Sequence[BoundingBox]], Sequence[np.ndarray]]
    ) -> None:
        self._predictor = predictor

    def segment(self, image: Path, boxes: Sequence[BoundingBox]) -> list[np.ndarray]:
        try:
            return list(self._predictor(image, boxes))
        except Exception as error:
            raise ModelRuntimeError(f"SAM2 segmentation failed: {error}") from error


class YoloAdapter:
    """Adapter around a configured Ultralytics YOLO11 predictor."""

    def __init__(
        self, predictor: Callable[[Sequence[Path]], Sequence[Sequence[dict[str, Any]]]]
    ) -> None:
        self._predictor = predictor

    def predict(self, images: Sequence[Path]) -> list[list[Detection]]:
        try:
            return [
                [
                    Detection(
                        class_name=str(item["class_name"]),
                        box=BoundingBox(*map(float, item["box"])),
                        confidence=float(item["confidence"]),
                        source=AnnotationSource.YOLO,
                    )
                    for item in batch
                ]
                for batch in self._predictor(images)
            ]
        except Exception as error:
            raise ModelRuntimeError(f"YOLO inference failed: {error}") from error
