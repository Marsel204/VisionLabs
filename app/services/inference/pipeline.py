"""Orchestration of candidate generation, segmentation, verification, and fusion."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.models.contracts import Detection, GroundingDinoModel, Sam2Model, YoloModel
from app.services.annotation.domain import Annotation, AnnotationSource
from app.services.fusion.fusion import FusedDetection, fuse_detections

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Output for one image, including detections requiring human review."""

    image_path: Path
    detections: tuple[Annotation, ...]
    disagreements: tuple[FusedDetection, ...]


class InferencePipeline:
    """Dependency-injected model pipeline with batch execution and progress callbacks."""

    def __init__(
        self,
        grounding_dino: GroundingDinoModel,
        sam2: Sam2Model,
        yolo: YoloModel,
        prompt: str = "motorcycle. car. bus. truck.",
    ) -> None:
        self._grounding_dino = grounding_dino
        self._sam2 = sam2
        self._yolo = yolo
        self._prompt = prompt

    def run(
        self,
        images: list[Path],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[InferenceResult]:
        """Run the pipeline in batches and report completed image counts."""
        if not images:
            return []
        candidates = self._grounding_dino.predict(images, self._prompt)
        verified = self._yolo.predict(images)
        results: list[InferenceResult] = []
        for index, image in enumerate(images):
            image_candidates = candidates[index]
            boxes = [item.box for item in image_candidates]
            masks = self._sam2.segment(image, boxes) if boxes else []
            refined = [
                Detection(
                    class_name=item.class_name,
                    box=item.box,
                    confidence=item.confidence,
                    source=AnnotationSource.SAM2,
                    mask=masks[item_index] if item_index < len(masks) else None,
                )
                for item_index, item in enumerate(image_candidates)
            ]
            fused = fuse_detections([*refined, *verified[index]])
            annotations = tuple(
                Annotation(item.class_name, item.box, item.confidence, AnnotationSource.FUSED)
                for item in fused
            )
            results.append(
                InferenceResult(
                    image, annotations, tuple(item for item in fused if item.disagreement)
                )
            )
            if progress:
                progress(index + 1, len(images))
        LOGGER.info("inference completed for %d images", len(results))
        return results
