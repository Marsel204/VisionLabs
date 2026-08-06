"""Generate overlapping image crops for high-resolution annotation."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from app.services.annotation.domain import Annotation, AnnotationDocument, BoundingBox
from app.services.crop_assisted.crop_models import CropRegion, CropSession

LOGGER = logging.getLogger(__name__)


class CropGenerator:
    """Create crop images and local annotation documents from one source image."""

    def generate(
        self,
        document: AnnotationDocument,
        destination: Path,
        *,
        tile_size: int = 640,
        overlap: float = 0.20,
    ) -> CropSession:
        """Generate overlapping crops with center-owned existing annotations."""
        if tile_size < 32:
            raise ValueError("tile_size must be at least 32")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")
        image = cv2.imread(str(document.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image: {document.image_path}")
        destination.mkdir(parents=True, exist_ok=True)
        height, width = image.shape[:2]
        if width != document.image_width or height != document.image_height:
            raise ValueError("image dimensions do not match annotation document")
        crop_width = tile_size
        crop_height = tile_size
        x_overlap = overlap
        y_overlap = overlap
        if width <= tile_size:
            crop_width = max(1, (width + 1) // 2)
            x_overlap = 0.0
        if height <= tile_size:
            crop_height = max(1, (height + 1) // 2)
            y_overlap = 0.0
        x_starts = self._starts(
            width,
            crop_width,
            max(1, int(crop_width * (1.0 - x_overlap))),
        )
        y_starts = self._starts(
            height,
            crop_height,
            max(1, int(crop_height * (1.0 - y_overlap))),
        )
        regions: list[CropRegion] = []
        documents: list[AnnotationDocument] = []
        index = 0
        for row, top in enumerate(y_starts):
            for column, left in enumerate(x_starts):
                right = min(width, left + crop_width)
                bottom = min(height, top + crop_height)
                core_left = (
                    0
                    if column == 0
                    else max(left, (left + x_starts[column - 1]) // 2)
                )
                core_right = (
                    width
                    if column == len(x_starts) - 1
                    else min(right, (left + x_starts[column + 1]) // 2)
                )
                core_top = (
                    0 if row == 0 else max(top, (top + y_starts[row - 1]) // 2)
                )
                core_bottom = (
                    height
                    if row == len(y_starts) - 1
                    else min(bottom, (top + y_starts[row + 1]) // 2)
                )
                crop_path = destination / f"crop_{index:04d}.jpg"
                crop = image[top:bottom, left:right]
                if not cv2.imwrite(str(crop_path), crop):
                    raise OSError(f"could not write crop: {crop_path}")
                region = CropRegion(
                    index,
                    left,
                    top,
                    right,
                    bottom,
                    core_left,
                    core_top,
                    core_right,
                    core_bottom,
                    crop_path,
                )
                local_annotations = tuple(
                    self._to_local(annotation, document, region)
                    for annotation in document.annotations
                    if region.owns_center(
                        (annotation.box.left + annotation.box.right) * document.image_width / 2,
                        (annotation.box.top + annotation.box.bottom) * document.image_height / 2,
                    )
                )
                regions.append(region)
                documents.append(
                    AnnotationDocument(
                        crop_path,
                        right - left,
                        bottom - top,
                        local_annotations,
                    )
                )
                index += 1
        LOGGER.info("generated %d crops for %s", len(regions), document.image_path)
        return CropSession(document, tuple(regions), tuple(documents), tile_size, overlap)

    @staticmethod
    def _starts(length: int, tile_size: int, step: int) -> list[int]:
        if length <= tile_size:
            return [0]
        starts = list(range(0, length - tile_size + 1, step))
        if starts[-1] != length - tile_size:
            starts.append(length - tile_size)
        return starts

    @staticmethod
    def _to_local(
        annotation: Annotation,
        document: AnnotationDocument,
        region: CropRegion,
    ) -> Annotation:
        left = max(annotation.box.left * document.image_width, region.left)
        top = max(annotation.box.top * document.image_height, region.top)
        right = min(annotation.box.right * document.image_width, region.right)
        bottom = min(annotation.box.bottom * document.image_height, region.bottom)
        return Annotation(
            class_name=annotation.class_name,
            box=BoundingBox(
                (left - region.left) / region.width,
                (top - region.top) / region.height,
                (right - region.left) / region.width,
                (bottom - region.top) / region.height,
            ),
            confidence=annotation.confidence,
            source=annotation.source,
        )
