"""Map crop annotations back to an original image."""

from __future__ import annotations

from collections.abc import Sequence

from app.services.annotation.domain import Annotation, AnnotationDocument, BoundingBox
from app.services.crop_assisted.crop_models import CropRegion
from app.services.fusion.overlap import remove_overlapping_annotations


class CropMerger:
    """Merge local crop annotations into one original-image document."""

    def merge(
        self,
        original: AnnotationDocument,
        regions: Sequence[CropRegion],
        crop_documents: Sequence[AnnotationDocument],
        *,
        remove_duplicates: bool = True,
    ) -> AnnotationDocument:
        """Map boxes to original coordinates and remove crop-boundary duplicates."""
        if len(regions) != len(crop_documents):
            raise ValueError("regions and crop_documents must have equal lengths")
        mapped: list[Annotation] = []
        for region, document in zip(regions, crop_documents, strict=True):
            mapped.extend(
                self._to_original(item, original, region) for item in document.annotations
            )
        if remove_duplicates:
            kept, _ = remove_overlapping_annotations(
                mapped,
                iou_threshold=0.50,
                containment_threshold=0.80,
                same_class_only=True,
            )
            mapped = list(kept)
        return AnnotationDocument(
            original.image_path,
            original.image_width,
            original.image_height,
            tuple(mapped),
        )

    @staticmethod
    def _to_original(
        annotation: Annotation,
        original: AnnotationDocument,
        region: CropRegion,
    ) -> Annotation:
        left = (region.left + annotation.box.left * region.width) / original.image_width
        top = (region.top + annotation.box.top * region.height) / original.image_height
        right = (region.left + annotation.box.right * region.width) / original.image_width
        bottom = (region.top + annotation.box.bottom * region.height) / original.image_height
        return Annotation(
            class_name=annotation.class_name,
            box=BoundingBox(left, top, right, bottom),
            confidence=annotation.confidence,
            source=annotation.source,
        )
