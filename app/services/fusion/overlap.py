"""Removal of redundant overlapping annotations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.services.annotation.domain import Annotation
from app.services.fusion.iou import pairwise_iou


def remove_overlapping_annotations(
    annotations: Sequence[Annotation],
    iou_threshold: float = 0.50,
    containment_threshold: float = 0.80,
    same_class_only: bool = True,
) -> tuple[tuple[Annotation, ...], int]:
    """Keep the highest-confidence annotation from each redundant overlap group.

    Different classes are retained by default because an overlap may represent a real
    occlusion or a model conflict rather than a duplicate object.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if not 0.0 < containment_threshold <= 1.0:
        raise ValueError("containment_threshold must be in (0, 1]")
    ordered = sorted(
        annotations,
        key=lambda item: item.confidence if item.confidence is not None else 0.0,
        reverse=True,
    )
    if len(ordered) < 2:
        return tuple(ordered), 0
    matrix = pairwise_iou([item.box for item in ordered], [item.box for item in ordered])
    areas = np.asarray([item.box.area for item in ordered])
    area_sum = areas[:, None] + areas[None, :]
    intersections = np.divide(
        matrix * area_sum,
        1.0 + matrix,
        out=np.zeros_like(matrix),
        where=(1.0 + matrix) > 0,
    )
    containment = np.divide(
        intersections,
        np.minimum(areas[:, None], areas[None, :]),
        out=np.zeros_like(intersections),
        where=np.minimum(areas[:, None], areas[None, :]) > 0,
    )
    removed: set[int] = set()
    for index, annotation in enumerate(ordered):
        if index in removed:
            continue
        for other_index in range(index + 1, len(ordered)):
            overlaps = (
                matrix[index, other_index] >= iou_threshold
                or containment[index, other_index] >= containment_threshold
            )
            if other_index in removed or not overlaps:
                continue
            if same_class_only and annotation.class_name != ordered[other_index].class_name:
                continue
            removed.add(other_index)
    return tuple(item for index, item in enumerate(ordered) if index not in removed), len(removed)
