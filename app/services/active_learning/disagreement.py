"""Model disagreement, missing-detection, and duplicate feature calculations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.models.contracts import Detection
from app.services.fusion.iou import pairwise_iou


def disagreement_features(
    detections: Sequence[Detection], iou_threshold: float = 0.50
) -> tuple[float, int, int, float]:
    """Return disagreement ratio, missing count, duplicate count, and occlusion score."""
    if not detections:
        return 0.0, 0, 0, 0.0
    duplicate_count = _duplicate_count(detections, iou_threshold)
    sources = sorted({item.source for item in detections}, key=str)
    if len(sources) < 2:
        return 0.0, len(detections), duplicate_count, _occlusion(detections)
    grouped = [[item for item in detections if item.source == source] for source in sources]
    matched_pairs = 0
    disagreements = 0
    missing = 0
    for first, second in zip(grouped, grouped[1:], strict=False):
        matrix = pairwise_iou([item.box for item in first], [item.box for item in second])
        matched_first: set[int] = set()
        matched_second: set[int] = set()
        rows, columns = (matrix >= iou_threshold).nonzero()
        for first_index, second_index in zip(rows, columns, strict=True):
            if first_index in matched_first or second_index in matched_second:
                continue
            matched_first.add(int(first_index))
            matched_second.add(int(second_index))
            matched_pairs += 1
            if first[int(first_index)].class_name != second[int(second_index)].class_name:
                disagreements += 1
        missing += (len(first) - len(matched_first)) + (len(second) - len(matched_second))
    ratio = disagreements / matched_pairs if matched_pairs else 0.0
    return ratio, missing, duplicate_count, _occlusion(detections)


def _duplicate_count(detections: Sequence[Detection], threshold: float) -> int:
    count = 0
    for source in {item.source for item in detections}:
        source_items = [item for item in detections if item.source == source]
        n = len(source_items)
        if n < 2:
            continue
        boxes = [item.box for item in source_items]
        matrix = pairwise_iou(boxes, boxes)
        classes = [item.class_name for item in source_items]
        class_eq = np.equal.outer(classes, classes)
        triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        count += int(np.count_nonzero((matrix >= threshold) & class_eq & triu_mask))
    return count


def _occlusion(detections: Sequence[Detection]) -> float:
    if len(detections) < 2:
        return 0.0
    boxes = [item.box for item in detections]
    matrix = pairwise_iou(boxes, boxes)
    triu_vals = matrix[np.triu_indices(len(detections), k=1)]
    return float(triu_vals.max()) if triu_vals.size > 0 else 0.0
