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
    n = len(detections)
    if not n:
        return 0.0, 0, 0, 0.0

    boxes = [item.box for item in detections]
    full_matrix = pairwise_iou(boxes, boxes) if n > 1 else np.zeros((1, 1), dtype=float)

    duplicate_count = _duplicate_count_from_matrix(detections, full_matrix, iou_threshold)
    occlusion = _occlusion_from_matrix(full_matrix, n)

    sources = sorted({item.source for item in detections}, key=str)
    if len(sources) < 2:
        return 0.0, n, duplicate_count, occlusion

    source_indices: dict[object, list[int]] = {s: [] for s in sources}
    for idx, item in enumerate(detections):
        source_indices[item.source].append(idx)

    grouped_indices = [source_indices[s] for s in sources]
    matched_pairs = 0
    disagreements = 0
    missing = 0

    for first_idx_list, second_idx_list in zip(grouped_indices, grouped_indices[1:], strict=False):
        matrix = full_matrix[np.ix_(first_idx_list, second_idx_list)]
        matched_first: set[int] = set()
        matched_second: set[int] = set()
        rows, columns = (matrix >= iou_threshold).nonzero()
        for first_index, second_index in zip(rows, columns, strict=True):
            if first_index in matched_first or second_index in matched_second:
                continue
            matched_first.add(int(first_index))
            matched_second.add(int(second_index))
            matched_pairs += 1
            f_idx = first_idx_list[int(first_index)]
            s_idx = second_idx_list[int(second_index)]
            if detections[f_idx].class_name != detections[s_idx].class_name:
                disagreements += 1
        missing += (len(first_idx_list) - len(matched_first)) + (
            len(second_idx_list) - len(matched_second)
        )

    ratio = disagreements / matched_pairs if matched_pairs else 0.0
    return ratio, missing, duplicate_count, occlusion


def _duplicate_count_from_matrix(
    detections: Sequence[Detection], full_matrix: np.ndarray, threshold: float
) -> int:
    count = 0
    source_map: dict[object, list[int]] = {}
    for idx, item in enumerate(detections):
        source_map.setdefault(item.source, []).append(idx)

    for _source, indices in source_map.items():
        k = len(indices)
        if k < 2:
            continue
        sub_matrix = full_matrix[np.ix_(indices, indices)]
        classes = [detections[i].class_name for i in indices]
        class_eq = np.equal.outer(classes, classes)
        triu_mask = np.triu(np.ones((k, k), dtype=bool), k=1)
        count += int(np.count_nonzero((sub_matrix >= threshold) & class_eq & triu_mask))
    return count


def _occlusion_from_matrix(full_matrix: np.ndarray, n: int) -> float:
    if n < 2:
        return 0.0
    triu_vals = full_matrix[np.triu_indices(n, k=1)]
    return float(triu_vals.max()) if triu_vals.size > 0 else 0.0


def _duplicate_count(detections: Sequence[Detection], threshold: float) -> int:
    if len(detections) < 2:
        return 0
    boxes = [item.box for item in detections]
    matrix = pairwise_iou(boxes, boxes)
    return _duplicate_count_from_matrix(detections, matrix, threshold)


def _occlusion(detections: Sequence[Detection]) -> float:
    if len(detections) < 2:
        return 0.0
    boxes = [item.box for item in detections]
    matrix = pairwise_iou(boxes, boxes)
    return _occlusion_from_matrix(matrix, len(detections))
