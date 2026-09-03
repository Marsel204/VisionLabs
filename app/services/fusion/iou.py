"""Scalar and vectorized intersection-over-union helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.services.annotation.domain import BoundingBox


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    """Return IoU for two normalized boxes."""
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.area + second.area - intersection
    return intersection / union if union else 0.0


def _extract_box_arrays(
    boxes: Sequence[BoundingBox],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (left, top, right, bottom, area) 1D numpy arrays efficiently."""
    n = len(boxes)
    left = np.empty(n, dtype=np.float64)
    top = np.empty(n, dtype=np.float64)
    right = np.empty(n, dtype=np.float64)
    bottom = np.empty(n, dtype=np.float64)
    for i, b in enumerate(boxes):
        left[i] = b.left
        top[i] = b.top
        right[i] = b.right
        bottom[i] = b.bottom
    areas = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    return left, top, right, bottom, areas


def pairwise_iou_and_containment(
    first: Sequence[BoundingBox], second: Sequence[BoundingBox]
) -> tuple[np.ndarray, np.ndarray]:
    """Return both ``(IoU, containment)`` matrices in a single vectorized pass."""
    if not first or not second:
        zeros = np.zeros((len(first), len(second)), dtype=float)
        return zeros, zeros

    left_a, top_a, right_a, bottom_a, areas_a = _extract_box_arrays(first)
    if first is second:
        left_b, top_b, right_b, bottom_b, areas_b = left_a, top_a, right_a, bottom_a, areas_a
    else:
        left_b, top_b, right_b, bottom_b, areas_b = _extract_box_arrays(second)

    inter_w = np.maximum(
        0.0,
        np.minimum(right_a[:, None], right_b[None, :])
        - np.maximum(left_a[:, None], left_b[None, :]),
    )
    inter_h = np.maximum(
        0.0,
        np.minimum(bottom_a[:, None], bottom_b[None, :])
        - np.maximum(top_a[:, None], top_b[None, :]),
    )
    intersection = inter_w * inter_h

    union = areas_a[:, None] + areas_b[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

    min_areas = np.minimum(areas_a[:, None], areas_b[None, :])
    containment = np.divide(
        intersection, min_areas, out=np.zeros_like(intersection), where=min_areas > 0
    )

    return iou, containment


def pairwise_iou(first: Sequence[BoundingBox], second: Sequence[BoundingBox]) -> np.ndarray:
    """Return an ``(len(first), len(second))`` IoU matrix using NumPy."""
    iou, _ = pairwise_iou_and_containment(first, second)
    return iou
