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


def pairwise_iou(first: Sequence[BoundingBox], second: Sequence[BoundingBox]) -> np.ndarray:
    """Return an ``(len(first), len(second))`` IoU matrix using NumPy."""
    if not first or not second:
        return np.zeros((len(first), len(second)), dtype=float)
    left_a = np.asarray([box.left for box in first])[:, None]
    top_a = np.asarray([box.top for box in first])[:, None]
    right_a = np.asarray([box.right for box in first])[:, None]
    bottom_a = np.asarray([box.bottom for box in first])[:, None]
    left_b = np.asarray([box.left for box in second])[None, :]
    top_b = np.asarray([box.top for box in second])[None, :]
    right_b = np.asarray([box.right for box in second])[None, :]
    bottom_b = np.asarray([box.bottom for box in second])[None, :]
    intersection = np.maximum(0.0, np.minimum(right_a, right_b) - np.maximum(left_a, left_b))
    intersection *= np.maximum(0.0, np.minimum(bottom_a, bottom_b) - np.maximum(top_a, top_b))
    areas_a = np.asarray([box.area for box in first])[:, None]
    areas_b = np.asarray([box.area for box in second])[None, :]
    union = areas_a + areas_b - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
