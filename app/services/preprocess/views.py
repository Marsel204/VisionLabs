"""Layered detector passes: transformed copies of an image whose detections are merged back.

Measured on held-out traffic data at equal false-positive counts, a horizontally flipped pass
finds somewhat more real objects than lowering the confidence threshold does (about +90 true
detections in one 150-image sample, about +25 in another). Photometric views (sharpen,
contrast, rain) add recall about as cheaply as a lower threshold would, so they are opt-in.
See ``DEFAULT_VIEWS`` in the audit detector for the audit default.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import ImageOps

from app.services.preprocess.enhance import PRESETS, enhance

VIEWS = frozenset({"original", "flip", *(name for name in PRESETS if name != "none")})
ORIGINAL_ONLY: tuple[str, ...] = ("original",)

# (label, confidence, (x1, y1, x2, y2) in pixels)
ViewBox = tuple[str, float, tuple[float, float, float, float]]


def validate_views(views: Sequence[str]) -> tuple[str, ...]:
    """Return ``views`` as a tuple, rejecting empty or unknown names."""
    unknown = [view for view in views if view not in VIEWS]
    if unknown or not views:
        raise ValueError(f"unknown detection views: {unknown or 'none given'}")
    return tuple(views)


def apply_view(image: Any, view: str) -> Any:
    """Return the PIL image as seen in ``view``."""
    if view == "original":
        return image
    if view == "flip":
        return ImageOps.mirror(image)
    return enhance(image, PRESETS[view])


def restore_box(
    box: Sequence[float], view: str, image_width: int
) -> tuple[float, float, float, float]:
    """Map a pixel box detected in ``view`` back to the original image's coordinates."""
    x1, y1, x2, y2 = (float(value) for value in box)
    if view == "flip":
        return image_width - x2, y1, image_width - x1, y2
    return x1, y1, x2, y2


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - inter
    )
    return inter / union if union else 0.0


def merge_view_boxes(
    base: Sequence[ViewBox], extra: Sequence[ViewBox], iou_threshold: float = 0.6
) -> list[ViewBox]:
    """Keep every base box and add extra-view boxes that do not repeat one of the same label.

    Base boxes are never dropped or altered, so a single-view result is unchanged by layering.
    """
    merged = list(base)
    for label, score, box in sorted(extra, key=lambda item: -item[1]):
        if any(label == other and _iou(box, other_box) >= iou_threshold
               for other, _, other_box in merged):
            continue
        merged.append((label, score, box))
    return merged
