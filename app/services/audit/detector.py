"""High-recall reference detection used as evidence by the annotation audit."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from app.services.annotation.class_names import normalize_class_name
from app.services.annotation.domain import BoundingBox
from app.services.audit.models import Prediction
from app.services.preprocess import apply_view, restore_box, validate_views

_EDGE_MARGIN = 2.0

Tile = tuple[int, int, int, int]

DEFAULT_VIEWS: tuple[str, ...] = ("original", "flip")


def tile_grid(width: int, height: int, tile_size: int, overlap: float) -> list[Tile]:
    """Return overlapping ``(left, top, right, bottom)`` tiles; empty if one tile covers it."""
    if width <= tile_size and height <= tile_size:
        return []
    stride = max(1, int(tile_size * (1.0 - overlap)))
    return [
        (left, top, min(width, left + tile_size), min(height, top + tile_size))
        for top in _starts(height, tile_size, stride)
        for left in _starts(width, tile_size, stride)
    ]


def _starts(length: int, tile_size: int, stride: int) -> list[int]:
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final = max(0, length - tile_size)
    if starts[-1] != final:
        starts.append(final)
    return starts


def detect_reference(
    model: Callable[..., Any],
    image: Any,
    *,
    confidence: float = 0.08,
    image_size: int = 1280,
    tile_size: int = 640,
    tile_overlap: float = 0.25,
    nms_iou: float = 0.60,
    allowed_classes: Iterable[str] | None = None,
    views: Sequence[str] = DEFAULT_VIEWS,
) -> list[Prediction]:
    """Run ``model`` on the whole image plus overlapping tiles, in every view, and merge.

    Each view is a transformed copy of the same window (see :data:`DEFAULT_VIEWS`); detections
    are mapped back to the original coordinates so objects one view misses can be recovered
    from another.

    Tiles recover small, distant objects. Boxes cut by an interior tile edge are discarded
    because the neighbouring tile or the full-image pass sees them whole.
    """
    views = validate_views(views)
    width, height = image.size
    wanted = {name.casefold() for name in allowed_classes} if allowed_classes else None
    full: Tile = (0, 0, width, height)
    windows = [full, *tile_grid(width, height, tile_size, tile_overlap)]
    predictions: list[Prediction] = []
    for window in windows:
        left, top, right, bottom = window
        is_tile = window != full
        crop = image.crop(window) if is_tile else image
        for view in views:
            result = model(
                apply_view(crop, view), conf=confidence, imgsz=image_size, iou=0.9, verbose=False
            )[0]
            names = getattr(model, "names", None) or getattr(result, "names", {})
            for detection in result.boxes:
                name = normalize_class_name(str(names[int(detection.cls[0])]))
                if wanted is not None and name.casefold() not in wanted:
                    continue
                x1, y1, x2, y2 = restore_box(
                    detection.xyxy[0].tolist(), view, crop.size[0]
                )
                if is_tile and _cut_by_interior_edge(
                    (x1, y1, x2, y2), window, (width, height)
                ):
                    continue
                box = _normalized(x1 + left, y1 + top, x2 + left, y2 + top, width, height)
                if box is not None:
                    predictions.append(Prediction(name, box, float(detection.conf[0])))
    return merge_predictions(predictions, nms_iou)


def _cut_by_interior_edge(
    box: tuple[float, float, float, float], window: Tile, image_size: tuple[int, int]
) -> bool:
    x1, y1, x2, y2 = box
    left, top, right, bottom = window
    width, height = image_size
    return (
        (x1 <= _EDGE_MARGIN and left > 0)
        or (y1 <= _EDGE_MARGIN and top > 0)
        or (x2 >= right - left - _EDGE_MARGIN and right < width)
        or (y2 >= bottom - top - _EDGE_MARGIN and bottom < height)
    )


def _normalized(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> BoundingBox | None:
    left, right = max(0.0, x1 / width), min(1.0, x2 / width)
    top, bottom = max(0.0, y1 / height), min(1.0, y2 / height)
    if left >= right or top >= bottom:
        return None
    return BoundingBox(left, top, right, bottom)


def merge_predictions(predictions: Sequence[Prediction], iou_threshold: float) -> list[Prediction]:
    """Class-wise NMS that keeps overlapping boxes unless they are near-identical."""
    kept: list[Prediction] = []
    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        if any(
            item.class_name == prediction.class_name
            and _iou(item.box, prediction.box) >= iou_threshold
            for item in kept
        ):
            continue
        kept.append(prediction)
    return kept


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    left, top = max(first.left, second.left), max(first.top, second.top)
    right, bottom = min(first.right, second.right), min(first.bottom, second.bottom)
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.area + second.area - inter
    return inter / union if union else 0.0
