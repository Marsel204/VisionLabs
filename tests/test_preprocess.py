import numpy as np
import pytest
from PIL import Image

from app.services.preprocess import PRESETS, enhance, resolve_preset


def _blurry_edge() -> Image.Image:
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:, 32:] = 120
    import cv2

    return Image.fromarray(cv2.GaussianBlur(pixels, (0, 0), 1.5))


def _edge_strength(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=float)
    return float(np.abs(np.diff(gray, axis=1)).max())


def test_none_preset_returns_the_same_image() -> None:
    image = _blurry_edge()
    assert enhance(image, resolve_preset(None)) is image


def test_every_preset_keeps_size_and_mode() -> None:
    image = _blurry_edge()
    for options in PRESETS.values():
        result = enhance(image, options)
        assert result.size == image.size and result.mode == "RGB"


def test_sharpen_steepens_edges_without_touching_the_input() -> None:
    image = _blurry_edge()
    before = np.asarray(image).copy()
    sharpened = enhance(image, resolve_preset("sharpen"))
    assert _edge_strength(sharpened) > _edge_strength(image) * 1.3
    assert np.array_equal(np.asarray(image), before)


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_preset("cartoon")


def test_restore_box_mirrors_only_the_flip_view() -> None:
    from app.services.preprocess import restore_box

    assert restore_box([10, 20, 30, 60], "flip", 100) == (70, 20, 90, 60)
    assert restore_box([10, 20, 30, 60], "sharpen", 100) == (10, 20, 30, 60)


def test_merge_keeps_every_base_box_and_adds_only_unseen_extras() -> None:
    from app.services.preprocess import merge_view_boxes

    base = [("motorcycle", 0.9, (0, 0, 10, 10)), ("motorcycle", 0.5, (0, 0, 9, 10))]
    extra = [
        ("motorcycle", 0.8, (0, 0, 10, 11)),   # repeats a base box
        ("motorcycle", 0.4, (50, 50, 60, 60)),  # genuinely new
        ("car", 0.7, (0, 0, 10, 10)),           # same place, different class
    ]
    merged = merge_view_boxes(base, extra)
    assert merged[:2] == base
    assert [item[:2] for item in merged[2:]] == [("car", 0.7), ("motorcycle", 0.4)]


def test_validate_views_rejects_unknown_and_empty() -> None:
    from app.services.preprocess import validate_views

    assert validate_views(["original", "flip"]) == ("original", "flip")
    for bad in ([], ["sepia"]):
        with pytest.raises(ValueError):
            validate_views(bad)
