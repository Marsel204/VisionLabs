"""Unit tests for Florence-2 VLM helper and crop verification utilities in src/vlm_helper.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from app.services.annotation.domain import BoundingBox
from src.vlm_helper import (
    Florence2VLM,
    crop_image,
    generate_annotations,
    load_image,
    map_od_label_to_class,
    match_caption_to_class,
    verify_crop_class,
)


# ==============================================================================
# Image Loading & Cropping Tests
# ==============================================================================


def test_load_image_from_pil() -> None:
    """Test loading from an existing PIL image."""
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 255))
    loaded = load_image(img)
    assert isinstance(loaded, Image.Image)
    assert loaded.mode == "RGB"
    assert loaded.size == (100, 100)


def test_load_image_from_file(tmp_path: Path) -> None:
    """Test loading from a disk file path."""
    file_path = tmp_path / "sample.jpg"
    Image.new("RGB", (64, 48), color="blue").save(file_path)

    loaded = load_image(file_path)
    assert isinstance(loaded, Image.Image)
    assert loaded.size == (64, 48)
    assert loaded.mode == "RGB"

    # Test string path as well
    loaded_str = load_image(str(file_path))
    assert loaded_str.size == (64, 48)


def test_load_image_file_not_found(tmp_path: Path) -> None:
    """Test that nonexistent image files raise FileNotFoundError."""
    non_existent = tmp_path / "does_not_exist.jpg"
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        load_image(non_existent)


def test_load_image_from_numpy_bgr() -> None:
    """Test loading from a NumPy array representing an OpenCV BGR image."""
    # 50x50 BGR image (all Blue: (255, 0, 0) in BGR -> (0, 0, 255) in RGB)
    bgr_array = np.zeros((50, 50, 3), dtype=np.uint8)
    bgr_array[:, :] = [255, 0, 0]

    loaded = load_image(bgr_array)
    assert isinstance(loaded, Image.Image)
    assert loaded.size == (50, 50)
    assert loaded.mode == "RGB"
    # Pixel (0,0) should be Red in RGB: (0, 0, 255)
    r, g, b = loaded.getpixel((0, 0))  # type: ignore[misc]
    assert (r, g, b) == (0, 0, 255)


def test_load_image_from_numpy_grayscale() -> None:
    """Test loading from a 2D grayscale NumPy array."""
    gray_array = np.full((30, 40), fill_value=128, dtype=np.uint8)
    loaded = load_image(gray_array)
    assert loaded.size == (40, 30)
    assert loaded.mode == "RGB"


def test_load_image_invalid_type() -> None:
    """Test that invalid types raise TypeError."""
    with pytest.raises(TypeError, match="Unsupported image source type"):
        load_image(12345)  # type: ignore[arg-type]


def test_crop_image_normalized() -> None:
    """Test cropping with normalized coordinates [0, 1]."""
    img = Image.new("RGB", (100, 200), color="white")
    # Normalized box (left=0.1, top=0.2, right=0.6, bottom=0.8)
    # in pixels: (10, 40, 60, 160) -> width 50, height 120
    crop = crop_image(img, (0.1, 0.2, 0.6, 0.8), normalized=True)
    assert crop.size == (50, 120)


def test_crop_image_pixel_coordinates() -> None:
    """Test cropping with absolute pixel coordinates."""
    img = Image.new("RGB", (200, 200), color="white")
    crop = crop_image(img, (20, 30, 80, 110), normalized=False)
    assert crop.size == (60, 80)


def test_crop_image_with_bounding_box_entity() -> None:
    """Test cropping with VisionLab BoundingBox dataclass."""
    img = Image.new("RGB", (100, 100), color="white")
    box = BoundingBox(0.2, 0.2, 0.8, 0.8)
    crop = crop_image(img, box, normalized=True)
    assert crop.size == (60, 60)


def test_crop_image_degenerate_bounds() -> None:
    """Test fallback when degenerate crop bounds are given."""
    img = Image.new("RGB", (50, 50), color="white")
    # Inverted box: right < left
    crop = crop_image(img, (0.8, 0.8, 0.2, 0.2), normalized=True)
    assert crop.size == (50, 50)


# ==============================================================================
# Caption Class Matching Tests
# ==============================================================================


@pytest.mark.parametrize(
    ("caption", "target_class", "expected"),
    [
        # Motorcycle positive tests
        ("a motorcycle parked by the roadside", "motorcycle", True),
        ("two motorbikes waiting at a red traffic light", "motorcycle", True),
        ("a rider on a black scooter", "motorcycle", True),
        ("vintage vespa in a garage", "motorcycle", True),
        ("a moped on the street", "motorcycle", True),
        ("a red bike leaning against a wall", "motorcycle", True),
        # Car positive tests
        ("a red car driving on the highway", "car", True),
        ("three cars in a parking lot", "car", True),
        ("a black automobile stopped at intersection", "car", True),
        ("silver sedan with headlights on", "car", True),
        ("a white suv on an open road", "car", True),
        ("a yellow taxi cab in new york", "car", True),
        ("a delivery van parked at the curb", "car", True),
        # Bus positive tests
        ("a double-decker bus in london", "bus", True),
        ("a city transit bus at the station", "bus", True),
        ("a yellow school bus", "bus", True),
        ("a blue minibus carrying passengers", "bus", True),
        # Truck positive tests
        ("a heavy freight truck on the bridge", "truck", True),
        ("a large lorry carrying cargo", "truck", True),
        ("a blue pickup truck on dirt road", "truck", True),
        ("a semi-truck driving on the interstate", "truck", True),
        # Person positive tests
        ("a pedestrian crossing the crosswalk", "person", True),
        ("a woman standing with an umbrella", "person", True),
        # Negative / Substring collision tests
        ("a red carpet in the living room", "car", False),  # 'carpet' != 'car'
        ("a business executive in meeting", "bus", False),  # 'business' != 'bus'
        ("a cardboard box on the table", "car", False),  # 'cardboard' != 'car'
        ("a tree in a green field", "motorcycle", False),
        ("an empty street with buildings", "car", False),
        ("", "car", False),
        ("a car driving", "", False),
    ],
)
def test_match_caption_to_class(caption: str, target_class: str, expected: bool) -> None:
    assert match_caption_to_class(caption, target_class) is expected


# ==============================================================================
# Florence-2 Execution & Crop Verification Tests
# ==============================================================================


def test_florence2_vlm_run_task_and_caption() -> None:
    """Test Florence2VLM task execution with injected mock model and processor."""
    mock_processor = MagicMock()
    mock_model = MagicMock()

    # Mock processor output tensors
    mock_processor.return_value = {
        "input_ids": MagicMock(),
        "pixel_values": MagicMock(),
    }
    # Mock model generation tokens
    mock_model.generate.return_value = MagicMock()
    # Mock batch decode
    mock_processor.batch_decode.return_value = ["<CAPTION> a blue motorcycle parked on the sidewalk"]
    # Mock post-process generation
    mock_processor.post_process_generation.return_value = {
        "<CAPTION>": "a blue motorcycle parked on the sidewalk"
    }

    vlm = Florence2VLM(
        model=mock_model,
        processor=mock_processor,
        device="cpu",
    )

    crop = Image.new("RGB", (64, 64), color="blue")
    result = vlm.run_task(crop, task_token="<CAPTION>")
    assert result == {"<CAPTION>": "a blue motorcycle parked on the sidewalk"}

    caption = vlm.generate_caption(crop)
    assert caption == "a blue motorcycle parked on the sidewalk"


def test_verify_crop_class_matches() -> None:
    """Test verify_crop_class returns True when caption matches target class."""
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_processor.return_value = {
        "input_ids": MagicMock(),
        "pixel_values": MagicMock(),
    }
    mock_model.generate.return_value = MagicMock()
    mock_processor.batch_decode.return_value = ["<CAPTION> a red car stopped at the traffic light"]
    mock_processor.post_process_generation.return_value = {
        "<CAPTION>": "a red car stopped at the traffic light"
    }

    vlm = Florence2VLM(
        model=mock_model,
        processor=mock_processor,
        device="cpu",
    )

    crop = Image.new("RGB", (64, 64), color="red")

    # Match target "car"
    assert verify_crop_class(crop, "car", vlm=vlm) is True

    # Do not match target "motorcycle"
    assert verify_crop_class(crop, "motorcycle", vlm=vlm) is False


def test_verify_crop_class_with_raw_string_fallback() -> None:
    """Test verify_crop_class when processor returns raw text without dictionary formatting."""
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_processor.return_value = {
        "input_ids": MagicMock(),
        "pixel_values": MagicMock(),
    }
    mock_model.generate.return_value = MagicMock()
    mock_processor.batch_decode.return_value = ["<CAPTION> a white transit bus at the terminal"]
    # Post process generation raises an exception or returns string
    mock_processor.post_process_generation.side_effect = RuntimeError("Post-process failed")

    vlm = Florence2VLM(
        model=mock_model,
        processor=mock_processor,
        device="cpu",
    )

    crop = Image.new("RGB", (64, 64), color="white")
    assert verify_crop_class(crop, "bus", vlm=vlm) is True
    assert verify_crop_class(crop, "truck", vlm=vlm) is False


# ==============================================================================
# Object Detection & Label Mapping Tests
# ==============================================================================


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("motorcycle", "motorcycle"),
        ("motorbike", "motorcycle"),
        ("scooter", "motorcycle"),
        ("vespa", "motorcycle"),
        ("car", "car"),
        ("sedan", "car"),
        ("suv", "car"),
        ("taxi", "car"),
        ("van", "car"),
        ("bus", "bus"),
        ("double-decker", "bus"),
        ("minibus", "bus"),
        ("truck", "truck"),
        ("lorry", "truck"),
        ("pickup", "truck"),
        ("semi-truck", "truck"),
        ("person", "person"),
        ("pedestrian", "person"),
        # Unmapped labels
        ("bicycle", None),
        ("traffic light", None),
        ("tree", None),
        ("", None),
    ],
)
def test_map_od_label_to_class(label: str, expected: str | None) -> None:
    assert map_od_label_to_class(label) == expected


def test_detect_objects_with_mock() -> None:
    """Test Florence2VLM.detect_objects() with a mocked <OD> response."""
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_processor.return_value = {
        "input_ids": MagicMock(),
        "pixel_values": MagicMock(),
    }
    mock_model.generate.return_value = MagicMock()
    mock_processor.batch_decode.return_value = ["<OD> car motorcycle"]
    mock_processor.post_process_generation.return_value = {
        "<OD>": {
            "bboxes": [[10.0, 20.0, 100.0, 150.0], [200.0, 50.0, 350.0, 200.0]],
            "labels": ["car", "motorcycle"],
        }
    }

    vlm = Florence2VLM(model=mock_model, processor=mock_processor, device="cpu")

    crop = Image.new("RGB", (640, 480), color="white")
    detections = vlm.detect_objects(crop)

    assert len(detections) == 2
    assert detections[0]["label"] == "car"
    assert detections[0]["box"] == [10.0, 20.0, 100.0, 150.0]
    assert detections[1]["label"] == "motorcycle"
    assert detections[1]["box"] == [200.0, 50.0, 350.0, 200.0]


def test_detect_objects_empty_result() -> None:
    """Test Florence2VLM.detect_objects() when no objects are detected."""
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_processor.return_value = {
        "input_ids": MagicMock(),
        "pixel_values": MagicMock(),
    }
    mock_model.generate.return_value = MagicMock()
    mock_processor.batch_decode.return_value = ["<OD>"]
    mock_processor.post_process_generation.return_value = {
        "<OD>": {"bboxes": [], "labels": []}
    }

    vlm = Florence2VLM(model=mock_model, processor=mock_processor, device="cpu")

    crop = Image.new("RGB", (100, 100), color="white")
    detections = vlm.detect_objects(crop)
    assert detections == []


def test_generate_annotations_end_to_end() -> None:
    """Test generate_annotations() returns correctly mapped and normalized results."""
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_processor.return_value = {
        "input_ids": MagicMock(),
        "pixel_values": MagicMock(),
    }
    mock_model.generate.return_value = MagicMock()
    mock_processor.batch_decode.return_value = ["<OD> car motorcycle tree"]
    mock_processor.post_process_generation.return_value = {
        "<OD>": {
            "bboxes": [
                [64.0, 48.0, 320.0, 240.0],   # car (valid)
                [100.0, 100.0, 400.0, 300.0],  # motorcycle (valid)
                [50.0, 50.0, 100.0, 100.0],    # tree (unmapped, should be skipped)
            ],
            "labels": ["sedan", "motorbike", "tree"],
        }
    }

    vlm = Florence2VLM(model=mock_model, processor=mock_processor, device="cpu")

    results = generate_annotations(
        image=Image.new("RGB", (640, 480), color="white"),
        image_width=640,
        image_height=480,
        vlm=vlm,
    )

    assert len(results) == 2
    class_name_0, bbox_0 = results[0]
    class_name_1, bbox_1 = results[1]

    assert class_name_0 == "car"
    assert class_name_1 == "motorcycle"

    # Verify normalization: 64/640 = 0.1, 48/480 = 0.1, 320/640 = 0.5, 240/480 = 0.5
    assert abs(bbox_0.left - 0.1) < 1e-6
    assert abs(bbox_0.top - 0.1) < 1e-6
    assert abs(bbox_0.right - 0.5) < 1e-6
    assert abs(bbox_0.bottom - 0.5) < 1e-6


def test_generate_annotations_filters_by_enabled_classes() -> None:
    """Test generate_annotations() respects enabled_classes filter."""
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_processor.return_value = {
        "input_ids": MagicMock(),
        "pixel_values": MagicMock(),
    }
    mock_model.generate.return_value = MagicMock()
    mock_processor.batch_decode.return_value = ["<OD> car motorcycle"]
    mock_processor.post_process_generation.return_value = {
        "<OD>": {
            "bboxes": [[10.0, 10.0, 100.0, 100.0], [200.0, 200.0, 400.0, 400.0]],
            "labels": ["car", "motorcycle"],
        }
    }

    vlm = Florence2VLM(model=mock_model, processor=mock_processor, device="cpu")

    results = generate_annotations(
        image=Image.new("RGB", (640, 480), color="white"),
        image_width=640,
        image_height=480,
        vlm=vlm,
        enabled_classes={"motorcycle"},  # Only motorcycle enabled
    )

    assert len(results) == 1
    assert results[0][0] == "motorcycle"
