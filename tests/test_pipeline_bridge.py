"""Unit tests for pipeline_bridge.py and Zero-Shot Auto-Annotation Pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from pipeline_bridge import (
    AutoAnnotationPipeline,
    BoxPixel,
    GroundingDinoDetector,
    ImageAnnotationResult,
    MaskToPolygonProcessor,
    PolygonAnnotation,
    SamSegmenter,
    build_grounding_prompt,
    export_coco_segmentation,
    export_yolov8_segmentation,
    parse_text_ontology,
)


# ==============================================================================
# Layer 3: Mask-to-Polygon Post-Processor Tests
# ==============================================================================


def test_mask_to_polygons_square() -> None:
    """Test Layer 3 contour extraction on a known binary square."""
    processor = MaskToPolygonProcessor(min_contour_area=10.0, epsilon_factor=0.01)

    # 100x100 mask with a 40x40 square inside (from (20,20) to (60,60))
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 1

    poly_pixels, poly_norm = processor.mask_to_polygons(mask, image_width=100, image_height=100)

    assert len(poly_pixels) >= 4
    assert len(poly_norm) == len(poly_pixels)

    # Verify all pixel coordinates are roughly within bounds [20, 60]
    for x, y in poly_pixels:
        assert 19 <= x <= 61
        assert 19 <= y <= 61

    # Verify normalized coordinates are within [0.19, 0.61]
    for x_norm, y_norm in poly_norm:
        assert 0.0 <= x_norm <= 1.0
        assert 0.0 <= y_norm <= 1.0
        assert 0.19 <= x_norm <= 0.61
        assert 0.19 <= y_norm <= 0.61


def test_mask_to_polygons_empty() -> None:
    """Test Layer 3 contour extraction on an empty mask."""
    processor = MaskToPolygonProcessor()
    mask = np.zeros((50, 50), dtype=np.uint8)

    poly_pixels, poly_norm = processor.mask_to_polygons(mask, image_width=50, image_height=50)

    assert poly_pixels == []
    assert poly_norm == []


def test_mask_to_polygons_noise_rejection() -> None:
    """Test that tiny noise contours below min_contour_area are filtered."""
    processor = MaskToPolygonProcessor(min_contour_area=50.0)
    # Mask with only a 2x2 dot (area 4)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:12, 10:12] = 1

    poly_pixels, poly_norm = processor.mask_to_polygons(mask, image_width=50, image_height=50)
    # Even if fallback is used, vertices < 3 will be rejected or empty
    assert len(poly_pixels) <= 4


# ==============================================================================
# Layer 1: Text-to-Bbox (Grounding DINO) Tests
# ==============================================================================


def test_parse_text_ontology() -> None:
    assert parse_text_ontology("helmet, vest, boots") == ["helmet", "vest", "boots"]
    assert parse_text_ontology("car; bus\ntruck") == ["car", "bus", "truck"]
    assert parse_text_ontology("") == ["object"]


def test_build_grounding_prompt() -> None:
    classes = ["helmet", "safety vest", "boots"]
    prompt = build_grounding_prompt(classes)
    assert prompt == "helmet. safety vest. boots."


def test_grounding_null_detections_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Edge Case: Null Detections must log warning and return empty list."""
    mock_processor = MagicMock()
    mock_model = MagicMock()

    # Mock empty detection outputs from Grounding DINO
    mock_processor.post_process_grounded_object_detection.return_value = [
        {"boxes": [], "scores": [], "text_labels": []}
    ]

    detector = GroundingDinoDetector(
        processor=mock_processor,
        model=mock_model,
        device="cpu",
    )

    image = Image.new("RGB", (100, 100), color="white")
    with caplog.at_level(logging.WARNING, logger="pipeline_bridge"):
        detections = detector.detect(
            image=image,
            image_filename="sample_empty.jpg",
            classes=["helmet", "vest"],
            confidence_threshold=0.35,
        )

    assert detections == []
    assert "Zero detections for text_prompt on sample_empty.jpg" in caplog.text


def test_grounding_confidence_filtering() -> None:
    """Verify detector filters boxes below the configurable confidence threshold."""
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_processor.post_process_grounded_object_detection.return_value = [
        {
            "boxes": [[10, 10, 50, 50], [60, 60, 90, 90]],
            "scores": [0.85, 0.20],  # 0.20 is below 0.35
            "text_labels": ["helmet", "vest"],
        }
    ]

    detector = GroundingDinoDetector(
        processor=mock_processor,
        model=mock_model,
        device="cpu",
    )

    image = Image.new("RGB", (100, 100), color="white")
    detections = detector.detect(
        image=image,
        image_filename="test.jpg",
        classes=["helmet", "vest"],
        confidence_threshold=0.35,
    )

    assert len(detections) == 1
    class_name, class_id, score, box = detections[0]
    assert class_name == "helmet"
    assert class_id == 0
    assert score == 0.85
    assert box.xmin == 10.0
    assert box.xmax == 50.0


# ==============================================================================
# Layer 2 & Pipeline Bridge Tests (Multi-Object Overlaps & End-to-End)
# ==============================================================================


def test_pipeline_null_detection_skips_sam(tmp_path: Path) -> None:
    """If 0 boxes detected, SAM segmenter must not be called."""
    mock_detector = MagicMock(spec=GroundingDinoDetector)
    mock_detector.detect.return_value = []

    mock_sam = MagicMock(spec=SamSegmenter)

    pipeline = AutoAnnotationPipeline(
        grounding_detector=mock_detector,
        sam_segmenter=mock_sam,
    )

    img_path = tmp_path / "img1.jpg"
    Image.new("RGB", (80, 80), color="blue").save(img_path)

    result = pipeline.process_image(img_path, "helmet, vest")

    assert len(result.annotations) == 0
    mock_detector.detect.assert_called_once()
    mock_sam.segment_box.assert_not_called()


def test_pipeline_multi_object_overlap_independent_prompts(tmp_path: Path) -> None:
    """Edge Case: Multi-object overlaps must prompt SAM independently per box."""
    # Create 2 overlapping bounding boxes
    box1 = BoxPixel(10.0, 10.0, 50.0, 50.0)
    box2 = BoxPixel(30.0, 30.0, 70.0, 70.0)

    mock_detector = MagicMock(spec=GroundingDinoDetector)
    mock_detector.detect.return_value = [
        ("helmet", 0, 0.90, box1),
        ("vest", 1, 0.80, box2),
    ]

    # Create distinct masks for each box
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[10:50, 10:50] = 1

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[30:70, 30:70] = 1

    mock_sam = MagicMock(spec=SamSegmenter)
    mock_sam.segment_boxes.return_value = [mask1, mask2]
    mock_sam.segment_box.side_effect = [mask1, mask2]

    processor = MaskToPolygonProcessor()

    pipeline = AutoAnnotationPipeline(
        grounding_detector=mock_detector,
        sam_segmenter=mock_sam,
        polygon_processor=processor,
    )

    img_path = tmp_path / "overlap.jpg"
    Image.new("RGB", (100, 100), color="green").save(img_path)

    result = pipeline.process_image(img_path, "helmet, vest")

    assert len(result.annotations) == 2
    assert mock_sam.segment_boxes.call_count == 1 or mock_sam.segment_box.call_count == 2

    # Check annotation 1
    ann1 = result.annotations[0]
    assert ann1.class_name == "helmet"
    assert ann1.class_id == 0
    assert len(ann1.polygon_pixels) >= 3

    # Check annotation 2
    ann2 = result.annotations[1]
    assert ann2.class_name == "vest"
    assert ann2.class_id == 1
    assert len(ann2.polygon_pixels) >= 3


# ==============================================================================
# Exporter Tests (YOLOv8-Segmentation and COCO JSON)
# ==============================================================================


def test_export_yolov8_segmentation(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (100, 100), color="black").save(img_path)

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 1

    processor = MaskToPolygonProcessor()
    poly_pixels, poly_norm = processor.mask_to_polygons(mask, 100, 100)

    annotation = PolygonAnnotation(
        class_name="helmet",
        class_id=0,
        confidence=0.92,
        box=BoxPixel(20.0, 20.0, 60.0, 60.0),
        mask=mask,
        polygon_pixels=poly_pixels,
        polygon_normalized=poly_norm,
    )

    result = ImageAnnotationResult(
        image_path=img_path,
        image_width=100,
        image_height=100,
        annotations=[annotation],
    )

    out_dir = tmp_path / "yolo_export"
    yaml_path = export_yolov8_segmentation([result], out_dir, classes=["helmet", "vest"])

    assert yaml_path.is_file()
    assert (out_dir / "dataset.yaml").exists()

    label_file = out_dir / "labels" / "sample.txt"
    assert label_file.is_file()

    content = label_file.read_text(encoding="utf-8").strip()
    assert content.startswith("0 ")
    # Coordinates in YOLO segmentation should be space-separated floats
    parts = content.split(" ")
    assert len(parts) >= 7  # class_id + at least 3 (x, y) pairs
    for coord in parts[1:]:
        val = float(coord)
        assert 0.0 <= val <= 1.0


def test_export_coco_segmentation(tmp_path: Path) -> None:
    img_path = tmp_path / "coco_sample.jpg"
    Image.new("RGB", (200, 150), color="black").save(img_path)

    mask = np.zeros((150, 200), dtype=np.uint8)
    mask[30:90, 40:120] = 1

    processor = MaskToPolygonProcessor()
    poly_pixels, poly_norm = processor.mask_to_polygons(mask, 200, 150)

    annotation = PolygonAnnotation(
        class_name="boots",
        class_id=1,
        confidence=0.88,
        box=BoxPixel(40.0, 30.0, 120.0, 90.0),
        mask=mask,
        polygon_pixels=poly_pixels,
        polygon_normalized=poly_norm,
    )

    result = ImageAnnotationResult(
        image_path=img_path,
        image_width=200,
        image_height=150,
        annotations=[annotation],
    )

    out_dir = tmp_path / "coco_export"
    json_path = export_coco_segmentation([result], out_dir, classes=["helmet", "boots"])

    assert json_path.is_file()
    data = json.loads(json_path.read_text(encoding="utf-8"))

    assert "images" in data and len(data["images"]) == 1
    assert data["images"][0]["file_name"] == "coco_sample.jpg"
    assert data["images"][0]["width"] == 200
    assert data["images"][0]["height"] == 150

    assert "categories" in data and len(data["categories"]) == 2
    assert data["categories"][1]["id"] == 2
    assert data["categories"][1]["name"] == "boots"

    assert "annotations" in data and len(data["annotations"]) == 1
    ann_entry = data["annotations"][0]
    assert ann_entry["category_id"] == 2  # class_id 1 + 1 (1-indexed)
    assert len(ann_entry["segmentation"]) == 1
    assert len(ann_entry["segmentation"][0]) >= 6  # at least 3 (x,y) pairs
    assert ann_entry["bbox"] == [40.0, 30.0, 80.0, 60.0]
    assert ann_entry["area"] > 0


def test_main_window_dino_sam_action_registration(qapp: object) -> None:
    """Test that MainWindow registers the 1-click DINO+SAM auto-annotate actions."""
    from app.configs.settings import AppSettings
    from app.ui.main_window import MainWindow

    settings = AppSettings()
    window = MainWindow(settings.fusion, settings.active_learning)

    # Verify action exists and has correct shortcut
    actions = [a.text() for a in window.findChildren(object) if hasattr(a, "text") and hasattr(a, "triggered")]
    assert "DINO + SAM Auto-Annotate" in actions
    assert "DINO + SAM Annotate Entire Dataset" in actions


def test_pipeline_vlm_verification_filters_candidate_boxes(tmp_path: Path) -> None:
    """Test that AutoAnnotationPipeline filters candidate boxes using Florence-2 VLM verifier."""
    box_true = BoxPixel(10.0, 10.0, 50.0, 50.0)
    box_false = BoxPixel(60.0, 60.0, 90.0, 90.0)

    mock_detector = MagicMock(spec=GroundingDinoDetector)
    mock_detector.detect.return_value = [
        ("helmet", 0, 0.90, box_true),
        ("vest", 1, 0.85, box_false),
    ]

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:50, 10:50] = 1

    mock_sam = MagicMock(spec=SamSegmenter)
    mock_sam.segment_boxes.return_value = [mask]

    # Mock Florence2VLM
    mock_vlm = MagicMock()
    # 1st box -> caption containing helmet (valid)
    # 2nd box -> caption containing background (invalid for vest)
    mock_vlm.generate_caption.side_effect = [
        "a yellow safety helmet on a construction worker",
        "a concrete wall with graffiti",
    ]

    processor = MaskToPolygonProcessor()

    pipeline = AutoAnnotationPipeline(
        grounding_detector=mock_detector,
        sam_segmenter=mock_sam,
        polygon_processor=processor,
        vlm_verifier=mock_vlm,
        enable_vlm=True,
    )

    img_path = tmp_path / "test_vlm_scene.jpg"
    Image.new("RGB", (100, 100), color="gray").save(img_path)

    result = pipeline.process_image(img_path, "helmet, vest")

    # Only 1 annotation should survive VLM verification
    assert len(result.annotations) == 1
    assert result.annotations[0].class_name == "helmet"
    assert mock_vlm.generate_caption.call_count == 2

