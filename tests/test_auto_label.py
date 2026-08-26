"""Unit tests for the Auto Label service and engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from app.services.annotation.domain import AnnotationDocument, AnnotationSource, BoundingBox
from app.services.auto_label.engine import AutoLabelEngine, compute_box_iou
from app.services.auto_label.models import (
    DEFAULT_AUTO_LABEL_CLASSES,
    AutoLabelClass,
    AutoLabelConfig,
    AutoLabelDetection,
    AutoLabelPipelineMode,
    AutoLabelResult,
)
from pipeline_bridge import BoxPixel


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a temporary test image."""
    img_path = tmp_path / "test_sample.jpg"
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    img.save(img_path)
    return img_path


def test_auto_label_class_effective_prompt() -> None:
    """Test effective prompt returns prompt if set or falls back to name."""
    cls1 = AutoLabelClass(name="truck", prompt="semi truck, flatbed")
    assert cls1.effective_prompt == "semi truck, flatbed"

    cls2 = AutoLabelClass(name="motorcycle", prompt="   ")
    assert cls2.effective_prompt == "motorcycle"


def test_auto_label_pipeline_mode_properties() -> None:
    """Test pipeline mode display names and capabilities."""
    mode_mask = AutoLabelPipelineMode.DINO_SAM2_MASKS
    assert mode_mask.produces_masks is True
    assert mode_mask.uses_vlm is False
    assert mode_mask.uses_yolo is False
    assert mode_mask.uses_grounding is True
    assert mode_mask.badge_label == "Mask labels"
    assert "SAM 2" in mode_mask.display_name

    mode_box = AutoLabelPipelineMode.DINO_BOXES
    assert mode_box.produces_masks is False
    assert mode_box.badge_label == "Box labels"
    assert mode_box.uses_grounding is True

    mode_yolo_masks = AutoLabelPipelineMode.YOLO_SAM2_MASKS
    assert mode_yolo_masks.produces_masks is True
    assert mode_yolo_masks.uses_yolo is True
    assert mode_yolo_masks.uses_vlm is False
    assert mode_yolo_masks.badge_label == "Mask labels"
    assert "YOLO + SAM 2" in mode_yolo_masks.display_name

    mode_yolo_boxes = AutoLabelPipelineMode.YOLO_BOXES
    assert mode_yolo_boxes.produces_masks is False
    assert mode_yolo_boxes.uses_yolo is True
    assert mode_yolo_boxes.badge_label == "Box labels"
    assert mode_yolo_boxes.display_name == "YOLO (Bounding Boxes)"

    mode_vlm_masks = AutoLabelPipelineMode.VLM_SAM2_MASKS
    assert mode_vlm_masks.produces_masks is True
    assert mode_vlm_masks.uses_vlm is True
    assert mode_vlm_masks.uses_yolo is False

    mode_vlm_boxes = AutoLabelPipelineMode.VLM_BOXES
    assert mode_vlm_boxes.produces_masks is False
    assert mode_vlm_boxes.uses_vlm is True

    mode_ensemble_masks = AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS
    assert mode_ensemble_masks.produces_masks is True
    assert mode_ensemble_masks.is_ensemble is True
    assert mode_ensemble_masks.display_name == "Multi-Model Ensemble + SAM 2 (Masks)"

    mode_ensemble_boxes = AutoLabelPipelineMode.ENSEMBLE_FUSION_BOXES
    assert mode_ensemble_boxes.produces_masks is False
    assert mode_ensemble_boxes.is_ensemble is True
    assert mode_ensemble_boxes.badge_label == "Box labels"


def test_compute_box_iou() -> None:
    """Test IoU computation between normalized bounding boxes."""
    b1 = BoundingBox(0.0, 0.0, 0.5, 0.5)
    b2 = BoundingBox(0.0, 0.0, 0.5, 0.5)
    assert compute_box_iou(b1, b2) == pytest.approx(1.0)

    b3 = BoundingBox(0.6, 0.6, 1.0, 1.0)
    assert compute_box_iou(b1, b3) == 0.0

    b4 = BoundingBox(0.25, 0.0, 0.75, 0.5)
    # intersection: [0.25, 0.0, 0.5, 0.5] = 0.25 * 0.5 = 0.125
    # union: 0.25 + 0.25 - 0.125 = 0.375 -> 0.125 / 0.375 = 1/3
    assert compute_box_iou(b1, b4) == pytest.approx(1.0 / 3.0)


def test_build_prompt_mapping() -> None:
    """Test compound Grounding DINO prompt construction and token mapping."""
    classes = [
        AutoLabelClass(name="truck", prompt="commercial delivery truck, semi"),
        AutoLabelClass(name="car", prompt="sedan, suv, automobile"),
        AutoLabelClass(name="disabled_class", prompt="ignored", enabled=False),
    ]

    prompt, token_map = AutoLabelEngine.build_prompt_mapping(classes)
    assert "commercial delivery truck, semi" in prompt
    assert "sedan, suv, automobile" in prompt
    assert "ignored" not in prompt

    assert "truck" in token_map
    assert "car" in token_map
    assert "semi" in token_map
    assert "sedan" in token_map


def test_match_detected_label() -> None:
    """Test matching raw detector phrases back to canonical AutoLabelClass."""
    classes = [
        AutoLabelClass(name="truck", prompt="commercial delivery truck, semi"),
        AutoLabelClass(name="car", prompt="sedan, suv, automobile"),
    ]
    _, token_map = AutoLabelEngine.build_prompt_mapping(classes)

    matched = AutoLabelEngine.match_detected_label("semi", classes, token_map)
    assert matched is not None
    assert matched.name == "truck"

    matched_car = AutoLabelEngine.match_detected_label("a red sedan parked", classes, token_map)
    assert matched_car is not None
    assert matched_car.name == "car"


def test_auto_label_engine_run_preview_dino_sam2(sample_image: Path) -> None:
    """Test single image preview with mocked Grounding DINO and SAM 2."""
    mock_dino = MagicMock()
    mock_dino.detect.return_value = [
        ("truck", 0, 0.92, BoxPixel(50.0, 60.0, 200.0, 180.0)),
        ("car", 1, 0.88, BoxPixel(300.0, 100.0, 500.0, 350.0)),
    ]

    mock_sam = MagicMock()
    mask1 = np.zeros((480, 640), dtype=np.uint8)
    mask1[60:180, 50:200] = 1
    mask2 = np.zeros((480, 640), dtype=np.uint8)
    mask2[100:350, 300:500] = 1
    mock_sam.segment_boxes.return_value = [mask1, mask2]

    engine = AutoLabelEngine(
        grounding_detector=mock_dino,
        sam_segmenter=mock_sam,
    )

    config = AutoLabelConfig(
        mode=AutoLabelPipelineMode.DINO_SAM2_MASKS,
        confidence_threshold=0.35,
        classes=[
            AutoLabelClass(name="truck", prompt="delivery truck"),
            AutoLabelClass(name="car", prompt="passenger car"),
        ],
    )

    result = engine.run_preview(sample_image, config)
    assert isinstance(result, AutoLabelResult)
    assert result.count == 2
    assert "truck" in result.counts_by_class
    assert "car" in result.counts_by_class
    assert len(result.detections[0].polygon_pixels) >= 3


def test_auto_label_engine_florence2_vlm_localization(sample_image: Path) -> None:
    """Test Florence-2 VLM localization generating candidate boxes and SAM 2 polygon masks."""
    mock_vlm = MagicMock()
    mock_vlm.detect_objects.return_value = [
        {"label": "truck", "box": [50.0, 60.0, 200.0, 180.0], "score": 0.90},
    ]

    mock_sam = MagicMock()
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[60:180, 50:200] = 1
    mock_sam.segment_boxes.return_value = [mask]

    engine = AutoLabelEngine(
        sam_segmenter=mock_sam,
        vlm_helper=mock_vlm,
    )

    config = AutoLabelConfig(
        mode=AutoLabelPipelineMode.VLM_SAM2_MASKS,
        confidence_threshold=0.35,
        classes=[
            AutoLabelClass(name="truck", prompt="delivery truck"),
            AutoLabelClass(name="car", prompt="passenger car"),
        ],
    )

    result = engine.run_preview(sample_image, config)
    assert result.count == 1
    assert result.detections[0].class_name == "truck"
    assert len(result.detections[0].polygon_pixels) >= 3


def test_auto_label_engine_run_batch(sample_image: Path) -> None:
    """Test batch auto-labeling across documents."""
    mock_dino = MagicMock()
    mock_dino.detect.return_value = [
        ("motorcycle", 0, 0.95, BoxPixel(100.0, 100.0, 250.0, 300.0)),
    ]

    mock_sam = MagicMock()
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:300, 100:250] = 1
    mock_sam.segment_boxes.return_value = [mask]

    engine = AutoLabelEngine(
        grounding_detector=mock_dino,
        sam_segmenter=mock_sam,
    )

    doc = AnnotationDocument(
        image_path=sample_image,
        image_width=640,
        image_height=480,
    )

    config = AutoLabelConfig(
        mode=AutoLabelPipelineMode.DINO_SAM2_MASKS,
        classes=[AutoLabelClass(name="motorcycle", prompt="motorbike")],
    )

    progress_calls = []

    def on_progress(cur: int, tot: int, p: Path, r: AutoLabelResult) -> None:
        progress_calls.append((cur, tot, p))

    updated = engine.run_batch([doc], config, progress_callback=on_progress)
    assert len(updated) == 1
    assert sample_image in updated
    assert len(updated[sample_image].annotations) == 1
    assert updated[sample_image].annotations[0].class_name == "motorcycle"
    assert updated[sample_image].annotations[0].source == AnnotationSource.SAM2
    assert len(progress_calls) == 1


def test_auto_label_engine_run_preview_yolo_boxes(sample_image: Path) -> None:
    """Test single image preview with mocked YOLO detector in bounding box mode."""
    mock_yolo = MagicMock()
    mock_box1 = MagicMock()
    mock_box1.cls = [0]
    mock_box1.conf = [0.89]
    mock_box1.xyxy = [[50.0, 60.0, 200.0, 180.0]]

    mock_box2 = MagicMock()
    mock_box2.cls = [1]
    mock_box2.conf = [0.94]
    mock_box2.xyxy = [[300.0, 100.0, 500.0, 350.0]]

    mock_res = MagicMock()
    mock_res.boxes = [mock_box1, mock_box2]
    mock_yolo.return_value = [mock_res]
    mock_yolo.names = {0: "truck", 1: "car"}

    engine = AutoLabelEngine(yolo_detector=mock_yolo)

    config = AutoLabelConfig(
        mode=AutoLabelPipelineMode.YOLO_BOXES,
        confidence_threshold=0.35,
        classes=[
            AutoLabelClass(name="truck", prompt="truck"),
            AutoLabelClass(name="car", prompt="car"),
        ],
    )

    result = engine.run_preview(sample_image, config)
    assert isinstance(result, AutoLabelResult)
    assert result.count == 2
    assert "truck" in result.counts_by_class
    assert "car" in result.counts_by_class
    assert len(result.detections[0].polygon_pixels) == 0  # Box mode has no polygon


def test_auto_label_engine_run_preview_yolo_sam2(sample_image: Path) -> None:
    """Test single image preview with YOLO proposals segmented by SAM 2."""
    mock_yolo = MagicMock()
    mock_box = MagicMock()
    mock_box.cls = [0]
    mock_box.conf = [0.91]
    mock_box.xyxy = [[60.0, 70.0, 220.0, 200.0]]

    mock_res = MagicMock()
    mock_res.boxes = [mock_box]
    mock_yolo.return_value = [mock_res]
    mock_yolo.names = {0: "bus"}

    mock_sam = MagicMock()
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[70:200, 60:220] = 1
    mock_sam.segment_boxes.return_value = [mask]

    engine = AutoLabelEngine(
        yolo_detector=mock_yolo,
        sam_segmenter=mock_sam,
    )

    config = AutoLabelConfig(
        mode=AutoLabelPipelineMode.YOLO_SAM2_MASKS,
        confidence_threshold=0.35,
        classes=[AutoLabelClass(name="bus", prompt="city bus")],
    )

    result = engine.run_preview(sample_image, config)
    assert result.count == 1
    assert result.detections[0].class_name == "bus"
    assert len(result.detections[0].polygon_pixels) >= 3


def test_auto_label_engine_yolo_anti_duplication(sample_image: Path) -> None:
    """Test YOLO proposals with anti-duplication (NMS) suppressing duplicate overlapping boxes."""
    mock_yolo = MagicMock()
    # Box 1: high confidence car
    mock_box1 = MagicMock()
    mock_box1.cls = [0]
    mock_box1.conf = [0.92]
    mock_box1.xyxy = [[100.0, 100.0, 300.0, 300.0]]

    # Box 2: overlapping duplicate car (IoU > 0.50) with lower confidence -> should be suppressed
    mock_box2 = MagicMock()
    mock_box2.cls = [0]
    mock_box2.conf = [0.75]
    mock_box2.xyxy = [[105.0, 105.0, 305.0, 295.0]]

    # Box 3: different car in another location -> should be kept
    mock_box3 = MagicMock()
    mock_box3.cls = [0]
    mock_box3.conf = [0.88]
    mock_box3.xyxy = [[400.0, 100.0, 600.0, 300.0]]

    mock_res = MagicMock()
    mock_res.boxes = [mock_box1, mock_box2, mock_box3]
    mock_yolo.return_value = [mock_res]
    mock_yolo.names = {0: "car"}

    engine = AutoLabelEngine(yolo_detector=mock_yolo)

    config = AutoLabelConfig(
        mode=AutoLabelPipelineMode.YOLO_BOXES,
        confidence_threshold=0.35,
        box_iou_threshold=0.50,
        classes=[AutoLabelClass(name="car", prompt="car")],
    )

    result = engine.run_preview(sample_image, config)
    # 2 cars remain (duplicate suppressed, distinct car kept)
    assert result.count == 2
    assert all(d.class_name == "car" for d in result.detections)


def test_auto_label_engine_multi_model_ensemble_fusion(sample_image: Path) -> None:
    """Test multi-model ensemble fusing Grounding DINO + YOLO + Florence-2 VLM + SAM 2."""
    # Mock Grounding DINO detecting truck
    mock_dino = MagicMock()
    mock_dino.detect.return_value = [
        ("truck", 0, 0.93, BoxPixel(50.0, 60.0, 250.0, 200.0)),
    ]

    # Mock YOLO detecting car (and a duplicate truck that will be fused / deduplicated)
    mock_yolo = MagicMock()
    b1 = MagicMock()
    b1.cls = [0]
    b1.conf = [0.89]
    b1.xyxy = [[300.0, 100.0, 500.0, 300.0]]  # car

    b2 = MagicMock()
    b2.cls = [1]
    b2.conf = [0.75]
    b2.xyxy = [[52.0, 62.0, 248.0, 198.0]]  # duplicate truck with lower confidence

    mock_res = MagicMock()
    mock_res.boxes = [b1, b2]
    mock_yolo.return_value = [mock_res]
    mock_yolo.names = {0: "car", 1: "truck"}

    # Mock Florence-2 VLM detecting motorcycle
    mock_vlm = MagicMock()
    mock_vlm.detect_objects.return_value = [
        {"label": "motorcycle", "box": [450.0, 200.0, 600.0, 420.0], "score": 0.88},
    ]

    # Mock SAM 2 segmenter
    mock_sam = MagicMock()
    m1 = np.zeros((480, 640), dtype=np.uint8)
    m2 = np.zeros((480, 640), dtype=np.uint8)
    m3 = np.zeros((480, 640), dtype=np.uint8)
    mock_sam.segment_boxes.return_value = [m1, m2, m3]

    engine = AutoLabelEngine(
        grounding_detector=mock_dino,
        yolo_detector=mock_yolo,
        vlm_helper=mock_vlm,
        sam_segmenter=mock_sam,
    )

    config = AutoLabelConfig(
        mode=AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS,
        confidence_threshold=0.30,
        enable_grounding_dino=True,
        enable_yolo=True,
        enable_florence2=True,
        classes=[
            AutoLabelClass(name="truck", prompt="truck"),
            AutoLabelClass(name="car", prompt="car"),
            AutoLabelClass(name="motorcycle", prompt="motorbike"),
        ],
    )

    result = engine.run_preview(sample_image, config)
    # 3 distinct objects: truck (from DINO, duplicate from YOLO suppressed), car (from YOLO), motorcycle (from Florence-2)
    assert result.count == 3
    assert {d.class_name for d in result.detections} == {"truck", "car", "motorcycle"}


def test_auto_label_engine_multi_yolo_simultaneous_inference(sample_image: Path) -> None:
    """Test running 2 to 3 YOLO models simultaneously with prediction fusion."""
    # Model 1 (e.g. YOLO11n): detects car1 and truck
    mock_yolo1 = MagicMock()
    b1_1 = MagicMock()
    b1_1.cls = [0]
    b1_1.conf = [0.85]
    b1_1.xyxy = [[50.0, 50.0, 200.0, 200.0]]  # car
    b1_2 = MagicMock()
    b1_2.cls = [1]
    b1_2.conf = [0.88]
    b1_2.xyxy = [[300.0, 100.0, 500.0, 350.0]]  # truck
    res1 = MagicMock()
    res1.boxes = [b1_1, b1_2]
    mock_yolo1.return_value = [res1]
    mock_yolo1.names = {0: "car", 1: "truck"}

    # Model 2 (e.g. YOLOv8m): detects same car with higher confidence, plus motorcycle
    mock_yolo2 = MagicMock()
    b2_1 = MagicMock()
    b2_1.cls = [0]
    b2_1.conf = [0.94]  # higher score for same car
    b2_1.xyxy = [[52.0, 51.0, 198.0, 201.0]]  # overlapping car
    b2_2 = MagicMock()
    b2_2.cls = [2]
    b2_2.conf = [0.78]
    b2_2.xyxy = [[10.0, 300.0, 150.0, 450.0]]  # motorcycle
    res2 = MagicMock()
    res2.boxes = [b2_1, b2_2]
    mock_yolo2.return_value = [res2]
    mock_yolo2.names = {0: "car", 1: "truck", 2: "motorcycle"}

    # Model 3 (e.g. custom.pt): detects bus
    mock_yolo3 = MagicMock()
    b3_1 = MagicMock()
    b3_1.cls = [3]
    b3_1.conf = [0.91]
    b3_1.xyxy = [[400.0, 20.0, 600.0, 220.0]]  # bus
    res3 = MagicMock()
    res3.boxes = [b3_1]
    mock_yolo3.return_value = [res3]
    mock_yolo3.names = {3: "bus"}

    engine = AutoLabelEngine()
    engine._yolo_detectors["yolo11n.pt"] = mock_yolo1
    engine._yolo_detectors["yolov8m.pt"] = mock_yolo2
    engine._yolo_detectors["custom_best.pt"] = mock_yolo3

    config_3_models = AutoLabelConfig(
        mode=AutoLabelPipelineMode.YOLO_BOXES,
        confidence_threshold=0.35,
        box_iou_threshold=0.50,
        enable_yolo=True,
        yolo_models=["yolo11n.pt", "yolov8m.pt", "custom_best.pt"],
        classes=[
            AutoLabelClass(name="car", prompt="car"),
            AutoLabelClass(name="truck", prompt="truck"),
            AutoLabelClass(name="motorcycle", prompt="motorcycle"),
            AutoLabelClass(name="bus", prompt="bus"),
        ],
    )

    result = engine.run_preview(sample_image, config_3_models)
    # 4 distinct objects: car (fused/deduplicated with 0.94 score), truck (from yolo1), motorcycle (from yolo2), bus (from yolo3)
    assert result.count == 4
    classes_found = {d.class_name for d in result.detections}
    assert classes_found == {"car", "truck", "motorcycle", "bus"}

    # Check that highest score (0.94) was preserved for the car
    car_det = next(d for d in result.detections if d.class_name == "car")
    assert car_det.confidence == pytest.approx(0.94)

