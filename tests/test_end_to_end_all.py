"""Comprehensive end-to-end integration test exercising all core capabilities:

1. Database/Dataset Indexing and Batch Metadata Updates
2. COCO & YOLO Dataset Import with Overlap Removal
3. Auto-Labeling with Grounding DINO, Florence-2 VLM, and SAM 2 Polygons
4. AI Auto-Tuner Iterative Prompt Optimization & Parameter Calibration
5. Active Learning Difficulty Ranking and Review Queue
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from app.services.active_learning import (
    ActiveLearningConfig,
    ActiveLearningEngine,
    ImageAnalysis,
    RankingMode,
)
from app.services.ai_tuner.models import TunerConfig
from app.services.ai_tuner.tuner_engine import AITunerEngine
from app.services.annotation.domain import (
    Annotation,
    AnnotationDocument,
    AnnotationSource,
    BoundingBox,
)
from app.services.auto_label.engine import AutoLabelEngine
from app.services.auto_label.models import (
    AutoLabelClass,
    AutoLabelConfig,
    AutoLabelPipelineMode,
)
from app.services.dataset.coco_importer import CocoImporter
from app.services.dataset.index import DatasetIndex
from app.services.dataset.yolo_importer import YoloImporter
from pipeline_bridge import BoxPixel


def _create_sample_image(path: Path, width: int = 640, height: int = 480) -> Path:
    """Create a sample synthetic test image."""
    img = Image.new("RGB", (width, height), color=(60, 80, 100))
    img.save(path)
    return path


def test_e2e_database_indexing_and_batch_metadata(tmp_path: Path) -> None:
    """1. Database Loading: Indexing image collections and batch updating metadata."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    paths: list[Path] = []
    for i in range(10):
        p = _create_sample_image(img_dir / f"img_{i:03d}.jpg", 640, 480)
        paths.append(p)

    db_path = tmp_path / "dataset_index.sqlite"
    with DatasetIndex(db_path) as index:
        # Scan and index collection
        scanned_count = index.scan(img_dir)
        assert scanned_count == 10
        assert index.count() == 10

        # Batch metadata update
        metadata_batch = [(p, 640, 480) for p in paths]
        index.set_metadata_batch(metadata_batch)

        # Set difficulty scores
        for i, p in enumerate(paths):
            index.set_difficulty(p, round(i * 0.1, 2), status="reviewed" if i > 5 else "unreviewed")

        indexed_paths = list(index.iter_paths(page_size=5))
        assert len(indexed_paths) == 10


def test_e2e_coco_and_yolo_dataset_imports(tmp_path: Path) -> None:
    """2. Dataset Import: Test both COCO and YOLO formats with automatic overlap cleanup."""
    # Setup COCO source
    coco_root = tmp_path / "coco_source"
    coco_root.mkdir()
    _create_sample_image(coco_root / "traffic.jpg", 100, 100)

    coco_json = tmp_path / "annotations.json"
    coco_json.write_text(
        json.dumps({
            "images": [{"id": 1, "file_name": "traffic.jpg", "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "motorcycle"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50]},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [12, 12, 48, 48]},  # Overlap
                {"id": 3, "image_id": 1, "category_id": 2, "bbox": [60, 60, 30, 30]},
            ],
        }),
        encoding="utf-8",
    )

    coco_result = CocoImporter().import_dataset(
        coco_json, coco_root, tmp_path / "coco_proj", remove_overlaps=True
    )
    assert coco_result.report.images_imported == 1
    assert coco_result.report.overlapping_removed == 1
    assert len(coco_result.documents) == 1
    assert len(coco_result.documents[0].annotations) == 2

    # Setup YOLO source
    yolo_root = tmp_path / "yolo_source"
    (yolo_root / "images").mkdir(parents=True)
    (yolo_root / "labels").mkdir(parents=True)
    _create_sample_image(yolo_root / "images" / "yolo_sample.jpg", 200, 200)

    (yolo_root / "labels" / "yolo_sample.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.2 0.2\n", encoding="utf-8"
    )
    data_yaml = yolo_root / "data.yaml"
    data_yaml.write_text(
        f"names:\n  0: car\n  1: motorcycle\npath: {yolo_root}\n",
        encoding="utf-8",
    )

    yolo_result = YoloImporter().import_dataset(
        data_yaml, tmp_path / "yolo_proj", remove_overlaps=True
    )
    assert yolo_result.report.images_imported == 1
    assert len(yolo_result.documents) == 1
    assert len(yolo_result.documents[0].annotations) == 2


def test_e2e_auto_labeling_pipeline(tmp_path: Path) -> None:
    """3. Auto-Labeling Pipeline: Grounding DINO + Florence-2 VLM + SAM 2 Polygons."""
    test_img = _create_sample_image(tmp_path / "auto_label.jpg", 640, 480)

    mock_dino = MagicMock()
    mock_dino.detect.return_value = [
        ("car", 0, 0.94, BoxPixel(40.0, 50.0, 220.0, 200.0)),
        ("motorcycle", 1, 0.89, BoxPixel(300.0, 150.0, 450.0, 380.0)),
    ]

    mock_vlm = MagicMock()
    mock_vlm.detect_objects.return_value = [
        {"label": "car", "box": [40.0, 50.0, 220.0, 200.0], "score": 0.95}
    ]

    mask1 = np.zeros((480, 640), dtype=np.uint8)
    mask1[50:200, 40:220] = 1
    mask2 = np.zeros((480, 640), dtype=np.uint8)
    mask2[150:380, 300:450] = 1

    mock_sam = MagicMock()
    mock_sam.segment_boxes.return_value = [mask1, mask2]

    engine = AutoLabelEngine(
        grounding_detector=mock_dino,
        vlm_helper=mock_vlm,
        sam_segmenter=mock_sam,
    )

    config = AutoLabelConfig(
        mode=AutoLabelPipelineMode.DINO_SAM2_MASKS,
        confidence_threshold=0.30,
        classes=[
            AutoLabelClass(name="car", prompt="car, sedan, suv"),
            AutoLabelClass(name="motorcycle", prompt="motorcycle, motorbike"),
        ],
        enable_florence2=True,
    )

    result = engine.run_preview(test_img, config)
    assert len(result.detections) == 2
    assert {d.class_name for d in result.detections} == {"car", "motorcycle"}

    # Verify polygon generation
    for det in result.detections:
        assert det.polygon_pixels is not None
        assert len(det.polygon_pixels) >= 3  # Valid polygon with at least 3 vertices
        assert len(det.polygon_normalized) >= 3
        assert det.confidence > 0.80


def test_e2e_ai_tuner_auto_adjusting_prompt(tmp_path: Path) -> None:
    """4. AI Auto-Tuner: Autonomous prompt adjustment and F1 evaluation."""
    img_sample = _create_sample_image(tmp_path / "tuner_sample.jpg", 640, 480)

    gt_doc = AnnotationDocument(
        image_path=img_sample,
        image_width=640,
        image_height=480,
        annotations=(
            Annotation(class_name="motorcycle", box=BoundingBox(0.1, 0.1, 0.4, 0.4)),
            Annotation(class_name="car", box=BoundingBox(0.5, 0.5, 0.8, 0.8)),
        ),
    )

    initial_config = AutoLabelConfig(
        classes=[
            AutoLabelClass(name="motorcycle", prompt="bike"),  # suboptimal prompt
            AutoLabelClass(name="car", prompt="sedan"),
        ],
        confidence_threshold=0.40,
    )

    tuner_config = TunerConfig(
        target_f1_score=0.85,
        max_iterations=3,
        optimize_prompts=True,
        optimize_thresholds=True,
    )

    mock_auto_label_engine = MagicMock()

    def mock_preview(path: Path, cfg: AutoLabelConfig):
        mc_prompt = next((c.prompt for c in cfg.classes if c.name == "motorcycle"), "")
        dets = [
            MagicMock(
                class_name="car",
                confidence=0.91,
                box=BoundingBox(0.5, 0.5, 0.8, 0.8),
                polygon=None,
            )
        ]
        # Once prompt is expanded with motorbike or scooter, motorcycle is detected
        if "motorbike" in mc_prompt or "scooter" in mc_prompt or "motorcycle" in mc_prompt:
            dets.append(
                MagicMock(
                    class_name="motorcycle",
                    confidence=0.88,
                    box=BoundingBox(0.1, 0.1, 0.4, 0.4),
                    polygon=None,
                )
            )
        res = MagicMock()
        res.image_path = path
        res.detections = dets
        return res

    mock_auto_label_engine.run_preview.side_effect = mock_preview

    tuner = AITunerEngine(auto_label_engine=mock_auto_label_engine)
    result = tuner.run_tuning(
        sample_images=[img_sample],
        ground_truth={img_sample: gt_doc},
        initial_config=initial_config,
        tuner_config=tuner_config,
    )

    # Prompt auto-adjustment verified
    assert result.final_f1 >= 0.85
    assert result.target_reached is True
    assert len(result.iterations) >= 2

    tuned_mc = next(c for c in result.final_config.classes if c.name == "motorcycle")
    assert any(syn in tuned_mc.prompt for syn in ("motorbike", "scooter", "motorcycle"))
    assert result.iterations[1].llm_reasoning != ""


def test_e2e_active_learning_review_queue(tmp_path: Path) -> None:
    """5. Active Learning: Difficulty scoring and review queue ranking."""
    cache_path = tmp_path / "al_cache.sqlite"
    config = ActiveLearningConfig(cache_path=cache_path)
    engine = ActiveLearningEngine(config)

    # Generate 5 sample image analyses with differing difficulty
    analyses: list[ImageAnalysis] = []
    for i in range(5):
        img_p = _create_sample_image(tmp_path / f"al_{i}.jpg", 640, 480)
        count = 25 if i == 0 else (5 - i)
        conf = 0.25 if i == 0 else 0.90
        dets = tuple(
            MagicMock(
                class_name="motorcycle" if j % 2 == 0 else "car",
                confidence=conf,
                box=BoundingBox(0.1, 0.1, 0.4, 0.4),
                source=AnnotationSource.YOLO,
            )
            for j in range(count)
        )
        analyses.append(ImageAnalysis(image_path=img_p, detections=dets))

    ranked = engine.score_many(analyses, ranking=RankingMode.HIGHEST_DIFFICULTY)
    assert len(ranked) == 5
    # The hardest image with low confidence and high density must be ranked first
    assert ranked[0].image_path == tmp_path / "al_0.jpg"
    assert ranked[0].difficulty_score > ranked[-1].difficulty_score

    # Re-score to verify cache speed
    re_scored = engine.score(analyses[0])
    assert re_scored.cached is True

    engine.close()
