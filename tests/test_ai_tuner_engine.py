"""Unit tests for AITunerEngine orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from app.services.ai_tuner.models import TunerConfig
from app.services.ai_tuner.tuner_engine import AITunerEngine
from app.services.annotation.domain import Annotation, AnnotationDocument, BoundingBox
from app.services.auto_label.models import (
    AutoLabelClass,
    AutoLabelConfig,
    AutoLabelDetection,
    AutoLabelResult,
)


@pytest.fixture
def dummy_images(tmp_path: Path) -> list[Path]:
    paths = []
    for i in range(2):
        p = tmp_path / f"engine_test_{i}.jpg"
        img = Image.new("RGB", (640, 480), color=(100 + i * 20, 100, 100))
        img.save(p)
        paths.append(p)
    return paths


def test_tuner_engine_reaches_target(dummy_images: list[Path]) -> None:
    img1, img2 = dummy_images

    # Ground truth documents
    gt_docs = {
        img1: AnnotationDocument(
            image_path=img1,
            image_width=640,
            image_height=480,
            annotations=(Annotation(class_name="truck", box=BoundingBox(0.1, 0.1, 0.4, 0.4)),),
        ),
        img2: AnnotationDocument(
            image_path=img2,
            image_width=640,
            image_height=480,
            annotations=(Annotation(class_name="car", box=BoundingBox(0.2, 0.2, 0.5, 0.5)),),
        ),
    }

    initial_config = AutoLabelConfig(
        classes=[
            AutoLabelClass(name="truck", prompt="truck"),
            AutoLabelClass(name="car", prompt="car"),
        ],
        confidence_threshold=0.35,
    )

    tuner_config = TunerConfig(
        target_f1_score=0.80,
        max_iterations=3,
    )

    # Mock AutoLabelEngine that returns low score on iter 0 and high score on refined prompt
    mock_auto_label_engine = MagicMock()

    def mock_run_preview(img_path: Path, config: AutoLabelConfig) -> AutoLabelResult:
        truck_prompt = next((c.prompt for c in config.classes if c.name == "truck"), "")
        if "semi-trailer" in truck_prompt or "flatbed" in truck_prompt:
            # Tuned detections
            if img_path == img1:
                return AutoLabelResult(
                    image_path=img1,
                    image_width=640,
                    image_height=480,
                    detections=[AutoLabelDetection("truck", 0.92, BoundingBox(0.1, 0.1, 0.4, 0.4))],
                )
            else:
                return AutoLabelResult(
                    image_path=img2,
                    image_width=640,
                    image_height=480,
                    detections=[AutoLabelDetection("car", 0.88, BoundingBox(0.2, 0.2, 0.5, 0.5))],
                )
        else:
            # Baseline detections (missed truck on img1)
            if img_path == img1:
                return AutoLabelResult(
                    image_path=img1, image_width=640, image_height=480, detections=[]
                )
            else:
                return AutoLabelResult(
                    image_path=img2,
                    image_width=640,
                    image_height=480,
                    detections=[AutoLabelDetection("car", 0.88, BoundingBox(0.2, 0.2, 0.5, 0.5))],
                )

    mock_auto_label_engine.run_preview.side_effect = mock_run_preview

    # Mock LLM Client
    mock_llm_client = MagicMock()
    mock_llm_client.is_configured = True
    mock_llm_client.refine_prompts_with_vision.return_value = (
        {"truck": "commercial heavy truck, semi-trailer, flatbed delivery lorry"},
        "Added semi-trailer and flatbed keywords to catch missed heavy vehicles.",
    )

    engine = AITunerEngine(
        auto_label_engine=mock_auto_label_engine,
        llm_client=mock_llm_client,
    )

    result = engine.run_tuning(
        sample_images=dummy_images,
        ground_truth=gt_docs,
        initial_config=initial_config,
        tuner_config=tuner_config,
    )

    assert result.initial_f1 < 0.80
    assert result.final_f1 >= 0.80
    assert result.target_reached is True
    assert len(result.iterations) >= 2
    assert "semi-trailer" in next(
        c.prompt for c in result.final_config.classes if c.name == "truck"
    )
