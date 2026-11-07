"""Tests for Optuna-backed threshold search and solver selection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from app.services.ai_tuner import optuna_solver
from app.services.ai_tuner.models import TunerConfig
from app.services.ai_tuner.optuna_solver import OptunaParametricSolver, _clamp_to_step
from app.services.ai_tuner.parametric_solver import FastParametricSolver
from app.services.ai_tuner.tuner_engine import AITunerEngine
from app.services.annotation.domain import Annotation, AnnotationDocument, BoundingBox
from app.services.auto_label.models import (
    AutoLabelClass,
    AutoLabelConfig,
    AutoLabelDetection,
    AutoLabelResult,
)

pytest.importorskip("optuna")


@pytest.fixture
def noisy_case(tmp_path: Path) -> tuple[Path, AutoLabelResult, AnnotationDocument]:
    img_path = tmp_path / "optuna_eval.jpg"
    Image.new("RGB", (640, 480), color=(180, 180, 180)).save(img_path)
    gt_doc = AnnotationDocument(
        image_path=img_path,
        image_width=640,
        image_height=480,
        annotations=(Annotation(class_name="car", box=BoundingBox(0.1, 0.1, 0.3, 0.3)),),
    )
    raw = AutoLabelResult(
        image_path=img_path,
        image_width=640,
        image_height=480,
        detections=[
            AutoLabelDetection("car", 0.85, BoundingBox(0.1, 0.1, 0.3, 0.3)),
            # Near-duplicate of the true box: only IoU dedup can remove it.
            AutoLabelDetection("car", 0.80, BoundingBox(0.11, 0.11, 0.31, 0.31)),
            AutoLabelDetection("car", 0.20, BoundingBox(0.4, 0.4, 0.5, 0.5)),
            AutoLabelDetection("car", 0.30, BoundingBox(0.8, 0.8, 0.9, 0.9)),
        ],
    )
    return img_path, raw, gt_doc


def test_optuna_solver_finds_perfect_thresholds(
    noisy_case: tuple[Path, AutoLabelResult, AnnotationDocument],
) -> None:
    img_path, raw, gt_doc = noisy_case
    solver = OptunaParametricSolver(n_trials=40, seed=7)

    conf, iou, report = solver.optimize_thresholds(
        {img_path: raw}, {img_path: gt_doc}, initial_conf=0.15, initial_iou=0.95
    )

    assert conf > 0.30
    assert iou < 0.81  # duplicate box overlap is ~0.82, so it must be deduplicated
    assert report.overall_macro_f1 == 1.0
    assert solver.last_summary is not None
    assert solver.last_summary.improved
    assert solver.last_summary.trials == 40


def test_optuna_solver_keeps_baseline_when_nothing_improves(
    noisy_case: tuple[Path, AutoLabelResult, AnnotationDocument],
) -> None:
    img_path, _raw, gt_doc = noisy_case
    perfect = AutoLabelResult(
        image_path=img_path,
        image_width=640,
        image_height=480,
        detections=[AutoLabelDetection("car", 0.9, BoundingBox(0.1, 0.1, 0.3, 0.3))],
    )
    solver = OptunaParametricSolver(n_trials=10, seed=1)

    conf, iou, report = solver.optimize_thresholds(
        {img_path: perfect}, {img_path: gt_doc}, initial_conf=0.42, initial_iou=0.55
    )

    assert (conf, iou) == (0.42, 0.55)
    assert report.overall_macro_f1 == 1.0
    assert solver.last_summary is not None and not solver.last_summary.improved


def test_optuna_solver_is_deterministic_with_seed(
    noisy_case: tuple[Path, AutoLabelResult, AnnotationDocument],
) -> None:
    img_path, raw, gt_doc = noisy_case
    runs = [
        OptunaParametricSolver(n_trials=15, seed=123).optimize_thresholds(
            {img_path: raw}, {img_path: gt_doc}, initial_conf=0.15, initial_iou=0.95
        )[:2]
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


def test_clamp_to_step_snaps_into_range() -> None:
    assert _clamp_to_step(0.01, 0.05, 0.90) == 0.05
    assert _clamp_to_step(0.95, 0.05, 0.90) == 0.90
    assert _clamp_to_step(0.333, 0.05, 0.90) == 0.33


def test_tuner_config_validates_search_mode() -> None:
    with pytest.raises(ValueError):
        TunerConfig(threshold_search="random")
    with pytest.raises(ValueError):
        TunerConfig(optuna_trials=0)
    assert TunerConfig.from_env(threshold_search="OPTUNA").threshold_search == "optuna"


@pytest.mark.parametrize(
    ("mode", "available", "expected"),
    [
        ("grid", True, FastParametricSolver),
        ("optuna", True, OptunaParametricSolver),
        ("auto", True, OptunaParametricSolver),
        ("auto", False, FastParametricSolver),
    ],
)
def test_engine_resolves_solver(
    monkeypatch: pytest.MonkeyPatch, mode: str, available: bool, expected: type
) -> None:
    monkeypatch.setattr(
        "app.services.ai_tuner.tuner_engine.optuna_available", lambda: available
    )
    engine = AITunerEngine(auto_label_engine=MagicMock())
    solver = engine._resolve_threshold_solver(TunerConfig(threshold_search=mode))
    assert isinstance(solver, expected)


def test_engine_explicit_optuna_without_package_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.ai_tuner.tuner_engine.optuna_available", lambda: False)
    engine = AITunerEngine(auto_label_engine=MagicMock())
    with pytest.raises(ImportError, match="optuna"):
        engine._resolve_threshold_solver(TunerConfig(threshold_search="optuna"))


def test_solver_constructor_errors_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(optuna_solver, "optuna_available", lambda: False)
    with pytest.raises(ImportError):
        OptunaParametricSolver()


def test_engine_injected_solver_wins() -> None:
    injected = FastParametricSolver()
    engine = AITunerEngine(auto_label_engine=MagicMock(), parametric_solver=injected)
    assert engine._resolve_threshold_solver(TunerConfig(threshold_search="optuna")) is injected


def test_engine_runs_end_to_end_with_optuna(
    noisy_case: tuple[Path, AutoLabelResult, AnnotationDocument],
) -> None:
    img_path, raw, gt_doc = noisy_case
    auto_label = MagicMock()
    auto_label.run_preview.return_value = raw

    engine = AITunerEngine(auto_label_engine=auto_label, llm_client=MagicMock(is_configured=False))
    result = engine.run_tuning(
        sample_images=[img_path],
        ground_truth={img_path: gt_doc},
        initial_config=AutoLabelConfig(
            classes=[AutoLabelClass(name="car", prompt="car")],
            confidence_threshold=0.15,
            box_iou_threshold=0.95,
        ),
        tuner_config=TunerConfig(
            threshold_search="optuna", optuna_trials=40, optuna_seed=7, max_iterations=2
        ),
    )

    assert result.target_reached
    assert result.final_f1 == 1.0
    assert result.final_config.confidence_threshold > 0.30
    assert "Optuna TPE" in result.iterations[-1].llm_reasoning


def test_best_config_prompts_are_not_mutated_by_later_iterations() -> None:
    engine = AITunerEngine(auto_label_engine=MagicMock())
    classes = [AutoLabelClass(name="car", prompt="car")]
    snapshot = engine._update_config_classes(AutoLabelConfig(classes=classes), classes)
    classes[0].prompt = "car, sedan"
    assert snapshot.classes[0].prompt == "car"
