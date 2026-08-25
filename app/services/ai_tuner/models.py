"""Data structures and configuration models for the AI Tuner agentic workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.services.annotation.domain import BoundingBox
from app.services.auto_label.models import AutoLabelConfig


def load_env_vars() -> dict[str, str]:
    """Helper to read .env file from project root or working directory."""
    loaded: dict[str, str] = {}
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[3] / ".env",
        Path.home() / ".env",
    ]
    for env_path in candidates:
        if env_path.is_file():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k not in os.environ and k not in loaded:
                            loaded[k] = v
            except Exception:
                pass
            break
    return loaded


@dataclass(frozen=True, slots=True)
class TunerConfig:
    """Settings controlling the AI Tuner optimization loop."""

    target_f1_score: float = 0.80
    max_iterations: int = 4
    iou_eval_threshold: float = 0.50
    model_name: str = "google/gemini-2.5-flash"
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    optimize_prompts: bool = True
    optimize_thresholds: bool = True
    max_error_crops_per_class: int = 2

    @classmethod
    def from_env(cls, **overrides: Any) -> TunerConfig:
        """Create TunerConfig populated from .env and environment variables."""
        env_file_vars = load_env_vars()
        api_key = (
            overrides.get("api_key")
            or os.getenv("OPENROUTER_API_KEY")
            or env_file_vars.get("OPENROUTER_API_KEY")
            or os.getenv("OPENCODE_API_KEY")
            or env_file_vars.get("OPENCODE_API_KEY")
        )
        model_name = (
            overrides.get("model_name")
            or os.getenv("OPENROUTER_MODEL")
            or env_file_vars.get("OPENROUTER_MODEL")
            or os.getenv("OPENCODE_MODEL")
            or env_file_vars.get("OPENCODE_MODEL")
            or "google/gemini-2.5-flash"
        )
        base_url = (
            overrides.get("base_url")
            or os.getenv("OPENROUTER_BASE_URL")
            or env_file_vars.get("OPENROUTER_BASE_URL")
            or os.getenv("OPENCODE_BASE_URL")
            or env_file_vars.get("OPENCODE_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )
        return cls(
            target_f1_score=overrides.get("target_f1_score", 0.80),
            max_iterations=overrides.get("max_iterations", 4),
            iou_eval_threshold=overrides.get("iou_eval_threshold", 0.50),
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            optimize_prompts=overrides.get("optimize_prompts", True),
            optimize_thresholds=overrides.get("optimize_thresholds", True),
            max_error_crops_per_class=overrides.get("max_error_crops_per_class", 2),
        )


@dataclass(slots=True)
class ClassMetric:
    """Detailed evaluation statistics for a single class."""

    class_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confused_with: dict[str, int] = field(default_factory=dict)

    def calculate_scores(self) -> None:
        """Compute precision, recall, and F1 based on TP/FP/FN."""
        total_pred = self.true_positives + self.false_positives
        self.precision = (self.true_positives / total_pred) if total_pred > 0 else 0.0

        total_gt = self.true_positives + self.false_negatives
        self.recall = (self.true_positives / total_gt) if total_gt > 0 else 0.0

        denom = self.precision + self.recall
        self.f1_score = (2 * self.precision * self.recall / denom) if denom > 0 else 0.0


@dataclass(frozen=True, slots=True)
class ErrorCrop:
    """Visual crop of an error region to be inspected by multimodal Vision LLM."""

    image_path: Path
    error_type: str  # "missed_ground_truth" | "false_positive" | "class_confusion"
    class_name: str
    predicted_class: str | None
    box: BoundingBox
    base64_jpeg: str | None = None
    note: str = ""


@dataclass(slots=True)
class EvaluationReport:
    """Complete evaluation report measuring predictions against Ground Truth."""

    overall_macro_f1: float = 0.0
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    total_ground_truth: int = 0
    total_detections: int = 0
    total_matches: int = 0
    class_metrics: dict[str, ClassMetric] = field(default_factory=dict)
    error_crops: list[ErrorCrop] = field(default_factory=list)
    semantic_diagnostics: list[str] = field(default_factory=list)

    @property
    def summary_text(self) -> str:
        """User-friendly metric summary."""
        return (
            f"F1: {int(round(self.overall_macro_f1 * 100))}% | "
            f"Precision: {int(round(self.overall_precision * 100))}% | "
            f"Recall: {int(round(self.overall_recall * 100))}% "
            f"({self.total_matches} matched of {self.total_ground_truth} GT objects)"
        )


@dataclass(frozen=True, slots=True)
class TunerIteration:
    """Record of a single optimization iteration step."""

    iteration_index: int
    f1_score: float
    precision: float
    recall: float
    prompt_updates: dict[str, str] = field(default_factory=dict)
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    diagnostics: str = ""
    llm_reasoning: str = ""
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class TunerResult:
    """Final outcome of the AI Tuner optimization run."""

    initial_config: AutoLabelConfig
    final_config: AutoLabelConfig
    initial_f1: float
    final_f1: float
    target_reached: bool
    iterations: list[TunerIteration] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0
    summary: str = ""
