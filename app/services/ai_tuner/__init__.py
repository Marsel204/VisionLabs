"""AI Tuner package for automated prompt and hyperparameter optimization."""

from app.services.ai_tuner.evaluator import GroundTruthEvaluator
from app.services.ai_tuner.llm_client import OpenRouterVisionClient
from app.services.ai_tuner.models import (
    ClassMetric,
    ErrorCrop,
    EvaluationReport,
    TunerConfig,
    TunerIteration,
    TunerResult,
)
from app.services.ai_tuner.parametric_solver import FastParametricSolver
from app.services.ai_tuner.tuner_engine import AITunerEngine

__all__ = [
    "AITunerEngine",
    "ClassMetric",
    "ErrorCrop",
    "EvaluationReport",
    "FastParametricSolver",
    "GroundTruthEvaluator",
    "OpenRouterVisionClient",
    "TunerConfig",
    "TunerIteration",
    "TunerResult",
]
