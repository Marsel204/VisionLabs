"""Bayesian (Optuna TPE) optimizer for confidence and IoU thresholds.

Drop-in alternative to :class:`FastParametricSolver`: same ``optimize_thresholds``
signature and return value, but searches continuous ranges with a seeded TPE
sampler instead of a fixed grid. Optuna is an optional dependency
(``pip install visionlab[tuning]``); use :func:`optuna_available` before
constructing the solver.
"""

from __future__ import annotations

import importlib.util
import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.services.ai_tuner.evaluator import GroundTruthEvaluator
from app.services.ai_tuner.models import EvaluationReport
from app.services.ai_tuner.parametric_solver import apply_thresholds, as_prediction_map
from app.services.annotation.domain import AnnotationDocument
from app.services.auto_label.models import AutoLabelResult

LOGGER = logging.getLogger(__name__)


def optuna_available() -> bool:
    """Return True when the optional ``optuna`` package can be imported."""
    return importlib.util.find_spec("optuna") is not None


@dataclass(frozen=True, slots=True)
class OptunaSearchSummary:
    """Outcome of the most recent Optuna study, for logging and UI reasoning text."""

    trials: int
    best_f1: float
    baseline_f1: float
    improved: bool

    @property
    def text(self) -> str:
        best_pct = int(round(self.best_f1 * 100))
        if not self.improved:
            return f"Optuna TPE: {self.trials} trials, no threshold beat the baseline"
        return f"Optuna TPE: {self.trials} trials, best threshold F1 {best_pct}%"


class OptunaParametricSolver:
    """Searches confidence/IoU thresholds with Optuna's TPE sampler to maximize macro F1."""

    def __init__(
        self,
        evaluator: GroundTruthEvaluator | None = None,
        n_trials: int = 60,
        seed: int | None = 42,
        timeout_seconds: float | None = None,
        conf_range: tuple[float, float] = (0.05, 0.90),
        iou_range: tuple[float, float] = (0.30, 0.80),
    ) -> None:
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1")
        if not optuna_available():
            raise ImportError(
                "Optuna is not installed. Install it with `pip install optuna` "
                "(or `uv sync --extra tuning`) to use Optuna threshold search."
            )
        self.evaluator = evaluator or GroundTruthEvaluator()
        self.n_trials = n_trials
        self.seed = seed
        self.timeout_seconds = timeout_seconds
        self.conf_range = conf_range
        self.iou_range = iou_range
        self.last_summary: OptunaSearchSummary | None = None

    def optimize_thresholds(
        self,
        raw_predictions: dict[Path, AutoLabelResult] | Sequence[AutoLabelResult],
        ground_truth: dict[Path, AnnotationDocument],
        initial_conf: float = 0.35,
        initial_iou: float = 0.45,
    ) -> tuple[float, float, EvaluationReport]:
        """Run a TPE study over thresholds; never returns a result worse than the baseline."""
        import optuna

        pred_map = as_prediction_map(raw_predictions)
        baseline_report = self.evaluator.evaluate(pred_map, ground_truth, extract_crops=False)
        baseline_f1 = baseline_report.overall_macro_f1

        conf_lo, conf_hi = self.conf_range
        iou_lo, iou_hi = self.iou_range

        def objective(trial: optuna.Trial) -> float:
            conf = trial.suggest_float("conf", conf_lo, conf_hi, step=0.01)
            iou = trial.suggest_float("iou", iou_lo, iou_hi, step=0.01)
            report = self.evaluator.evaluate(
                apply_thresholds(pred_map, conf, iou), ground_truth, extract_crops=False
            )
            return report.overall_macro_f1

        previous_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=optuna.exceptions.ExperimentalWarning)
                sampler = optuna.samplers.TPESampler(
                    seed=self.seed,
                    n_startup_trials=max(1, min(10, self.n_trials // 3)),
                    multivariate=True,
                )
                study = optuna.create_study(direction="maximize", sampler=sampler)
                # Start the search from the current operating point.
                study.enqueue_trial(
                    {
                        "conf": _clamp_to_step(initial_conf, conf_lo, conf_hi),
                        "iou": _clamp_to_step(initial_iou, iou_lo, iou_hi),
                    }
                )
                study.optimize(
                    objective,
                    n_trials=self.n_trials,
                    timeout=self.timeout_seconds,
                    show_progress_bar=False,
                )
        finally:
            optuna.logging.set_verbosity(previous_verbosity)

        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        best_conf, best_iou = initial_conf, initial_iou
        best_f1 = baseline_f1
        improved = False
        if completed:
            # Ties go to the earliest trial (the enqueued current thresholds come first).
            best_trial = max(completed, key=lambda t: (t.value, -t.number))
            if best_trial.value is not None and best_trial.value > baseline_f1:
                best_conf = float(best_trial.params["conf"])
                best_iou = float(best_trial.params["iou"])
                best_f1 = float(best_trial.value)
                improved = True

        self.last_summary = OptunaSearchSummary(
            trials=len(completed), best_f1=best_f1, baseline_f1=baseline_f1, improved=improved
        )
        LOGGER.info(self.last_summary.text)

        best_report_with_crops = self.evaluator.evaluate(
            apply_thresholds(pred_map, best_conf, best_iou), ground_truth, extract_crops=True
        )
        return round(best_conf, 2), round(best_iou, 2), best_report_with_crops


def _clamp_to_step(value: float, low: float, high: float, step: float = 0.01) -> float:
    """Clamp a value into the search range and snap it to the step grid Optuna expects."""
    clamped = min(max(value, low), high)
    snapped = low + round((clamped - low) / step) * step
    return round(min(snapped, high), 2)
