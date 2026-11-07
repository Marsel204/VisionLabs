"""High-speed mathematical hyperparameter optimizer for confidence and IoU thresholds."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from app.services.ai_tuner.evaluator import GroundTruthEvaluator
from app.services.ai_tuner.models import EvaluationReport
from app.services.annotation.domain import AnnotationDocument
from app.services.auto_label.engine import compute_box_iou
from app.services.auto_label.models import AutoLabelDetection, AutoLabelResult

LOGGER = logging.getLogger(__name__)


def as_prediction_map(
    raw_predictions: dict[Path, AutoLabelResult] | Sequence[AutoLabelResult],
) -> dict[Path, AutoLabelResult]:
    """Normalize predictions into a path-keyed mapping."""
    if isinstance(raw_predictions, dict):
        return raw_predictions
    return {r.image_path: r for r in raw_predictions}


def apply_thresholds(
    pred_map: dict[Path, AutoLabelResult], conf: float, iou: float
) -> dict[Path, AutoLabelResult]:
    """Re-filter cached detections with a confidence cut-off and greedy IoU deduplication."""
    filtered_results: dict[Path, AutoLabelResult] = {}
    for img_path, res in pred_map.items():
        conf_filtered = [d for d in res.detections if d.confidence >= conf]
        conf_filtered.sort(key=lambda d: d.confidence, reverse=True)
        deduped: list[AutoLabelDetection] = []
        for det in conf_filtered:
            if not any(compute_box_iou(ex.box, det.box) >= iou for ex in deduped):
                deduped.append(det)
        filtered_results[img_path] = AutoLabelResult(
            image_path=res.image_path,
            image_width=res.image_width,
            image_height=res.image_height,
            detections=deduped,
            elapsed_seconds=res.elapsed_seconds,
        )
    return filtered_results


class FastParametricSolver:
    """Finds the optimal confidence and IoU deduplication thresholds on candidate detections."""

    CONF_GRID: tuple[float, ...] = (
        0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
    )
    IOU_GRID: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65)

    def __init__(self, evaluator: GroundTruthEvaluator | None = None) -> None:
        self.evaluator = evaluator or GroundTruthEvaluator()

    def optimize_thresholds(
        self,
        raw_predictions: dict[Path, AutoLabelResult] | Sequence[AutoLabelResult],
        ground_truth: dict[Path, AnnotationDocument],
        initial_conf: float = 0.35,
        initial_iou: float = 0.45,
    ) -> tuple[float, float, EvaluationReport]:
        """Sweep confidence and IoU thresholds in milliseconds to maximize macro F1 score."""
        pred_map = as_prediction_map(raw_predictions)

        best_conf = initial_conf
        best_iou = initial_iou
        best_report = self.evaluator.evaluate(pred_map, ground_truth, extract_crops=False)
        best_f1 = best_report.overall_macro_f1

        for conf in self.CONF_GRID:
            for iou in self.IOU_GRID:
                report = self.evaluator.evaluate(
                    apply_thresholds(pred_map, conf, iou), ground_truth, extract_crops=False
                )
                if report.overall_macro_f1 > best_f1:
                    best_f1 = report.overall_macro_f1
                    best_conf = conf
                    best_iou = iou
                    best_report = report

        # Extract final error crops on best result
        best_report_with_crops = self.evaluator.evaluate(
            apply_thresholds(pred_map, best_conf, best_iou), ground_truth, extract_crops=True
        )
        return round(best_conf, 2), round(best_iou, 2), best_report_with_crops
