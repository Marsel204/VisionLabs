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


class FastParametricSolver:
    """Finds the optimal confidence and IoU deduplication thresholds on candidate detections."""

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
        if isinstance(raw_predictions, dict):
            pred_map = raw_predictions
        else:
            pred_map = {r.image_path: r for r in raw_predictions}

        conf_grid = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        iou_grid = [0.35, 0.45, 0.55, 0.65]

        best_conf = initial_conf
        best_iou = initial_iou
        best_report = self.evaluator.evaluate(pred_map, ground_truth, extract_crops=False)
        best_f1 = best_report.overall_macro_f1

        for conf in conf_grid:
            for iou in iou_grid:
                # Filter and deduplicate detections with test parameters
                filtered_results: dict[Path, AutoLabelResult] = {}
                for img_path, res in pred_map.items():
                    # 1. Filter by confidence
                    conf_filtered = [d for d in res.detections if d.confidence >= conf]
                    # 2. Sort descending by confidence
                    conf_filtered.sort(key=lambda d: d.confidence, reverse=True)
                    # 3. Apply IoU deduplication
                    deduped: list[AutoLabelDetection] = []
                    for det in conf_filtered:
                        overlap = any(
                            compute_box_iou(ex.box, det.box) >= iou for ex in deduped
                        )
                        if not overlap:
                            deduped.append(det)

                    filtered_results[img_path] = AutoLabelResult(
                        image_path=res.image_path,
                        image_width=res.image_width,
                        image_height=res.image_height,
                        detections=deduped,
                        elapsed_seconds=res.elapsed_seconds,
                    )

                report = self.evaluator.evaluate(
                    filtered_results, ground_truth, extract_crops=False
                )
                if report.overall_macro_f1 > best_f1:
                    best_f1 = report.overall_macro_f1
                    best_conf = conf
                    best_iou = iou
                    best_report = report

        # Extract final error crops on best result
        final_filtered: dict[Path, AutoLabelResult] = {}
        for img_path, res in pred_map.items():
            conf_filtered = [d for d in res.detections if d.confidence >= best_conf]
            conf_filtered.sort(key=lambda d: d.confidence, reverse=True)
            deduped = []
            for det in conf_filtered:
                overlap = any(
                    compute_box_iou(ex.box, det.box) >= best_iou for ex in deduped
                )
                if not overlap:
                    deduped.append(det)
            final_filtered[img_path] = AutoLabelResult(
                image_path=res.image_path,
                image_width=res.image_width,
                image_height=res.image_height,
                detections=deduped,
                elapsed_seconds=res.elapsed_seconds,
            )

        best_report_with_crops = self.evaluator.evaluate(
            final_filtered, ground_truth, extract_crops=True
        )
        return round(best_conf, 2), round(best_iou, 2), best_report_with_crops
