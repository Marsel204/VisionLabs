"""Agentic orchestrator driving iterative prompt refinement and threshold tuning."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from app.services.ai_tuner.evaluator import GroundTruthEvaluator
from app.services.ai_tuner.llm_client import OpenRouterVisionClient
from app.services.ai_tuner.models import TunerConfig, TunerIteration, TunerResult
from app.services.ai_tuner.parametric_solver import FastParametricSolver
from app.services.annotation.domain import AnnotationDocument
from app.services.auto_label.engine import AutoLabelEngine
from app.services.auto_label.models import AutoLabelClass, AutoLabelConfig, AutoLabelResult

LOGGER = logging.getLogger(__name__)


class AITunerEngine:
    """Orchestrates the iterative loop: Infer -> Evaluate -> Fast Optimize -> Vision Reflection."""

    def __init__(
        self,
        auto_label_engine: AutoLabelEngine | None = None,
        evaluator: GroundTruthEvaluator | None = None,
        parametric_solver: FastParametricSolver | None = None,
        llm_client: OpenRouterVisionClient | None = None,
    ) -> None:
        self.auto_label_engine = auto_label_engine or AutoLabelEngine()
        self.evaluator = evaluator or GroundTruthEvaluator()
        self.parametric_solver = parametric_solver or FastParametricSolver(self.evaluator)
        self.llm_client = llm_client

    def run_tuning(
        self,
        sample_images: list[Path],
        ground_truth: dict[Path, AnnotationDocument],
        initial_config: AutoLabelConfig,
        tuner_config: TunerConfig,
        progress_callback: Callable[[TunerIteration], None] | None = None,
    ) -> TunerResult:
        """Execute autonomous tuning loop until target F1 is achieved or max iterations reached."""
        start_time = time.perf_counter()

        # Validate ground truth sample coverage
        valid_gt = {
            p: doc
            for p, doc in ground_truth.items()
            if p in sample_images and doc.annotations
        }
        if not valid_gt:
            raise ValueError(
                "No ground truth annotations found on any of the provided sample images."
            )

        current_config = initial_config
        active_classes = [
            AutoLabelClass(c.name, c.prompt, c.color, c.enabled)
            for c in initial_config.classes
        ]

        llm_client = self.llm_client or OpenRouterVisionClient(tuner_config)

        iterations: list[TunerIteration] = []
        best_f1 = 0.0
        best_config = current_config

        # ----------------------------------------------------------------------
        # Iteration 0: Baseline Evaluation
        # ----------------------------------------------------------------------
        iter_start = time.perf_counter()
        predictions = self._infer_samples(sample_images, current_config)
        report = self.evaluator.evaluate(predictions, valid_gt, extract_crops=True)
        initial_f1 = report.overall_macro_f1
        best_f1 = initial_f1

        iter_0 = TunerIteration(
            iteration_index=0,
            f1_score=report.overall_macro_f1,
            precision=report.overall_precision,
            recall=report.overall_recall,
            prompt_updates={c.name: c.prompt for c in active_classes},
            confidence_threshold=current_config.confidence_threshold,
            iou_threshold=current_config.box_iou_threshold,
            diagnostics="Initial baseline performance",
            llm_reasoning="Baseline evaluation prior to optimization",
            elapsed_seconds=time.perf_counter() - iter_start,
        )
        iterations.append(iter_0)
        if progress_callback:
            progress_callback(iter_0)

        if initial_f1 >= tuner_config.target_f1_score:
            LOGGER.info("Baseline already meets or exceeds target F1 (%0.2f)", initial_f1)
            init_pct = int(round(initial_f1 * 100))
            return TunerResult(
                initial_config=initial_config,
                final_config=current_config,
                initial_f1=initial_f1,
                final_f1=initial_f1,
                target_reached=True,
                iterations=iterations,
                total_elapsed_seconds=time.perf_counter() - start_time,
                summary=f"Baseline accuracy already reached target ({init_pct}%)",
            )

        # ----------------------------------------------------------------------
        # Iterations 1..N: Agentic Optimization Loop
        # ----------------------------------------------------------------------
        for iter_idx in range(1, tuner_config.max_iterations + 1):
            iter_start = time.perf_counter()
            reasoning_summary = ""
            prompt_diffs: dict[str, str] = {}

            # Phase 1: Fast Parametric Optimizer (Confidence & IoU sweep)
            if tuner_config.optimize_thresholds:
                opt_conf, opt_iou, opt_report = (
                    self.parametric_solver.optimize_thresholds(
                        predictions,
                        valid_gt,
                        initial_conf=current_config.confidence_threshold,
                        initial_iou=current_config.box_iou_threshold,
                    )
                )
                if opt_report.overall_macro_f1 > report.overall_macro_f1:
                    current_config = self._update_config_thresholds(
                        current_config, opt_conf, opt_iou
                    )
                    report = opt_report

            # Phase 2: Multimodal LLM Prompt Refinement
            if (
                tuner_config.optimize_prompts
                and report.overall_macro_f1 < tuner_config.target_f1_score
                and llm_client.is_configured
            ):
                try:
                    prompt_diffs, reasoning_summary = llm_client.refine_prompts_with_vision(
                        active_classes, report
                    )
                    if prompt_diffs:
                        # Apply new prompts
                        for cls_item in active_classes:
                            if cls_item.name in prompt_diffs:
                                cls_item.prompt = prompt_diffs[cls_item.name]

                        current_config = self._update_config_classes(current_config, active_classes)

                        # Re-run inference with improved prompts
                        predictions = self._infer_samples(sample_images, current_config)
                        report = self.evaluator.evaluate(predictions, valid_gt, extract_crops=True)

                        # Re-tune thresholds with the new prompt predictions
                        if tuner_config.optimize_thresholds:
                            opt_conf, opt_iou, opt_report = (
                                self.parametric_solver.optimize_thresholds(
                                    predictions,
                                    valid_gt,
                                    initial_conf=current_config.confidence_threshold,
                                    initial_iou=current_config.box_iou_threshold,
                                )
                            )
                            current_config = self._update_config_thresholds(
                                current_config, opt_conf, opt_iou
                            )
                            report = opt_report
                except Exception as err:
                    LOGGER.warning(
                        "LLM prompt optimization step failed on iteration %d: %s",
                        iter_idx,
                        err,
                    )
                    reasoning_summary = f"LLM prompt refinement skipped: {err}"

            # Track best configuration
            if report.overall_macro_f1 > best_f1:
                best_f1 = report.overall_macro_f1
                best_config = current_config

            diag_text = (
                "; ".join(report.semantic_diagnostics[:2])
                or "Optimization step completed"
            )
            reason_text = (
                reasoning_summary or "Threshold and prompt parameters calibrated"
            )
            iter_record = TunerIteration(
                iteration_index=iter_idx,
                f1_score=report.overall_macro_f1,
                precision=report.overall_precision,
                recall=report.overall_recall,
                prompt_updates=prompt_diffs or {c.name: c.prompt for c in active_classes},
                confidence_threshold=current_config.confidence_threshold,
                iou_threshold=current_config.box_iou_threshold,
                diagnostics=diag_text,
                llm_reasoning=reason_text,
                elapsed_seconds=time.perf_counter() - iter_start,
            )
            iterations.append(iter_record)
            if progress_callback:
                progress_callback(iter_record)

            # Stopping Condition 1: Target reached
            if report.overall_macro_f1 >= tuner_config.target_f1_score:
                LOGGER.info(
                    "Target F1 (%0.2f >= %0.2f) reached at iteration %d",
                    report.overall_macro_f1,
                    tuner_config.target_f1_score,
                    iter_idx,
                )
                break

        target_reached = best_f1 >= tuner_config.target_f1_score
        init_pct = int(round(initial_f1 * 100))
        best_pct = int(round(best_f1 * 100))
        summary_msg = (
            f"Tuning finished: F1 improved from {init_pct}% to {best_pct}% "
            f"in {len(iterations) - 1} iterations."
        )

        return TunerResult(
            initial_config=initial_config,
            final_config=best_config,
            initial_f1=initial_f1,
            final_f1=best_f1,
            target_reached=target_reached,
            iterations=iterations,
            total_elapsed_seconds=time.perf_counter() - start_time,
            summary=summary_msg,
        )

    def _infer_samples(
        self, sample_images: list[Path], config: AutoLabelConfig
    ) -> dict[Path, AutoLabelResult]:
        """Run batch preview inference across sample images."""
        results: dict[Path, AutoLabelResult] = {}
        for img_path in sample_images:
            if img_path.is_file():
                results[img_path] = self.auto_label_engine.run_preview(img_path, config)
        return results

    def _update_config_thresholds(
        self, config: AutoLabelConfig, conf: float, iou: float
    ) -> AutoLabelConfig:
        """Return a copy of config with updated thresholds."""
        return AutoLabelConfig(
            mode=config.mode,
            confidence_threshold=conf,
            text_threshold=max(0.15, conf - 0.10),
            box_iou_threshold=iou,
            classes=config.classes,
            device=config.device,
            yolo_model_name=config.yolo_model_name,
            enable_grounding_dino=config.enable_grounding_dino,
            enable_yolo=config.enable_yolo,
            enable_locate_anything=config.enable_locate_anything,
            enable_florence2=config.enable_florence2,
            enable_sam2_masks=config.enable_sam2_masks,
        )

    def _update_config_classes(
        self, config: AutoLabelConfig, classes: list[AutoLabelClass]
    ) -> AutoLabelConfig:
        """Return a copy of config with updated classes."""
        return AutoLabelConfig(
            mode=config.mode,
            confidence_threshold=config.confidence_threshold,
            text_threshold=config.text_threshold,
            box_iou_threshold=config.box_iou_threshold,
            classes=classes,
            device=config.device,
            yolo_model_name=config.yolo_model_name,
            enable_grounding_dino=config.enable_grounding_dino,
            enable_yolo=config.enable_yolo,
            enable_locate_anything=config.enable_locate_anything,
            enable_florence2=config.enable_florence2,
            enable_sam2_masks=config.enable_sam2_masks,
        )
