"""Fast evaluation engine matching predictions against Ground Truth annotations."""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from app.services.ai_tuner.models import ClassMetric, ErrorCrop, EvaluationReport
from app.services.annotation.domain import AnnotationDocument, BoundingBox
from app.services.auto_label.engine import compute_box_iou
from app.services.auto_label.models import AutoLabelResult

LOGGER = logging.getLogger(__name__)


class GroundTruthEvaluator:
    """Evaluates AutoLabel predictions against human-annotated Ground Truth documents."""

    def __init__(
        self,
        iou_threshold: float = 0.50,
        max_crops_per_class: int = 2,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_crops_per_class = max_crops_per_class

    def evaluate(
        self,
        predictions: dict[Path, AutoLabelResult] | Sequence[AutoLabelResult],
        ground_truth: dict[Path, AnnotationDocument],
        extract_crops: bool = True,
    ) -> EvaluationReport:
        """Run full evaluation across the sample set and compute class-wise and macro metrics."""
        pred_map: dict[Path, AutoLabelResult] = (
            predictions if isinstance(predictions, dict) else {r.image_path: r for r in predictions}
        )

        all_gt_classes: set[str] = set()
        all_pred_classes: set[str] = set()
        for doc in ground_truth.values():
            for ann in doc.annotations:
                all_gt_classes.add(ann.class_name)
        for res in pred_map.values():
            for det in res.detections:
                all_pred_classes.add(det.class_name)

        active_classes = sorted(all_gt_classes | all_pred_classes)
        class_metrics = {c: ClassMetric(class_name=c) for c in active_classes}

        total_gt_count = 0
        total_pred_count = 0
        total_matched_count = 0
        error_crops: list[ErrorCrop] = []

        # Image-by-image evaluation
        for img_path, gt_doc in ground_truth.items():
            gt_annotations = list(gt_doc.annotations)
            total_gt_count += len(gt_annotations)

            pred_result = pred_map.get(img_path)
            detections = list(pred_result.detections) if pred_result else []
            total_pred_count += len(detections)

            # Evaluate per-class
            matched_gt_ids: set[int] = set()
            matched_det_ids: set[int] = set()

            for c in active_classes:
                gt_class_indices = [
                    i for i, ann in enumerate(gt_annotations) if ann.class_name == c
                ]
                det_class_indices = [
                    i for i, det in enumerate(detections) if det.class_name == c
                ]

                # Pairwise matching with greedy IoU assignment
                matches: list[tuple[float, int, int]] = []
                for gi in gt_class_indices:
                    for di in det_class_indices:
                        iou = compute_box_iou(gt_annotations[gi].box, detections[di].box)
                        if iou >= self.iou_threshold:
                            matches.append((iou, gi, di))

                matches.sort(key=lambda x: x[0], reverse=True)

                class_tp = 0
                used_gt: set[int] = set()
                used_det: set[int] = set()

                for _iou, gi, di in matches:
                    if gi not in used_gt and di not in used_det:
                        used_gt.add(gi)
                        used_det.add(di)
                        matched_gt_ids.add(gi)
                        matched_det_ids.add(di)
                        class_tp += 1

                class_metrics[c].true_positives += class_tp
                total_matched_count += class_tp

                # False negatives for this class
                for gi in gt_class_indices:
                    if gi not in used_gt:
                        class_metrics[c].false_negatives += 1
                        # Check if any detection of another class overlapped (confusion)
                        confused_class = None
                        for _di, det in enumerate(detections):
                            if compute_box_iou(gt_annotations[gi].box, det.box) >= 0.40:
                                confused_class = det.class_name
                                break

                        if confused_class:
                            class_metrics[c].confused_with[confused_class] = (
                                class_metrics[c].confused_with.get(confused_class, 0) + 1
                            )

                        crop_count = len([ec for ec in error_crops if ec.class_name == c])
                        if extract_crops and crop_count < self.max_crops_per_class:
                            crop_b64 = self._crop_box_base64(img_path, gt_annotations[gi].box)
                            conf_note = f" (confused as {confused_class})" if confused_class else ""
                            error_crops.append(
                                ErrorCrop(
                                    image_path=img_path,
                                    error_type=(
                                        "class_confusion"
                                        if confused_class
                                        else "missed_ground_truth"
                                    ),
                                    class_name=c,
                                    predicted_class=confused_class,
                                    box=gt_annotations[gi].box,
                                    base64_jpeg=crop_b64,
                                    note=f"Missed {c}{conf_note}",
                                )
                            )

                # False positives for this class
                for di in det_class_indices:
                    if di not in used_det:
                        class_metrics[c].false_positives += 1
                        crop_count = len([ec for ec in error_crops if ec.class_name == c])
                        if extract_crops and crop_count < self.max_crops_per_class:
                            crop_b64 = self._crop_box_base64(img_path, detections[di].box)
                            det_conf = detections[di].confidence
                            error_crops.append(
                                ErrorCrop(
                                    image_path=img_path,
                                    error_type="false_positive",
                                    class_name=c,
                                    predicted_class=c,
                                    box=detections[di].box,
                                    base64_jpeg=crop_b64,
                                    note=f"False positive {c} (conf {det_conf:.2f})",
                                )
                            )

        # Calculate scores per class
        eval_classes = [
            c
            for c in active_classes
            if (class_metrics[c].true_positives + class_metrics[c].false_negatives) > 0
        ]
        if not eval_classes:
            eval_classes = list(active_classes)

        sum_f1 = 0.0
        sum_prec = 0.0
        sum_rec = 0.0

        for c, metric in class_metrics.items():
            metric.calculate_scores()
            if c in eval_classes:
                sum_f1 += metric.f1_score
                sum_prec += metric.precision
                sum_rec += metric.recall

        macro_f1 = (sum_f1 / len(eval_classes)) if eval_classes else 0.0
        macro_prec = (sum_prec / len(eval_classes)) if eval_classes else 0.0
        macro_rec = (sum_rec / len(eval_classes)) if eval_classes else 0.0

        diagnostics = self._build_semantic_diagnostics(class_metrics, eval_classes)

        return EvaluationReport(
            overall_macro_f1=round(macro_f1, 4),
            overall_precision=round(macro_prec, 4),
            overall_recall=round(macro_rec, 4),
            total_ground_truth=total_gt_count,
            total_detections=total_pred_count,
            total_matches=total_matched_count,
            class_metrics=class_metrics,
            error_crops=error_crops,
            semantic_diagnostics=diagnostics,
        )

    def _crop_box_base64(self, image_path: Path, box: BoundingBox) -> str | None:
        """Crop region of interest with small padding and encode to base64 JPEG."""
        try:
            if not image_path.is_file():
                return None
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                w, h = img.width, img.height

                pad_x = box.width * 0.15
                pad_y = box.height * 0.15

                left = max(0, int((box.left - pad_x) * w))
                top = max(0, int((box.top - pad_y) * h))
                right = min(w, int((box.right + pad_x) * w))
                bottom = min(h, int((box.bottom + pad_y) * h))

                if right <= left or bottom <= top:
                    return None

                cropped = img.crop((left, top, right, bottom))
                # Resize if excessively large to keep API payloads lean
                if cropped.width > 512 or cropped.height > 512:
                    cropped.thumbnail((512, 512), Image.Resampling.BILINEAR)

                buffer = io.BytesIO()
                cropped.save(buffer, format="JPEG", quality=85)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as err:
            LOGGER.warning("Could not create error crop for %s: %s", image_path, err)
            return None

    def _build_semantic_diagnostics(
        self, class_metrics: dict[str, ClassMetric], eval_classes: list[str]
    ) -> list[str]:
        """Generate human & LLM readable error diagnosis list."""
        diagnostics: list[str] = []
        for c in eval_classes:
            m = class_metrics.get(c)
            pct = int(round(m.f1_score * 100))
            if m.f1_score >= 0.85:
                diagnostics.append(f"Class '{c}' is performing well ({pct}% F1).")
            elif m.recall < 0.60 and m.precision >= 0.70:
                confusions = ", ".join(
                    f"{count} as '{other}'" for other, count in m.confused_with.items()
                )
                conf_str = f" (confused: {confusions})" if confusions else ""
                rec_pct = int(round(m.recall * 100))
                diagnostics.append(
                    f"Class '{c}' has low recall ({rec_pct}%): {m.false_negatives} missed "
                    f"objects{conf_str}. Needs broader prompt keywords or lower confidence."
                )
            elif m.precision < 0.60 and m.recall >= 0.70:
                prec_pct = int(round(m.precision * 100))
                diagnostics.append(
                    f"Class '{c}' has high false positives ({m.false_positives} false detections, "
                    f"precision {prec_pct}%). Needs stricter prompt disambiguation."
                )
            elif m.f1_score < 0.60:
                confusions = ", ".join(
                    f"{count} as '{other}'" for other, count in m.confused_with.items()
                )
                conf_str = f" (confused: {confusions})" if confusions else ""
                diagnostics.append(
                    f"Class '{c}' is struggling (F1 {pct}%): {m.false_negatives} missed, "
                    f"{m.false_positives} false positives{conf_str}."
                )
        return diagnostics
