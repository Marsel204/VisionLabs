"""Core Auto Label inference engine coordinating Grounding DINO, Florence-2 VLM, and SAM 2."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.services.annotation.domain import (
    Annotation,
    AnnotationDocument,
    AnnotationSource,
    BoundingBox,
)
from app.services.auto_label.models import (
    AutoLabelClass,
    AutoLabelConfig,
    AutoLabelDetection,
    AutoLabelPipelineMode,
    AutoLabelResult,
)

LOGGER = logging.getLogger(__name__)


def compute_box_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute Intersection-over-Union between two normalized bounding boxes."""
    inter_left = max(box1.left, box2.left)
    inter_top = max(box1.top, box2.top)
    inter_right = min(box1.right, box2.right)
    inter_bottom = min(box1.bottom, box2.bottom)

    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0

    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    box1_area = max(1e-8, (box1.right - box1.left) * (box1.bottom - box1.top))
    box2_area = max(1e-8, (box2.right - box2.left) * (box2.bottom - box2.top))
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


class AutoLabelEngine:
    """Orchestrates zero-shot prompt grounded detection, VLM verification, and SAM 2 mask segmentation."""

    _grounding_detector: Any | None = None
    _sam_segmenter: Any | None = None
    _vlm_helper: Any | None = None
    _polygon_processor: Any | None = None
    _yolo_detector: Any | None = None
    _yolo_model_name: str = "yolo11n.pt"
    _device: str = "auto"

    def __init__(
        self,
        grounding_detector: Any | None = None,
        sam_segmenter: Any | None = None,
        vlm_helper: Any | None = None,
        polygon_processor: Any | None = None,
        yolo_detector: Any | None = None,
        yolo_model_name: str = "yolo11n.pt",
        device: str = "auto",
    ) -> None:
        self._grounding_detector = grounding_detector
        self._sam_segmenter = sam_segmenter
        self._vlm_helper = vlm_helper
        self._polygon_processor = polygon_processor
        self._yolo_detector = yolo_detector
        self._yolo_model_name = yolo_model_name
        self._device = device

    def _get_grounding_detector(self) -> Any:
        if self._grounding_detector is None:
            from pipeline_bridge import GroundingDinoDetector

            self._grounding_detector = GroundingDinoDetector(device=self._device)
        return self._grounding_detector

    def _get_sam_segmenter(self) -> Any:
        if self._sam_segmenter is None:
            from pipeline_bridge import SamSegmenter

            self._sam_segmenter = SamSegmenter(device=self._device)
        return self._sam_segmenter

    def _get_vlm_helper(self) -> Any:
        if self._vlm_helper is None:
            from src.vlm_helper import Florence2VLM

            self._vlm_helper = Florence2VLM(device=self._device)
        return self._vlm_helper

    def _get_polygon_processor(self) -> Any:
        if self._polygon_processor is None:
            from pipeline_bridge import MaskToPolygonProcessor

            self._polygon_processor = MaskToPolygonProcessor()
        return self._polygon_processor

    def _get_yolo_detector(self) -> Any:
        if self._yolo_detector is None:
            from ultralytics import YOLO

            self._yolo_detector = YOLO(self._yolo_model_name)
        return self._yolo_detector

    @property
    def yolo_model_name(self) -> str:
        return self._yolo_model_name

    @yolo_model_name.setter
    def yolo_model_name(self, value: str) -> None:
        self._yolo_model_name = value

    @staticmethod
    def suppress_duplicate_boxes(
        boxes_px: list[list[float]],
        classes: list[AutoLabelClass],
        scores: list[float],
        iou_threshold: float = 0.45,
        same_class_only: bool = False,
        containment_threshold: float = 0.70,
    ) -> tuple[list[list[float]], list[AutoLabelClass], list[float]]:
        """Suppress duplicate overlapping bounding boxes using IoU Non-Maximum Suppression (anti-duplication).

        Handles both same-class duplicates and cross-class containment/overlap conflicts
        by prioritizing higher-confidence detections without relying on complex VLM crop checks.
        """
        if not boxes_px:
            return [], [], []

        indices = sorted(range(len(boxes_px)), key=lambda i: scores[i], reverse=True)
        kept_indices: list[int] = []

        for i in indices:
            box_a = boxes_px[i]
            cls_a = classes[i]

            area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
            if area_a <= 0.0:
                continue

            duplicate = False
            for k in kept_indices:
                box_b = boxes_px[k]
                cls_b = classes[k]

                area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
                if area_b <= 0.0:
                    continue

                xA = max(box_a[0], box_b[0])
                yA = max(box_a[1], box_b[1])
                xB = min(box_a[2], box_b[2])
                yB = min(box_a[3], box_b[3])
                inter = max(0.0, xB - xA) * max(0.0, yB - yA)

                if inter <= 0.0:
                    continue

                union = area_a + area_b - inter
                iou = inter / union if union > 0 else 0.0
                min_area = min(area_a, area_b)
                containment = inter / min_area if min_area > 0 else 0.0

                if same_class_only:
                    if cls_a.name == cls_b.name and (
                        iou >= iou_threshold or containment >= containment_threshold
                    ):
                        duplicate = True
                        break
                else:
                    if cls_a.name == cls_b.name:
                        if iou >= iou_threshold or containment >= containment_threshold:
                            duplicate = True
                            break
                    else:
                        if iou >= iou_threshold or containment >= containment_threshold:
                            duplicate = True
                            break

            if not duplicate:
                kept_indices.append(i)

        return (
            [boxes_px[i] for i in kept_indices],
            [classes[i] for i in kept_indices],
            [scores[i] for i in kept_indices],
        )

    @staticmethod
    def build_prompt_mapping(classes: list[AutoLabelClass]) -> tuple[str, dict[str, AutoLabelClass]]:
        """Construct a Grounding DINO text prompt and token-to-class lookup map.

        Each class is formatted using its custom visual description prompt.
        """
        active_classes = [c for c in classes if c.enabled and c.name.strip()]
        if not active_classes:
            return "", {}

        prompt_parts: list[str] = []
        token_to_class: dict[str, AutoLabelClass] = {}

        for cls_item in active_classes:
            eff_prompt = cls_item.effective_prompt.strip().rstrip(".")
            prompt_parts.append(eff_prompt)
            # Map canonical name and description keywords to class item
            token_to_class[cls_item.name.lower().strip()] = cls_item
            for sub_phrase in eff_prompt.split(","):
                sub_clean = sub_phrase.strip().lower()
                if sub_clean:
                    token_to_class[sub_clean] = cls_item

        full_prompt = ". ".join(prompt_parts) + "."
        return full_prompt, token_to_class

    @staticmethod
    def match_detected_label(
        detected_label: str,
        classes: list[AutoLabelClass],
        token_to_class: dict[str, AutoLabelClass],
    ) -> AutoLabelClass | None:
        """Map a detected raw label/phrase back to the target AutoLabelClass."""
        clean = detected_label.lower().strip(" .")
        if not clean:
            return None

        # Direct token match
        if clean in token_to_class:
            return token_to_class[clean]

        # Check for phrase / substring inclusion
        for phrase, cls_obj in token_to_class.items():
            if phrase in clean or clean in phrase:
                return cls_obj

        # Fallback match by class name
        for cls_obj in classes:
            if cls_obj.name.lower() in clean or clean in cls_obj.name.lower():
                return cls_obj

        return None

    def run_preview(
        self,
        image_path: Path | str,
        config: AutoLabelConfig,
    ) -> AutoLabelResult:
        """Execute Auto Label pipeline on a single image and return structured results."""
        start_time = time.perf_counter()
        path = Path(image_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path).convert("RGB")
        width, height = image.width, image.height

        active_classes = [c for c in config.classes if c.enabled and c.name.strip()]
        if not active_classes:
            return AutoLabelResult(
                image_path=path,
                image_width=width,
                image_height=height,
                detections=[],
                elapsed_seconds=time.perf_counter() - start_time,
            )

        color_map = {c.name: c.color for c in active_classes}

        candidate_boxes_px: list[list[float]] = []
        candidate_classes: list[AutoLabelClass] = []
        candidate_scores: list[float] = []

        # ------------------------------------------------------------------
        # Phase 1: Candidate Proposal Gathering (Ensemble or Single Model)
        # ------------------------------------------------------------------
        run_dino = False
        run_yolo = False
        run_florence2 = False

        if config.mode.is_ensemble:
            run_dino = config.enable_grounding_dino
            run_yolo = config.enable_yolo
            run_florence2 = config.enable_florence2
        elif config.mode in (
            AutoLabelPipelineMode.DINO_SAM2_MASKS,
            AutoLabelPipelineMode.DINO_BOXES,
        ):
            run_dino = True
        elif config.mode in (
            AutoLabelPipelineMode.YOLO_SAM2_MASKS,
            AutoLabelPipelineMode.YOLO_BOXES,
        ):
            run_yolo = True
        elif config.mode in (
            AutoLabelPipelineMode.VLM_SAM2_MASKS,
            AutoLabelPipelineMode.VLM_BOXES,
        ):
            run_florence2 = True

        prompt_str, token_map = self.build_prompt_mapping(active_classes)

        # 1. Grounding DINO
        if run_dino:
            detector = self._get_grounding_detector()
            raw_detections = detector.detect(
                image=image,
                image_filename=path.name,
                classes=[c.effective_prompt for c in active_classes],
                confidence_threshold=config.confidence_threshold,
                text_threshold=config.text_threshold,
            )

            for raw_label, _class_id, score, box in raw_detections:
                matched_cls = self.match_detected_label(raw_label, active_classes, token_map)
                if matched_cls is None:
                    matched_cls = active_classes[0]

                candidate_boxes_px.append([box.xmin, box.ymin, box.xmax, box.ymax])
                candidate_classes.append(matched_cls)
                candidate_scores.append(score)

        # 2. YOLO
        if run_yolo:
            yolo = self._get_yolo_detector()
            from app.core.runtime import detect_gpu

            yolo_device = (
                0
                if (
                    self._device in ("cuda", "gpu")
                    or (self._device == "auto" and detect_gpu().device == "cuda")
                )
                else "cpu"
            )

            try:
                yolo_results = yolo(
                    image,
                    conf=config.confidence_threshold,
                    device=yolo_device,
                    verbose=False,
                )
            except Exception as err:
                LOGGER.warning("YOLO inference on PIL Image failed, trying filepath: %s", err)
                yolo_results = yolo(
                    str(path),
                    conf=config.confidence_threshold,
                    device=yolo_device,
                    verbose=False,
                )

            if yolo_results:
                res = yolo_results[0]
                names = getattr(yolo, "names", {}) or getattr(res, "names", {})
                boxes = getattr(res, "boxes", None)
                if boxes is not None:
                    for box in boxes:
                        cls_val = box.cls[0].item() if hasattr(box.cls[0], "item") else box.cls[0]
                        class_id = int(cls_val)
                        raw_label = str(names.get(class_id, class_id))
                        conf_val = (
                            box.conf[0].item() if hasattr(box.conf[0], "item") else box.conf[0]
                        )
                        score = float(conf_val)
                        raw_xyxy = box.xyxy[0]
                        xyxy = raw_xyxy.tolist() if hasattr(raw_xyxy, "tolist") else list(raw_xyxy)

                        matched_cls = self.match_detected_label(
                            raw_label, active_classes, token_map
                        )
                        if matched_cls is not None:
                            candidate_boxes_px.append([float(c) for c in xyxy])
                            candidate_classes.append(matched_cls)
                            candidate_scores.append(score)

        # 3. Florence-2 VLM
        if run_florence2:
            vlm = self._get_vlm_helper()
            from src.vlm_helper import map_od_label_to_class

            od_detections = vlm.detect_objects(image=image)
            for det in od_detections:
                raw_label = det.get("label", "")
                box_px = det.get("box", [])
                score = float(det.get("score", 0.85))
                if len(box_px) != 4:
                    continue

                mapped_name = map_od_label_to_class(raw_label)
                matched_cls = None
                if mapped_name:
                    for cls_obj in active_classes:
                        if cls_obj.name.lower() == mapped_name.lower():
                            matched_cls = cls_obj
                            break

                if matched_cls is None:
                    for cls_obj in active_classes:
                        if cls_obj.name.lower() in raw_label.lower():
                            matched_cls = cls_obj
                            break

                if matched_cls is not None:
                    candidate_boxes_px.append(box_px)
                    candidate_classes.append(matched_cls)
                    candidate_scores.append(score)

        # Apply Anti-Duplication Suppression (NMS / Fusion across all models)
        if candidate_boxes_px:
            candidate_boxes_px, candidate_classes, candidate_scores = self.suppress_duplicate_boxes(
                candidate_boxes_px,
                candidate_classes,
                candidate_scores,
                iou_threshold=config.box_iou_threshold,
            )

        # Early exit if no candidates survived
        if not candidate_boxes_px:
            return AutoLabelResult(
                image_path=path,
                image_width=width,
                image_height=height,
                detections=[],
                elapsed_seconds=time.perf_counter() - start_time,
            )

        # ------------------------------------------------------------------
        # Phase 2 & 3: SAM 2 Mask Segmentation / Polygon Generation
        # ------------------------------------------------------------------
        final_detections: list[AutoLabelDetection] = []

        if config.mode.produces_masks:
            from pipeline_bridge import BoxPixel

            sam = self._get_sam_segmenter()
            poly_proc = self._get_polygon_processor()

            box_pixels_list = [
                BoxPixel(b[0], b[1], b[2], b[3]) for b in candidate_boxes_px
            ]
            try:
                masks = sam.segment_boxes(image, box_pixels_list)
            except Exception as err:
                LOGGER.warning("Batched SAM failed, falling back to sequential: %s", err)
                masks = [sam.segment_box(image, b) for b in box_pixels_list]

            for box_px, cls_item, score, mask in zip(
                candidate_boxes_px, candidate_classes, candidate_scores, masks, strict=False
            ):
                xmin_n = max(0.0, min(float(box_px[0]) / width, 1.0))
                ymin_n = max(0.0, min(float(box_px[1]) / height, 1.0))
                xmax_n = max(0.0, min(float(box_px[2]) / width, 1.0))
                ymax_n = max(0.0, min(float(box_px[3]) / height, 1.0))

                poly_pixels, poly_norm = poly_proc.mask_to_polygons(
                    mask=mask, image_width=width, image_height=height
                )

                if poly_norm and len(poly_norm) >= 3:
                    # Refine bounding box from polygon bounds
                    xs = [pt[0] for pt in poly_norm]
                    ys = [pt[1] for pt in poly_norm]
                    min_x = max(0.0, min(1.0, float(min(xs))))
                    min_y = max(0.0, min(1.0, float(min(ys))))
                    max_x = max(0.0, min(1.0, float(max(xs))))
                    max_y = max(0.0, min(1.0, float(max(ys))))
                    if min_x >= max_x or min_y >= max_y:
                        if xmin_n >= xmax_n or ymin_n >= ymax_n:
                            continue
                        refined_box = BoundingBox(xmin_n, ymin_n, xmax_n, ymax_n)
                    else:
                        refined_box = BoundingBox(
                            left=min_x,
                            top=min_y,
                            right=max_x,
                            bottom=max_y,
                        )
                else:
                    if xmin_n >= xmax_n or ymin_n >= ymax_n:
                        continue
                    refined_box = BoundingBox(xmin_n, ymin_n, xmax_n, ymax_n)

                # IoU and containment deduplication check against already accepted detections
                if any(
                    compute_box_iou(det.box, refined_box) >= config.box_iou_threshold
                    or det.box.intersection_over_min(refined_box) >= 0.70
                    for det in final_detections
                ):
                    continue

                final_detections.append(
                    AutoLabelDetection(
                        class_name=cls_item.name,
                        confidence=score,
                        box=refined_box,
                        color=color_map.get(cls_item.name, cls_item.color),
                        polygon_pixels=poly_pixels,
                        polygon_normalized=poly_norm,
                        mask=mask,
                    )
                )
        else:
            # Bounding Box Only Mode
            for box_px, cls_item, score in zip(
                candidate_boxes_px, candidate_classes, candidate_scores, strict=True
            ):
                xmin_n = max(0.0, min(float(box_px[0]) / width, 1.0))
                ymin_n = max(0.0, min(float(box_px[1]) / height, 1.0))
                xmax_n = max(0.0, min(float(box_px[2]) / width, 1.0))
                ymax_n = max(0.0, min(float(box_px[3]) / height, 1.0))

                if xmin_n >= xmax_n or ymin_n >= ymax_n:
                    continue

                box_obj = BoundingBox(xmin_n, ymin_n, xmax_n, ymax_n)

                # IoU and containment deduplication check against already accepted detections
                if any(
                    compute_box_iou(det.box, box_obj) >= config.box_iou_threshold
                    or det.box.intersection_over_min(box_obj) >= 0.70
                    for det in final_detections
                ):
                    continue

                final_detections.append(
                    AutoLabelDetection(
                        class_name=cls_item.name,
                        confidence=score,
                        box=box_obj,
                        color=color_map.get(cls_item.name, cls_item.color),
                    )
                )

        elapsed = time.perf_counter() - start_time
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return AutoLabelResult(
            image_path=path,
            image_width=width,
            image_height=height,
            detections=final_detections,
            elapsed_seconds=elapsed,
        )

    def convert_to_annotations(
        self,
        result: AutoLabelResult,
        source: AnnotationSource | None = None,
    ) -> list[Annotation]:
        """Convert AutoLabelDetections into VisionLab Annotation domain entities."""
        from app.services.annotation.domain import TARGET_CLASSES

        annotations: list[Annotation] = []
        ann_source = source or AnnotationSource.SAM2

        for det in result.detections:
            if det.class_name not in TARGET_CLASSES:
                # If custom class not in strict TARGET_CLASSES, skip domain validation error
                continue
            annotations.append(
                Annotation(
                    class_name=det.class_name,
                    box=det.box,
                    confidence=det.confidence,
                    source=ann_source,
                )
            )
        return annotations

    def run_batch(
        self,
        documents: list[AnnotationDocument],
        config: AutoLabelConfig,
        progress_callback: Callable[[int, int, Path, AutoLabelResult], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> dict[Path, AnnotationDocument]:
        """Run Auto Label across a batch of documents and return updated documents."""
        updated: dict[Path, AnnotationDocument] = {}
        total = len(documents)

        if config.mode.is_ensemble:
            ann_source = (
                AnnotationSource.SAM2
                if config.mode.produces_masks
                else AnnotationSource.FUSED
            )
        elif config.mode.produces_masks:
            ann_source = AnnotationSource.SAM2
        elif config.mode in (AutoLabelPipelineMode.YOLO_BOXES,):
            ann_source = AnnotationSource.YOLO
        elif config.mode in (AutoLabelPipelineMode.VLM_BOXES,):
            ann_source = AnnotationSource.FLORENCE2
        elif config.mode in (AutoLabelPipelineMode.LOCATE_ANYTHING_BOXES,):
            ann_source = AnnotationSource.LOCATE_ANYTHING
        else:
            ann_source = AnnotationSource.GROUNDING_DINO

        ai_sources = (
            AnnotationSource.GROUNDING_DINO,
            AnnotationSource.SAM2,
            AnnotationSource.YOLO,
            AnnotationSource.FLORENCE2,
            AnnotationSource.LOCATE_ANYTHING,
            AnnotationSource.FUSED,
        )

        for index, doc in enumerate(documents, start=1):
            if is_cancelled is not None and is_cancelled():
                break

            result = self.run_preview(doc.image_path, config)

            new_annotations = self.convert_to_annotations(result, source=ann_source)

            # Preserve non-AI annotations or previous manual annotations
            preserved = [
                ann
                for ann in doc.annotations
                if ann.source not in ai_sources
            ]

            # Merge new annotations avoiding duplication
            for new_ann in new_annotations:
                if not any(
                    p.class_name == new_ann.class_name
                    and compute_box_iou(p.box, new_ann.box) >= config.box_iou_threshold
                    for p in preserved
                ):
                    preserved.append(new_ann)

            updated_doc = AnnotationDocument(
                image_path=doc.image_path,
                image_width=doc.image_width,
                image_height=doc.image_height,
                annotations=tuple(preserved),
            )
            updated[doc.image_path] = updated_doc

            if progress_callback is not None:
                progress_callback(index, total, doc.image_path, result)

        return updated
