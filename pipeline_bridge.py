"""Zero-Shot Auto-Annotation Pipeline Integration (Grounding DINO + SAM Bridge).

This module bridges Grounding DINO candidate box detection and SAM (Segment Anything Model)
segmentation into a unified, zero-shot polygon auto-annotation pipeline.

Pipeline Layers:
- Layer 1 (Text-to-Bbox): Grounding DINO detects bounding boxes given a text prompt.
- Layer 2 (Bbox-to-Mask): SAM converts each bounding box prompt into a 2D binary pixel mask.
- Layer 3 (Mask-to-Polygon): OpenCV traces binary mask perimeters into ordered sequential polygon vertices.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

LOGGER = logging.getLogger("pipeline_bridge")


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass(frozen=True, slots=True)
class BoxPixel:
    """Bounding box coordinates in absolute pixels [xmin, ymin, xmax, ymax]."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def to_coco_bbox(self) -> list[float]:
        """Convert to COCO format [xmin, ymin, width, height]."""
        return [self.xmin, self.ymin, self.width, self.height]


@dataclass(frozen=True, slots=True)
class PolygonAnnotation:
    """A single object annotation containing bounding box, binary mask, and polygon coordinates."""

    class_name: str
    class_id: int
    confidence: float
    box: BoxPixel
    mask: np.ndarray  # 2D boolean or uint8 binary mask array (H, W)
    polygon_pixels: list[list[float]]  # List of [x, y] vertices in absolute pixels
    polygon_normalized: list[list[float]]  # List of [x_norm, y_norm] vertices in [0, 1]

    @property
    def flattened_pixels(self) -> list[float]:
        """Return flattened pixel coordinates [x1, y1, x2, y2, ...] for COCO format."""
        return [coord for pt in self.polygon_pixels for coord in pt]

    @property
    def flattened_normalized(self) -> list[float]:
        """Return flattened normalized coordinates [x1, y1, x2, y2, ...] for YOLOv8 format."""
        return [coord for pt in self.polygon_normalized for coord in pt]


@dataclass(frozen=True, slots=True)
class ImageAnnotationResult:
    """Auto-annotation results for a single image."""

    image_path: Path
    image_width: int
    image_height: int
    annotations: list[PolygonAnnotation] = field(default_factory=list)


# ==============================================================================
# Layer 3: Mask-to-Polygon Post-Processor
# ==============================================================================


class MaskToPolygonProcessor:
    """Layer 3: Traces binary mask perimeters into ordered sequential polygon coordinates."""

    def __init__(
        self,
        min_contour_area: float = 10.0,
        epsilon_factor: float = 0.002,
        min_vertices: int = 3,
    ) -> None:
        """Initialize the contour post-processor.

        Args:
            min_contour_area: Minimum pixel area to consider a contour valid.
            epsilon_factor: Factor for Ramer-Douglas-Peucker polygon approximation.
                            Set to 0.0 to disable approximation and keep all perimeter points.
            min_vertices: Minimum number of vertices required for a valid polygon (default: 3).
        """
        self.min_contour_area = min_contour_area
        self.epsilon_factor = epsilon_factor
        self.min_vertices = min_vertices

    def mask_to_polygons(
        self,
        mask: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Convert a 2D binary mask into pixel and normalized polygon coordinates.

        Args:
            mask: 2D numpy array (binary 0/1 or bool or uint8).
            image_width: Width of the image in pixels.
            image_height: Height of the image in pixels.

        Returns:
            Tuple of (polygon_pixels, polygon_normalized).
            If no valid polygon is found, returns ([], []).
        """
        if mask is None or mask.size == 0 or not np.any(mask):
            return [], []

        # Ensure mask is uint8 binary image for cv2.findContours without redundant allocations
        if mask.dtype == np.uint8:
            mask_uint8 = mask if mask.flags.c_contiguous else np.ascontiguousarray(mask)
        else:
            mask_uint8 = (mask > 0).astype(np.uint8)

        # Find external contours
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return [], []

        # Select the contour with the largest area
        valid_contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) >= self.min_contour_area
        ]
        if not valid_contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) == 0:
                return [], []
        else:
            largest_contour = max(valid_contours, key=cv2.contourArea)

        # Simplify contour using Douglas-Peucker algorithm if epsilon_factor > 0
        if self.epsilon_factor > 0:
            perimeter = cv2.arcLength(largest_contour, closed=True)
            epsilon = self.epsilon_factor * perimeter
            approx = cv2.approxPolyDP(largest_contour, epsilon, closed=True)
        else:
            approx = largest_contour

        # Reshape to (N, 2)
        points = approx.reshape(-1, 2)

        if len(points) < self.min_vertices:
            points = largest_contour.reshape(-1, 2)
            if len(points) < self.min_vertices:
                return [], []

        # Extract absolute pixel coordinates
        polygon_pixels: list[list[float]] = [
            [float(pt[0]), float(pt[1])] for pt in points
        ]

        # Extract normalized coordinates clamped to [0.0, 1.0]
        polygon_normalized: list[list[float]] = [
            [
                min(max(float(pt[0]) / image_width, 0.0), 1.0),
                min(max(float(pt[1]) / image_height, 0.0), 1.0),
            ]
            for pt in points
        ]

        return polygon_pixels, polygon_normalized


# ==============================================================================
# Layer 1: Text-to-Bbox (Grounding DINO)
# ==============================================================================


def parse_text_ontology(text_ontology: str) -> list[str]:
    """Parse a comma, semicolon, or newline separated ontology string into clean class names."""
    tokens = [t.strip() for t in re.split(r"[,;\n]+", text_ontology) if t.strip()]
    if not tokens:
        tokens = [text_ontology.strip()] if text_ontology.strip() else ["object"]
    return tokens


def build_grounding_prompt(classes: list[str]) -> str:
    """Format target classes into a Grounding DINO text prompt."""
    return ". ".join(cls.rstrip(".") for cls in classes) + "."


class GroundingDinoDetector:
    """Layer 1: Grounding DINO text-to-bounding-box detector."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "auto",
        processor: Any = None,
        model: Any = None,
    ) -> None:
        self.model_id = model_id
        self._device_str = device
        self._processor = processor
        self._model = model

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        if self._device_str == "auto":
            from app.core.runtime import detect_gpu

            device = "cuda" if detect_gpu().device == "cuda" else "cpu"
        else:
            device = self._device_str

        dtype = torch.float16 if device == "cuda" else torch.float32
        LOGGER.info("Loading Grounding DINO model '%s' on %s (dtype: %s)", self.model_id, device, dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = GroundingDinoForObjectDetection.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        ).to(torch.device(device))
        self._model.eval()

    def detect(
        self,
        image: Image.Image,
        image_filename: str,
        classes: list[str],
        confidence_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> list[tuple[str, int, float, BoxPixel]]:
        """Detect candidate bounding boxes for an image given target classes.

        Args:
            image: PIL RGB image.
            image_filename: Source file name for logging edge cases.
            classes: List of target class names.
            confidence_threshold: Minimum box detection confidence score.
            text_threshold: Minimum text alignment score.

        Returns:
            List of (class_name, class_id, confidence_score, BoxPixel).
        """
        self._ensure_loaded()
        import torch

        prompt = build_grounding_prompt(classes)
        inputs = self._processor(images=image, text=prompt, return_tensors="pt")

        try:
            device = next(self._model.parameters()).device
            model_dtype = getattr(self._model, "dtype", None)
            device_inputs = {}
            for key, value in inputs.items():
                if hasattr(value, "to"):
                    if (
                        model_dtype is not None
                        and hasattr(value, "dtype")
                        and value.dtype in (torch.float32, torch.float64)
                        and model_dtype in (torch.float16, torch.bfloat16)
                    ):
                        device_inputs[key] = value.to(device=device, dtype=model_dtype)
                    else:
                        device_inputs[key] = value.to(device)
                else:
                    device_inputs[key] = value
        except (StopIteration, AttributeError):
            device_inputs = dict(inputs)

        with torch.inference_mode():
            outputs = self._model(**device_inputs)

        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        kwargs = {
            "text_threshold": text_threshold,
            "target_sizes": [(image.height, image.width)],
        }
        try:
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                threshold=confidence_threshold,
                **kwargs,
            )[0]
        except TypeError:
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                box_threshold=confidence_threshold,
                **kwargs,
            )[0]

        raw_boxes = results.get("boxes", [])
        raw_scores = results.get("scores", [])
        raw_labels = results.get("text_labels", results.get("labels", []))

        detections: list[tuple[str, int, float, BoxPixel]] = []

        for idx, (box, score) in enumerate(zip(raw_boxes, raw_scores, strict=False)):
            score_val = float(score)
            if score_val < confidence_threshold:
                continue

            raw_label = str(raw_labels[idx]) if idx < len(raw_labels) else classes[0]
            matched_class, matched_id = self._match_class(raw_label, classes)

            coords = box.tolist() if hasattr(box, "tolist") else list(box)
            xmin, ymin, xmax, ymax = (
                max(0.0, float(coords[0])),
                max(0.0, float(coords[1])),
                min(float(image.width), float(coords[2])),
                min(float(image.height), float(coords[3])),
            )

            if xmax > xmin and ymax > ymin:
                detections.append(
                    (
                        matched_class,
                        matched_id,
                        score_val,
                        BoxPixel(xmin, ymin, xmax, ymax),
                    )
                )

        if not detections:
            LOGGER.warning("Zero detections for text_prompt on %s", image_filename)

        return detections

    @staticmethod
    def _match_class(detected_label: str, classes: list[str]) -> tuple[str, int]:
        """Match detected label string against target ontology classes."""
        clean = detected_label.lower().strip(" .")
        for idx, cls in enumerate(classes):
            cls_clean = cls.lower().strip(" .")
            if cls_clean in clean or clean in cls_clean:
                return cls, idx
        return classes[0], 0


# ==============================================================================
# Layer 2: Bbox-to-Mask (SAM / SAM 2)
# ==============================================================================


class SamSegmenter:
    """Layer 2: SAM/SAM2 bounding-box-to-binary-mask segmenter."""

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-tiny",
        device: str = "auto",
        processor: Any = None,
        model: Any = None,
    ) -> None:
        self.model_id = model_id
        self._device_str = device
        self._processor = processor
        self._model = model

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        import torch

        if self._device_str == "auto":
            from app.core.runtime import detect_gpu

            device = "cuda" if detect_gpu().device == "cuda" else "cpu"
        else:
            device = self._device_str

        dtype = torch.float16 if device == "cuda" else torch.float32
        LOGGER.info("Loading SAM model '%s' on %s (dtype: %s)", self.model_id, device, dtype)

        if "sam2" in self.model_id.lower():
            from transformers import Sam2Model, Sam2Processor

            self._processor = Sam2Processor.from_pretrained(self.model_id)
            self._model = Sam2Model.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            ).to(torch.device(device))
        else:
            from transformers import SamModel, SamProcessor

            self._processor = SamProcessor.from_pretrained(self.model_id)
            self._model = SamModel.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            ).to(torch.device(device))

        self._model.eval()

    def segment_box(self, image: Image.Image, box: BoxPixel) -> np.ndarray:
        """Run SAM on a single bounding box prompt to produce a 2D binary mask.

        Processing each box independently ensures overlapping objects generate
        clean, separated polygon layers rather than merging into corrupt geometry.

        Args:
            image: PIL RGB image.
            box: Bounding box in absolute pixel coordinates.

        Returns:
            2D numpy binary array (H, W) with boolean or 0/1 values.
        """
        self._ensure_loaded()
        import torch

        pixel_box = [[[box.xmin, box.ymin, box.xmax, box.ymax]]]
        inputs = self._processor(
            images=image, input_boxes=pixel_box, return_tensors="pt"
        )

        try:
            device = next(self._model.parameters()).device
            model_dtype = getattr(self._model, "dtype", None)
            device_inputs = {}
            for key, value in inputs.items():
                if hasattr(value, "to"):
                    if (
                        model_dtype is not None
                        and hasattr(value, "dtype")
                        and value.dtype in (torch.float32, torch.float64)
                        and model_dtype in (torch.float16, torch.bfloat16)
                    ):
                        device_inputs[key] = value.to(device=device, dtype=model_dtype)
                    else:
                        device_inputs[key] = value.to(device)
                else:
                    device_inputs[key] = value
        except (StopIteration, AttributeError):
            device_inputs = dict(inputs)

        with torch.inference_mode():
            outputs = self._model(**device_inputs, multimask_output=False)

        orig_sizes = (
            inputs["original_sizes"]
            if hasattr(inputs, "__getitem__") and "original_sizes" in inputs
            else [(image.height, image.width)]
        )
        pred_masks = (
            outputs.pred_masks.cpu()
            if hasattr(outputs.pred_masks, "cpu")
            else outputs.pred_masks
        )

        masks = self._processor.post_process_masks(pred_masks, orig_sizes)

        mask_tensor = masks[0].squeeze()
        if hasattr(mask_tensor, "ndim") and mask_tensor.ndim > 2:
            mask_tensor = mask_tensor[0]

        if hasattr(mask_tensor, "cpu"):
            mask_tensor = mask_tensor.cpu().numpy()
        elif hasattr(mask_tensor, "numpy"):
            mask_tensor = mask_tensor.numpy()

        mask_np = np.asarray(mask_tensor)
        return (mask_np > 0).astype(np.uint8)

    def segment_boxes(self, image: Image.Image, boxes: list[BoxPixel]) -> list[np.ndarray]:
        """Run SAM on a batch of bounding box prompts in a single forward pass.

        Batched prompting encodes the image ONCE through the Vision Transformer
        backbone, providing massive speedup compared to sequential passes,
        while maintaining independent per-box prompt separation and unmerged masks.

        Args:
            image: PIL RGB image.
            boxes: List of bounding boxes in absolute pixel coordinates.

        Returns:
            List of 2D numpy binary arrays (H, W) corresponding to each box prompt.
        """
        if not boxes:
            return []
        if len(boxes) == 1:
            return [self.segment_box(image, boxes[0])]

        self._ensure_loaded()
        import torch

        prompt_boxes = [[[box.xmin, box.ymin, box.xmax, box.ymax] for box in boxes]]
        try:
            inputs = self._processor(
                images=image, input_boxes=prompt_boxes, return_tensors="pt"
            )
            try:
                device = next(self._model.parameters()).device
                model_dtype = getattr(self._model, "dtype", None)
                device_inputs = {}
                for key, value in inputs.items():
                    if hasattr(value, "to"):
                        if (
                            model_dtype is not None
                            and hasattr(value, "dtype")
                            and value.dtype in (torch.float32, torch.float64)
                            and model_dtype in (torch.float16, torch.bfloat16)
                        ):
                            device_inputs[key] = value.to(device=device, dtype=model_dtype)
                        else:
                            device_inputs[key] = value.to(device)
                    else:
                        device_inputs[key] = value
            except (StopIteration, AttributeError):
                device_inputs = dict(inputs)

            with torch.inference_mode():
                outputs = self._model(**device_inputs, multimask_output=False)

            orig_sizes = (
                inputs["original_sizes"]
                if hasattr(inputs, "__getitem__") and "original_sizes" in inputs
                else [(image.height, image.width)]
            )
            pred_masks = (
                outputs.pred_masks.cpu()
                if hasattr(outputs.pred_masks, "cpu")
                else outputs.pred_masks
            )
            masks = self._processor.post_process_masks(pred_masks, orig_sizes)

            mask_batch = masks[0]
            result_masks: list[np.ndarray] = []
            for idx in range(len(boxes)):
                mask_i = mask_batch[idx].squeeze()
                if hasattr(mask_i, "ndim") and mask_i.ndim > 2:
                    mask_i = mask_i[0]
                if hasattr(mask_i, "cpu"):
                    mask_i = mask_i.cpu().numpy()
                elif hasattr(mask_i, "numpy"):
                    mask_i = mask_i.numpy()
                mask_np = np.asarray(mask_i)
                result_masks.append((mask_np > 0).astype(np.uint8))
            return result_masks
        except Exception as err:
            LOGGER.warning("Batched SAM segmentation failed, falling back to sequential: %s", err)
            return [self.segment_box(image, box) for box in boxes]


# ==============================================================================
# End-to-End Pipeline Bridge
# ==============================================================================


class AutoAnnotationPipeline:
    """Sequential Zero-Shot Auto-Annotation Pipeline bridging Grounding DINO, VLM verification, and SAM."""

    def __init__(
        self,
        grounding_detector: GroundingDinoDetector | None = None,
        sam_segmenter: SamSegmenter | None = None,
        polygon_processor: MaskToPolygonProcessor | None = None,
        vlm_verifier: Any | None = None,
        grounding_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_model_id: str = "facebook/sam2.1-hiera-tiny",
        vlm_model_id: str = "microsoft/Florence-2-base",
        enable_vlm: bool = False,
        device: str = "auto",
    ) -> None:
        self.grounding_detector = grounding_detector or GroundingDinoDetector(
            model_id=grounding_model_id, device=device
        )
        self.sam_segmenter = sam_segmenter or SamSegmenter(
            model_id=sam_model_id, device=device
        )
        self.polygon_processor = polygon_processor or MaskToPolygonProcessor()
        self.vlm_verifier = vlm_verifier
        self.vlm_model_id = vlm_model_id
        self.enable_vlm = enable_vlm
        self._device = device

    def process_image(
        self,
        image_path: Path | str,
        text_ontology: str,
        confidence_threshold: float = 0.35,
    ) -> ImageAnnotationResult:
        """Process a single image through the complete pipeline with optional VLM verification.

        Dataflow:
        Layer 1 (Text-to-Bbox): Grounding DINO predicts candidate boxes.
        Layer 1.5 (VLM Verification, Optional): Florence-2 verifies if crop matches target class.
        Layer 2 (Bbox-to-Mask): SAM predicts binary masks for each box independently in batch.
        Layer 3 (Mask-to-Polygon): OpenCV post-processes masks into polygon vertices.

        Args:
            image_path: Path to the image file.
            text_ontology: Target classes prompt (e.g., 'helmet, vest, boots').
            confidence_threshold: Minimum detection confidence (default: 0.35).

        Returns:
            ImageAnnotationResult containing all detected polygon annotations.
        """
        path = Path(image_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path).convert("RGB")
        width, height = image.width, image.height
        classes = parse_text_ontology(text_ontology)

        # Layer 1: Grounding DINO Text-to-Bbox
        detections = self.grounding_detector.detect(
            image=image,
            image_filename=path.name,
            classes=classes,
            confidence_threshold=confidence_threshold,
        )

        # Layer 1.5: Florence-2 VLM Crop Verification
        if self.enable_vlm and detections:
            from src.vlm_helper import (
                Florence2VLM,
                crop_image,
                verify_crop_classes_batch,
            )

            if self.vlm_verifier is None:
                self.vlm_verifier = Florence2VLM(model_id=self.vlm_model_id, device=self._device)

            crops = [
                crop_image(
                    image,
                    (box.xmin, box.ymin, box.xmax, box.ymax),
                    normalized=False,
                )
                for _, _, _, box in detections
            ]
            target_classes = [class_name for class_name, _, _, _ in detections]
            matches = verify_crop_classes_batch(
                crops, target_classes, vlm=self.vlm_verifier
            )

            verified_detections = []
            for (class_name, class_id, score, box), is_matched in zip(
                detections, matches, strict=True
            ):
                if is_matched:
                    verified_detections.append((class_name, class_id, score, box))
                else:
                    LOGGER.info(
                        "VLM rejected false positive '%s' candidate box %s on %s",
                        class_name,
                        box.to_list(),
                        path.name,
                    )
            detections = verified_detections

        # Edge Case: Null detections -> skip SAM and return empty annotations gracefully
        if not detections:
            return ImageAnnotationResult(
                image_path=path,
                image_width=width,
                image_height=height,
                annotations=[],
            )

        annotations: list[PolygonAnnotation] = []

        # Layer 2: Fast batched Bbox-to-Mask (single vision encoder pass, separate masks)
        detected_boxes = [box for _, _, _, box in detections]
        try:
            masks = self.sam_segmenter.segment_boxes(image, detected_boxes)
        except Exception:
            masks = [self.sam_segmenter.segment_box(image, b) for b in detected_boxes]

        # Layer 3: Mask-to-Polygon Post-Processor
        for (class_name, class_id, score, box), mask in zip(detections, masks, strict=False):
            try:
                poly_pixels, poly_norm = self.polygon_processor.mask_to_polygons(
                    mask=mask,
                    image_width=width,
                    image_height=height,
                )

                if poly_pixels and len(poly_pixels) >= 3:
                    annotations.append(
                        PolygonAnnotation(
                            class_name=class_name,
                            class_id=class_id,
                            confidence=score,
                            box=box,
                            mask=mask,
                            polygon_pixels=poly_pixels,
                            polygon_normalized=poly_norm,
                        )
                    )
            except Exception as err:
                LOGGER.error(
                    "Failed to segment box %s on %s: %s",
                    box.to_list(),
                    path.name,
                    err,
                )

        return ImageAnnotationResult(
            image_path=path,
            image_width=width,
            image_height=height,
            annotations=annotations,
        )

    def process_source(
        self,
        target_source: Path | str,
        text_ontology: str,
        confidence_threshold: float = 0.35,
    ) -> list[ImageAnnotationResult]:
        """Process a directory or single image file through the pipeline."""
        source_path = Path(target_source).resolve()
        if source_path.is_file():
            image_files = [source_path]
        elif source_path.is_dir():
            valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
            image_files = sorted(
                p for p in source_path.iterdir() if p.suffix.lower() in valid_extensions
            )
        else:
            raise FileNotFoundError(f"Target source not found: {source_path}")

        LOGGER.info(
            "Running pipeline on %d image(s) from %s with ontology: '%s'",
            len(image_files),
            target_source,
            text_ontology,
        )

        results: list[ImageAnnotationResult] = []
        for index, img_path in enumerate(image_files, start=1):
            LOGGER.info("[%d/%d] Processing: %s", index, len(image_files), img_path.name)
            result = self.process_image(
                image_path=img_path,
                text_ontology=text_ontology,
                confidence_threshold=confidence_threshold,
            )
            results.append(result)

        return results


# ==============================================================================
# Exporters (YOLOv8-Segmentation & COCO JSON)
# ==============================================================================


def export_yolov8_segmentation(
    results: list[ImageAnnotationResult],
    output_dir: Path | str,
    classes: list[str],
    save_images: bool = True,
) -> Path:
    """Export annotations to YOLOv8-segmentation TXT format.

    Format per line in <output_dir>/labels/<image_name>.txt:
    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn> (normalized to [0, 1])

    Args:
        results: Pipeline annotation results.
        output_dir: Destination directory.
        classes: List of class names corresponding to class_id indices.
        save_images: Whether to copy image files into <output_dir>/images/.

    Returns:
        Path to the generated dataset.yaml file.
    """
    out = Path(output_dir).resolve()
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    images_dir = out / "images"
    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    for res in results:
        label_file = labels_dir / f"{res.image_path.stem}.txt"
        lines: list[str] = []

        for ann in res.annotations:
            if not ann.polygon_normalized:
                continue
            coords_str = " ".join(f"{coord:.6f}" for coord in ann.flattened_normalized)
            lines.append(f"{ann.class_id} {coords_str}")

        label_file.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

        if save_images and res.image_path.is_file():
            dest_img = images_dir / res.image_path.name
            if not dest_img.exists() or dest_img.resolve() != res.image_path.resolve():
                dest_img.write_bytes(res.image_path.read_bytes())

    # Write dataset.yaml
    yaml_file = out / "dataset.yaml"
    names_yaml = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(classes))
    yaml_content = f"""path: {out}
train: images
val: images

names:
{names_yaml}
"""
    yaml_file.write_text(yaml_content, encoding="utf-8")
    LOGGER.info("Exported YOLOv8-segmentation dataset to %s", out)
    return yaml_file


def export_coco_segmentation(
    results: list[ImageAnnotationResult],
    output_dir: Path | str,
    classes: list[str],
    save_images: bool = True,
) -> Path:
    """Export annotations to standard COCO JSON segmentation dataset.

    Args:
        results: Pipeline annotation results.
        output_dir: Destination directory.
        classes: List of target class names.
        save_images: Whether to copy image files into <output_dir>/images/.

    Returns:
        Path to the generated annotations.json file.
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    images_dir = out / "images"
    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    coco_categories = [
        {"id": idx + 1, "name": name, "supercategory": "none"}
        for idx, name in enumerate(classes)
    ]

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    ann_id_counter = 1

    for img_id, res in enumerate(results, start=1):
        coco_images.append(
            {
                "id": img_id,
                "file_name": res.image_path.name,
                "width": res.image_width,
                "height": res.image_height,
            }
        )

        if save_images and res.image_path.is_file():
            dest_img = images_dir / res.image_path.name
            if not dest_img.exists() or dest_img.resolve() != res.image_path.resolve():
                dest_img.write_bytes(res.image_path.read_bytes())

        for ann in res.annotations:
            if not ann.polygon_pixels:
                continue

            # Compute polygon area using cv2 contourArea
            pts_array = np.array(ann.polygon_pixels, dtype=np.float32).reshape(-1, 1, 2)
            poly_area = float(cv2.contourArea(pts_array))
            if poly_area <= 0:
                poly_area = float(ann.box.area)

            coco_annotations.append(
                {
                    "id": ann_id_counter,
                    "image_id": img_id,
                    "category_id": ann.class_id + 1,  # 1-indexed for COCO
                    "segmentation": [ann.flattened_pixels],
                    "area": poly_area,
                    "bbox": ann.box.to_coco_bbox(),
                    "iscrowd": 0,
                    "confidence": ann.confidence,
                }
            )
            ann_id_counter += 1

    payload = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

    json_file = out / "annotations.json"
    json_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Exported COCO segmentation dataset to %s", json_file)
    return json_file


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for the pipeline bridge."""
    parser = argparse.ArgumentParser(
        description="Zero-Shot Auto-Annotation Pipeline Integration (Grounding DINO + SAM)"
    )
    parser.add_argument(
        "--target-source",
        "-s",
        type=Path,
        required=True,
        help="Path to an image file or directory containing unannotated images.",
    )
    parser.add_argument(
        "--text-ontology",
        "-t",
        type=str,
        required=True,
        help="Comma-separated target classes to detect (e.g. 'helmet, vest, boots').",
    )
    parser.add_argument(
        "--confidence-threshold",
        "-c",
        type=float,
        default=0.35,
        help="Minimum confidence threshold for Grounding DINO detections (default: 0.35).",
    )
    parser.add_argument(
        "--export-format",
        "-f",
        type=str,
        choices=["yolo", "coco", "both"],
        default="yolo",
        help="Export format: 'yolo' (YOLOv8-seg), 'coco' (COCO JSON), or 'both' (default: yolo).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("./output_annotations"),
        help="Destination directory for exported annotations (default: ./output_annotations).",
    )
    parser.add_argument(
        "--grounding-model",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="HuggingFace model ID for Grounding DINO.",
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="facebook/sam2.1-hiera-tiny",
        help="HuggingFace model ID for SAM / SAM2.",
    )
    parser.add_argument(
        "--enable-vlm",
        action="store_true",
        help="Enable Florence-2 VLM verification to filter false-positive candidate boxes.",
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default="microsoft/Florence-2-base",
        help="HuggingFace model ID for Florence-2 VLM verifier (default: microsoft/Florence-2-base).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Inference device accelerator (default: auto).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main execution function for CLI usage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)

    try:
        pipeline = AutoAnnotationPipeline(
            grounding_model_id=args.grounding_model,
            sam_model_id=args.sam_model,
            vlm_model_id=args.vlm_model,
            enable_vlm=args.enable_vlm,
            device=args.device,
        )

        results = pipeline.process_source(
            target_source=args.target_source,
            text_ontology=args.text_ontology,
            confidence_threshold=args.confidence_threshold,
        )

        classes = parse_text_ontology(args.text_ontology)

        if args.export_format in ("yolo", "both"):
            export_yolov8_segmentation(
                results=results,
                output_dir=args.output_dir / "yolo" if args.export_format == "both" else args.output_dir,
                classes=classes,
            )

        if args.export_format in ("coco", "both"):
            export_coco_segmentation(
                results=results,
                output_dir=args.output_dir / "coco" if args.export_format == "both" else args.output_dir,
                classes=classes,
            )

        total_anns = sum(len(r.annotations) for r in results)
        LOGGER.info(
            "Auto-annotation pipeline finished successfully. Processed %d images, produced %d polygon annotations.",
            len(results),
            total_anns,
        )
        return 0
    except Exception as error:
        LOGGER.exception("Pipeline execution failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
