"""Data structures and configuration models for Auto Labeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from app.services.annotation.domain import BoundingBox


class AutoLabelPipelineMode(StrEnum):
    """Supported pipeline architectures for Auto Labeling."""

    DINO_SAM2_MASKS = "sam2_dino_masks"
    DINO_BOXES = "dino_boxes"
    YOLO_SAM2_MASKS = "yolo_sam2_masks"
    YOLO_BOXES = "yolo_boxes"
    VLM_SAM2_MASKS = "vlm_sam2_masks"
    VLM_BOXES = "vlm_boxes"
    ENSEMBLE_FUSION_SAM2_MASKS = "ensemble_fusion_sam2_masks"
    ENSEMBLE_FUSION_BOXES = "ensemble_fusion_boxes"

    @property
    def display_name(self) -> str:
        """User-facing model name in UI dropdown."""
        match self:
            case AutoLabelPipelineMode.DINO_SAM2_MASKS:
                return "SAM 2 + Grounding DINO (Masks)"
            case AutoLabelPipelineMode.DINO_BOXES:
                return "Grounding DINO (Bounding Boxes)"
            case AutoLabelPipelineMode.YOLO_SAM2_MASKS:
                return "YOLO + SAM 2 (Masks)"
            case AutoLabelPipelineMode.YOLO_BOXES:
                return "YOLO (Bounding Boxes)"
            case AutoLabelPipelineMode.VLM_SAM2_MASKS:
                return "Florence-2 VLM + SAM 2 (Masks)"
            case AutoLabelPipelineMode.VLM_BOXES:
                return "Florence-2 VLM (Bounding Boxes)"
            case AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS:
                return "Multi-Model Ensemble + SAM 2 (Masks)"
            case AutoLabelPipelineMode.ENSEMBLE_FUSION_BOXES:
                return "Multi-Model Ensemble (Bounding Boxes)"

    @property
    def badge_label(self) -> str:
        """UI badge describing output type."""
        match self:
            case (
                AutoLabelPipelineMode.DINO_SAM2_MASKS
                | AutoLabelPipelineMode.YOLO_SAM2_MASKS
                | AutoLabelPipelineMode.VLM_SAM2_MASKS
                | AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS
            ):
                return "Mask labels"
            case (
                AutoLabelPipelineMode.DINO_BOXES
                | AutoLabelPipelineMode.YOLO_BOXES
                | AutoLabelPipelineMode.VLM_BOXES
                | AutoLabelPipelineMode.ENSEMBLE_FUSION_BOXES
            ):
                return "Box labels"

    @property
    def produces_masks(self) -> bool:
        """Whether this mode produces polygon segmentation masks."""
        return self in (
            AutoLabelPipelineMode.DINO_SAM2_MASKS,
            AutoLabelPipelineMode.YOLO_SAM2_MASKS,
            AutoLabelPipelineMode.VLM_SAM2_MASKS,
            AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS,
        )

    @property
    def is_ensemble(self) -> bool:
        """Whether this mode combines multiple detectors with fusion."""
        return self in (
            AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS,
            AutoLabelPipelineMode.ENSEMBLE_FUSION_BOXES,
        )

    @property
    def uses_vlm(self) -> bool:
        """Whether a Vision-Language Model (Florence-2) is used for localization."""
        return self in (
            AutoLabelPipelineMode.VLM_SAM2_MASKS,
            AutoLabelPipelineMode.VLM_BOXES,
            AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS,
            AutoLabelPipelineMode.ENSEMBLE_FUSION_BOXES,
        )

    @property
    def uses_yolo(self) -> bool:
        """Whether a YOLO model is actively engaged in this mode."""
        return self in (
            AutoLabelPipelineMode.YOLO_SAM2_MASKS,
            AutoLabelPipelineMode.YOLO_BOXES,
            AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS,
            AutoLabelPipelineMode.ENSEMBLE_FUSION_BOXES,
        )

    @property
    def uses_grounding(self) -> bool:
        """Whether Grounding DINO is actively engaged in this mode."""
        return self in (
            AutoLabelPipelineMode.DINO_SAM2_MASKS,
            AutoLabelPipelineMode.DINO_BOXES,
            AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS,
            AutoLabelPipelineMode.ENSEMBLE_FUSION_BOXES,
        )


@dataclass
class AutoLabelClass:
    """Class definition with custom visual prompt description and color."""

    name: str
    prompt: str = ""  # Custom visual description. If empty, name is used.
    color: str = "#29b6f6"
    enabled: bool = True

    @property
    def effective_prompt(self) -> str:
        """Return the effective text prompt for detection."""
        cleaned = self.prompt.strip()
        return cleaned if cleaned else self.name.strip()


DEFAULT_AUTO_LABEL_CLASSES: list[AutoLabelClass] = [
    AutoLabelClass(
        name="truck",
        prompt="large commercial delivery truck, flatbed, semi-trailer, or dump truck",
        color="#ef5350",
    ),
    AutoLabelClass(
        name="motorcycle",
        prompt="motorcycle, motorbike, scooter, or moped with rider",
        color="#ff9800",
    ),
    AutoLabelClass(
        name="car",
        prompt="passenger car, sedan, suv, coupe, taxi, or hatchback",
        color="#29b6f6",
    ),
    AutoLabelClass(
        name="bus",
        prompt="city transit bus, coach bus, minibus, or double-decker",
        color="#66bb6a",
    ),
]


@dataclass(frozen=True, slots=True)
class AutoLabelConfig:
    """Configuration options for running Auto Label inference."""

    mode: AutoLabelPipelineMode = AutoLabelPipelineMode.DINO_SAM2_MASKS
    confidence_threshold: float = 0.35
    text_threshold: float = 0.25
    box_iou_threshold: float = 0.50
    classes: list[AutoLabelClass] = field(default_factory=lambda: list(DEFAULT_AUTO_LABEL_CLASSES))
    device: str = "auto"
    yolo_model_name: str = "yolo11n.pt"
    yolo_models: list[str] = field(default_factory=lambda: ["yolo11n.pt"])
    # Multi-model detector toggles for ensemble fusion
    enable_grounding_dino: bool = True
    enable_yolo: bool = True
    enable_florence2: bool = False
    enable_sam2_masks: bool = True


@dataclass(frozen=True, slots=True)
class AutoLabelDetection:
    """A single detected object with optional polygon and mask information."""

    class_name: str
    confidence: float
    box: BoundingBox  # Normalized coordinates [0, 1]
    color: str = "#29b6f6"
    polygon_pixels: list[list[float]] = field(default_factory=list)
    polygon_normalized: list[list[float]] = field(default_factory=list)
    mask: np.ndarray | None = None

    @property
    def flattened_normalized(self) -> list[float]:
        """Return flattened normalized coordinates [x1, y1, x2, y2, ...] for YOLOv8 format."""
        return [coord for pt in self.polygon_normalized for coord in pt]


@dataclass(frozen=True, slots=True)
class AutoLabelResult:
    """Complete results for an Auto Label inference run on a single image."""

    image_path: Path
    image_width: int
    image_height: int
    detections: list[AutoLabelDetection] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def counts_by_class(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    @property
    def summary_text(self) -> str:
        """Formatted summary string (e.g. 'Found 4 objects: 2 cars, 1 truck, 1 motorcycle in 0.34s')."""
        if not self.detections:
            return f"No objects detected ({self.elapsed_seconds:.2f}s)"
        breakdown = ", ".join(
            f"{count} {cls_name}{'s' if count > 1 and not cls_name.endswith('s') else ''}"
            for cls_name, count in sorted(self.counts_by_class.items())
        )
        return f"Found {len(self.detections)} object{'s' if len(self.detections) != 1 else ''}: {breakdown} ({self.elapsed_seconds:.2f}s)"
