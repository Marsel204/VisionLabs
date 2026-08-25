"""Auto Label service module integrating Grounding DINO, SAM 2, and Florence-2 VLM."""

from app.services.auto_label.engine import AutoLabelEngine, compute_box_iou
from app.services.auto_label.models import (
    DEFAULT_AUTO_LABEL_CLASSES,
    AutoLabelClass,
    AutoLabelConfig,
    AutoLabelDetection,
    AutoLabelPipelineMode,
    AutoLabelResult,
)

__all__ = [
    "AutoLabelClass",
    "AutoLabelConfig",
    "AutoLabelDetection",
    "AutoLabelEngine",
    "AutoLabelPipelineMode",
    "AutoLabelResult",
    "DEFAULT_AUTO_LABEL_CLASSES",
    "compute_box_iou",
]
