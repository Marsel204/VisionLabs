"""Public label fusion services."""

from app.services.fusion.fusion_engine import FusedDetection, FusionEngine, fuse_detections
from app.services.fusion.fusion_models import (
    Detection,
    FusionConfig,
    FusionResult,
    FusionStatistics,
    FusionStatus,
)
from app.services.fusion.overlap import remove_overlapping_annotations

__all__ = [
    "Detection",
    "FusedDetection",
    "FusionConfig",
    "FusionEngine",
    "FusionResult",
    "FusionStatistics",
    "FusionStatus",
    "fuse_detections",
    "remove_overlapping_annotations",
]
