"""Backward-compatible imports for the label fusion service."""

from app.services.fusion.fusion_engine import FusedDetection, FusionEngine, fuse_detections
from app.services.fusion.iou import intersection_over_union

__all__ = ["FusedDetection", "FusionEngine", "fuse_detections", "intersection_over_union"]
