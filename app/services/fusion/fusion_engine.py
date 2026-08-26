"""Production label fusion engine."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from uuid import uuid4

from app.models.contracts import Detection as ModelDetection
from app.services.annotation.domain import AnnotationSource, BoundingBox
from app.services.fusion.fusion_models import (
    Detection,
    FusionConfig,
    FusionResult,
    FusionStatistics,
    FusionStatus,
)
from app.services.fusion.fusion_rules import decide_status
from app.services.fusion.iou import pairwise_iou

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _Group:
    members: list[ModelDetection]
    ids: set[int]
    iou: float = 0.0


class FusionEngine:
    """Compare detections from multiple model sources into unified results."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self.config.validate()

    def fuse(self, detections: Iterable[ModelDetection]) -> FusionResult:
        """Fuse an iterable of model detections and return decisions and metrics."""
        source_detections = list(detections)
        LOGGER.info("Fusion started with %d detections", len(source_detections))
        unique, duplicate_removed = self._remove_duplicates(source_detections)
        groups = self._match_groups(unique)
        output: list[Detection] = []
        ious: list[float] = []
        for group in groups:
            status = decide_status(group.members, self.config)
            winner = max(group.members, key=lambda item: item.confidence)
            matched_ids = tuple(uuid4() for _ in group.members)
            output.append(
                Detection(
                    id=matched_ids[0],
                    class_name=winner.class_name,
                    confidence=winner.confidence,
                    bbox=winner.box,
                    source_model=winner.source.value,
                    mask=winner.mask,
                    status=status,
                    accepted=status is FusionStatus.ACCEPTED,
                    review=status in {FusionStatus.NEEDS_REVIEW, FusionStatus.CONFLICT},
                    rejected=status is FusionStatus.REJECTED,
                    matched_ids=matched_ids,
                    matched_iou=group.iou or None,
                    source_models=frozenset(item.source for item in group.members),
                )
            )
            if group.iou:
                ious.append(group.iou)
            if len(group.members) > 1:
                LOGGER.debug(
                    "Objects matched: %d detections, IoU %.3f", len(group.members), group.iou
                )
            if status is FusionStatus.CONFLICT:
                LOGGER.warning("Conflict found for overlapping classes: %s", group.members)
        conflicts = sum(item.status is FusionStatus.CONFLICT for item in output)
        statistics = FusionStatistics(
            accepted=sum(item.status is FusionStatus.ACCEPTED for item in output),
            needs_review=sum(item.status is FusionStatus.NEEDS_REVIEW for item in output),
            rejected=sum(item.status is FusionStatus.REJECTED for item in output),
            conflicts=conflicts,
            duplicate_removed=duplicate_removed,
            average_iou=sum(ious) / len(ious) if ious else 0.0,
        )
        LOGGER.info("Duplicates removed: %d", duplicate_removed)
        LOGGER.info("Fusion completed: %d unified detections", len(output))
        return FusionResult(tuple(output), statistics)

    def _remove_duplicates(
        self, detections: list[ModelDetection]
    ) -> tuple[list[ModelDetection], int]:
        if not self.config.enable_duplicate_removal:
            return detections, 0
        kept: list[ModelDetection] = []
        removed = 0
        for source in sorted({item.source for item in detections}, key=str):
            source_items = sorted(
                (item for item in detections if item.source == source),
                key=lambda item: item.confidence,
                reverse=True,
            )
            if not source_items:
                continue
            boxes = [item.box for item in source_items]
            matrix = pairwise_iou(boxes, boxes)
            suppressed: set[int] = set()
            for index, item in enumerate(source_items):
                if index in suppressed:
                    continue
                kept.append(item)
                duplicates = (matrix[index] >= self.config.iou_threshold) & (
                    [other.class_name == item.class_name for other in source_items]
                )
                for duplicate_index in range(index + 1, len(source_items)):
                    if duplicates[duplicate_index]:
                        suppressed.add(duplicate_index)
                        removed += 1
        return kept, removed

    def _match_groups(self, detections: list[ModelDetection]) -> list[_Group]:
        if not detections:
            return []
        groups = [_Group([item], {index}) for index, item in enumerate(detections)]
        elem_to_group: list[_Group] = list(groups)
        all_boxes = [item.box for item in detections]
        matrix = pairwise_iou(all_boxes, all_boxes)
        candidates = sorted(
            (
                float(matrix[first, second]),
                first,
                second,
            )
            for first in range(len(detections))
            for second in range(first + 1, len(detections))
            if detections[first].source != detections[second].source
            and matrix[first, second] >= self.config.iou_threshold
        )
        for iou, first, second in reversed(candidates):
            left = elem_to_group[first]
            right = elem_to_group[second]
            if left is right:
                continue
            left_sources = {item.source for item in left.members}
            right_sources = {item.source for item in right.members}
            if not left_sources.isdisjoint(right_sources):
                continue
            left.members.extend(right.members)
            left.ids.update(right.ids)
            left.iou = max(left.iou, iou)
            for idx in right.ids:
                elem_to_group[idx] = left

        unique_groups: list[_Group] = []
        seen_group_ids: set[int] = set()
        for g in elem_to_group:
            gid = id(g)
            if gid not in seen_group_ids:
                seen_group_ids.add(gid)
                unique_groups.append(g)
        return unique_groups


def fuse_detections(
    detections: Iterable[ModelDetection],
    iou_threshold: float = 0.60,
    disagreement_threshold: float = 0.25,
) -> list[FusedDetection]:
    """Compatibility wrapper for the original fusion API."""
    source_detections = list(detections)
    result = FusionEngine(FusionConfig(iou_threshold, disagreement_threshold)).fuse(
        source_detections
    )
    return [
        FusedDetection(
            class_name=item.class_name,
            box=item.bbox,
            confidence=item.confidence,
            disagreement=item.status in {FusionStatus.NEEDS_REVIEW, FusionStatus.CONFLICT},
            supporting_sources=frozenset(item.source_models),
        )
        for item in result.detections
    ]


@dataclass(frozen=True, slots=True)
class FusedDetection:
    """Legacy fused detection shape retained for existing pipeline callers."""

    class_name: str
    box: BoundingBox
    confidence: float
    source: AnnotationSource = AnnotationSource.FUSED
    disagreement: bool = False
    supporting_sources: frozenset[AnnotationSource] = frozenset()
