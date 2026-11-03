"""Compare existing annotations with a high-recall detector pass to find likely mistakes.

The audit is detector-agnostic: it consumes :class:`Prediction` lists, so any model (or a
tiled, low-threshold YOLO pass) can drive it. Occlusion is estimated geometrically and used
to relax thresholds for hidden objects, which detectors score low and often miss.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.services.annotation.domain import BoundingBox
from app.services.audit.models import (
    AuditBox,
    AuditConfig,
    AuditFinding,
    AuditReport,
    FindingKind,
    Prediction,
)
from app.services.audit.occlusion import estimate_occlusion, hidden_fraction
from app.services.fusion.iou import pairwise_iou_and_containment

_CANDIDATE_ID = "__candidate__"


def audit_annotations(
    existing: Sequence[AuditBox],
    predictions: Sequence[Prediction],
    image_width: int,
    image_height: int,
    config: AuditConfig | None = None,
) -> AuditReport:
    """Return findings ranked by severity for one image."""
    cfg = config or AuditConfig()
    preds = [p for p in predictions if p.confidence >= cfg.min_prediction_confidence]
    occlusion = estimate_occlusion(existing, depth_tolerance=cfg.depth_tolerance)
    findings: list[AuditFinding] = []

    findings.extend(_duplicate_findings(existing, occlusion, cfg))
    findings.extend(_shape_findings(existing, occlusion, image_width, image_height, cfg))

    if preds:
        iou, containment = pairwise_iou_and_containment(
            [item.box for item in existing], [item.box for item in preds]
        )
        same = np.array(
            [[_same_class(e.class_name, p.class_name) for p in preds] for e in existing],
            dtype=bool,
        ).reshape(len(existing), len(preds))
        matched_existing, matched_preds, match_findings = _match(existing, preds, iou, same, cfg)
        findings.extend(match_findings)
        findings.extend(
            _unsupported_findings(existing, occlusion, iou, matched_existing, cfg)
        )
        findings.extend(
            _missed_findings(
                existing, preds, iou, containment, same, matched_preds,
                image_width, image_height, cfg,
            )
        )

    ranked = sorted(
        (item for item in findings if item.severity >= cfg.min_severity),
        key=lambda item: item.severity,
        reverse=True,
    )
    return AuditReport(tuple(ranked), tuple(occlusion.values()))


def _same_class(first: str, second: str) -> bool:
    return first.strip().casefold() == second.strip().casefold()


def _greedy_pairs(
    iou: np.ndarray,
    same: np.ndarray,
    min_iou: float,
    free_rows: set[int],
    free_cols: set[int],
    *,
    same_class_only: bool,
) -> list[tuple[int, int]]:
    """Return one-to-one (row, column) matches, best overlap first."""
    candidates = [
        (float(iou[row, col]) + (0.05 if same[row, col] else 0.0), row, col)
        for row in free_rows
        for col in free_cols
        if iou[row, col] >= min_iou and (same[row, col] or not same_class_only)
    ]
    pairs: list[tuple[int, int]] = []
    for _score, row, col in sorted(candidates, reverse=True):
        if row in free_rows and col in free_cols:
            free_rows.discard(row)
            free_cols.discard(col)
            pairs.append((row, col))
    return pairs


def _match(
    existing: Sequence[AuditBox],
    preds: Sequence[Prediction],
    iou: np.ndarray,
    same: np.ndarray,
    cfg: AuditConfig,
) -> tuple[set[int], set[int], list[AuditFinding]]:
    free_rows = set(range(len(existing)))
    free_cols = set(range(len(preds)))
    pairs = _greedy_pairs(iou, same, cfg.match_iou, free_rows, free_cols, same_class_only=False)
    # Second pass: a same-class prediction that overlaps weakly is most likely the same
    # object drawn badly, not a different object.
    pairs += _greedy_pairs(
        iou, same, cfg.relaxed_match_iou, free_rows, free_cols, same_class_only=True
    )
    findings: list[AuditFinding] = []
    for row, col in pairs:
        box, pred, overlap = existing[row], preds[col], float(iou[row, col])
        if not same[row, col]:
            if pred.confidence >= cfg.class_mismatch_min_confidence:
                findings.append(AuditFinding(
                    FindingKind.CLASS_MISMATCH,
                    pred.confidence,
                    f"Labelled {box.class_name} but the detector sees {pred.class_name} "
                    f"({pred.confidence:.0%})",
                    annotation_id=box.id,
                    box=pred.box,
                    class_name=pred.class_name,
                    confidence=pred.confidence,
                ))
        elif overlap < cfg.realign_iou and pred.confidence >= cfg.misaligned_min_confidence:
            findings.append(AuditFinding(
                FindingKind.MISALIGNED,
                min(1.0, 1.0 - overlap),
                f"Box overlaps the detected {pred.class_name} by only {overlap:.0%}",
                annotation_id=box.id,
                box=pred.box,
                class_name=pred.class_name,
                confidence=pred.confidence,
            ))
    return set(range(len(existing))) - free_rows, set(range(len(preds))) - free_cols, findings


def _unsupported_findings(
    existing: Sequence[AuditBox],
    occlusion: dict[str, object],
    iou: np.ndarray,
    matched: set[int],
    cfg: AuditConfig,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for row, item in enumerate(existing):
        if row in matched or float(iou[row].max(initial=0.0)) >= cfg.unsupported_iou:
            continue
        hidden = occlusion[item.id].fraction  # type: ignore[attr-defined]
        severity = 0.4 if item.source == "human" else 0.6
        if item.confidence is not None and item.confidence < 0.5 and item.source != "human":
            severity += 0.2
        # Detectors legitimately miss hidden objects, so absence of support means less.
        severity *= 1.0 - hidden
        findings.append(AuditFinding(
            FindingKind.UNSUPPORTED,
            min(1.0, severity),
            "No detector evidence for this box"
            + (f" (about {hidden:.0%} hidden, so it may be fine)" if hidden >= 0.25 else ""),
            annotation_id=item.id,
            class_name=item.class_name,
            occlusion=hidden,
            likely_occluded=hidden >= cfg.occluded_threshold,
        ))
    return findings


def _missed_findings(
    existing: Sequence[AuditBox],
    preds: Sequence[Prediction],
    iou: np.ndarray,
    containment: np.ndarray,
    same: np.ndarray,
    matched_preds: set[int],
    image_width: int,
    image_height: int,
    cfg: AuditConfig,
) -> list[AuditFinding]:
    candidates: list[tuple[int, Prediction]] = []
    for col, pred in sorted(enumerate(preds), key=lambda item: -item[1].confidence):
        if col in matched_preds or not _plausible_shape(pred.box, image_width, image_height, cfg):
            continue
        if _is_redundant(existing, pred, col, iou, containment, same, cfg):
            continue
        if any(
            _same_class(pred.class_name, kept.class_name)
            and _box_iou(pred.box, kept.box) >= cfg.candidate_nms_iou
            for _, kept in candidates
        ):
            continue
        candidates.append((col, pred))

    findings: list[AuditFinding] = []
    for col, pred in candidates:
        overlaps_existing = bool((iou[:, col] > 0.0).any()) if len(existing) else False
        threshold = cfg.missed_min_confidence * (
            cfg.occluded_confidence_relief if overlaps_existing else 1.0
        )
        if pred.confidence < threshold:
            continue
        hidden = hidden_fraction(
            AuditBox(_CANDIDATE_ID, pred.class_name, pred.box), existing,
            depth_tolerance=cfg.depth_tolerance,
        ).fraction
        likely_occluded = hidden >= cfg.occluded_threshold
        findings.append(AuditFinding(
            FindingKind.POSSIBLY_MISSED,
            min(1.0, pred.confidence + (0.15 if likely_occluded else 0.0)),
            f"Possible unlabelled {pred.class_name} ({pred.confidence:.0%})"
            + (f", about {hidden:.0%} hidden behind other objects" if likely_occluded else ""),
            box=pred.box,
            class_name=pred.class_name,
            confidence=pred.confidence,
            occlusion=hidden,
            likely_occluded=likely_occluded,
        ))
    return findings


def _is_redundant(
    existing: Sequence[AuditBox],
    pred: Prediction,
    col: int,
    iou: np.ndarray,
    containment: np.ndarray,
    same: np.ndarray,
    cfg: AuditConfig,
) -> bool:
    """Return whether an unmatched prediction merely repeats an existing same-class box."""
    for row, item in enumerate(existing):
        if not same[row, col]:
            continue
        if iou[row, col] >= cfg.match_iou:
            return True
        ratio = min(item.box.area, pred.box.area) / max(item.box.area, pred.box.area)
        if (containment[row, col] >= cfg.duplicate_containment
                and ratio >= cfg.duplicate_min_area_ratio):
            return True
    return False


def _duplicate_findings(
    existing: Sequence[AuditBox],
    occlusion: dict[str, object],
    cfg: AuditConfig,
) -> list[AuditFinding]:
    if len(existing) < 2:
        return []
    iou, containment = pairwise_iou_and_containment(
        [item.box for item in existing], [item.box for item in existing]
    )
    flagged: set[int] = set()
    findings: list[AuditFinding] = []
    for first in range(len(existing)):
        for second in range(first + 1, len(existing)):
            a, b = existing[first], existing[second]
            if not _same_class(a.class_name, b.class_name):
                continue
            ratio = min(a.box.area, b.box.area) / max(a.box.area, b.box.area)
            if not (
                iou[first, second] >= cfg.duplicate_iou
                or (containment[first, second] >= cfg.duplicate_containment
                    and ratio >= cfg.duplicate_min_area_ratio)
            ):
                continue
            loser = second if (b.confidence or 1.0) <= (a.confidence or 1.0) else first
            winner = first if loser == second else second
            if loser in flagged:
                continue
            flagged.add(loser)
            item = existing[loser]
            findings.append(AuditFinding(
                FindingKind.DUPLICATE,
                0.5 + 0.5 * float(iou[first, second]),
                f"Duplicate of another {existing[winner].class_name} box "
                f"(overlap {float(iou[first, second]):.0%})",
                annotation_id=item.id,
                class_name=item.class_name,
                occlusion=occlusion[item.id].fraction,  # type: ignore[attr-defined]
            ))
    return findings


def _shape_findings(
    existing: Sequence[AuditBox],
    occlusion: dict[str, object],
    image_width: int,
    image_height: int,
    cfg: AuditConfig,
) -> list[AuditFinding]:
    return [
        AuditFinding(
            FindingKind.IMPLAUSIBLE_SHAPE,
            0.5,
            "Box is extremely small or has an extreme aspect ratio",
            annotation_id=item.id,
            class_name=item.class_name,
            occlusion=occlusion[item.id].fraction,  # type: ignore[attr-defined]
        )
        for item in existing
        if not _plausible_shape(item.box, image_width, image_height, cfg)
    ]


def _plausible_shape(box: BoundingBox, width: int, height: int, cfg: AuditConfig) -> bool:
    pixel_width = box.width * width
    pixel_height = box.height * height
    if pixel_width * pixel_height < cfg.min_pixel_area:
        return False
    return cfg.min_aspect <= pixel_width / pixel_height <= cfg.max_aspect


def _box_iou(first: BoundingBox, second: BoundingBox) -> float:
    left, top = max(first.left, second.left), max(first.top, second.top)
    right, bottom = min(first.right, second.right), min(first.bottom, second.bottom)
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.area + second.area - inter
    return inter / union if union else 0.0
