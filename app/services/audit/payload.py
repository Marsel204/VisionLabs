"""JSON-ready representation of an audit report for the HTTP API."""

from __future__ import annotations

from typing import Any

from app.services.annotation.domain import BoundingBox
from app.services.audit.models import AuditConfig, AuditFinding, AuditReport


def _box_payload(box: BoundingBox, width: int, height: int) -> dict[str, float]:
    return {
        "norm_left": round(box.left, 5),
        "norm_top": round(box.top, 5),
        "norm_right": round(box.right, 5),
        "norm_bottom": round(box.bottom, 5),
        "x": round(box.left * width, 1),
        "y": round(box.top * height, 1),
        "width": round(box.width * width, 1),
        "height": round(box.height * height, 1),
    }


def _finding_payload(index: int, finding: AuditFinding, width: int, height: int) -> dict[str, Any]:
    return {
        "id": f"finding-{index}",
        "kind": str(finding.kind),
        "severity": round(finding.severity, 3),
        "message": finding.message,
        "annotation_id": finding.annotation_id,
        "class_name": finding.class_name,
        "confidence": None if finding.confidence is None else round(finding.confidence, 3),
        "occlusion": round(finding.occlusion, 3),
        "likely_occluded": finding.likely_occluded,
        "box": None if finding.box is None else _box_payload(finding.box, width, height),
    }


def report_payload(
    report: AuditReport,
    width: int,
    height: int,
    config: AuditConfig | None = None,
) -> dict[str, Any]:
    """Serialize a report, including per-box occlusion and the findings tally."""
    threshold = (config or AuditConfig()).occluded_threshold
    summary: dict[str, int] = {}
    for finding in report.findings:
        summary[str(finding.kind)] = summary.get(str(finding.kind), 0) + 1
    return {
        "findings": [
            _finding_payload(index, finding, width, height)
            for index, finding in enumerate(report.findings)
        ],
        "occlusion": [
            {
                "annotation_id": item.annotation_id,
                "fraction": round(item.fraction, 3),
                "occluded": item.fraction >= threshold,
                "occluder_ids": list(item.occluder_ids),
            }
            for item in report.occlusion
        ],
        "summary": summary,
        "occlusion_threshold": threshold,
    }
