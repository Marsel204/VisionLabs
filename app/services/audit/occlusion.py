"""Geometric occlusion estimation for annotation boxes.

Traffic scenes are shot from above the road, so of two overlapping vehicles the one whose
bottom edge sits lower in the frame is nearer to the camera and hides part of the other.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.services.audit.models import AuditBox, OcclusionEstimate

_GRID = 24


def hidden_fraction(
    target: AuditBox,
    others: Sequence[AuditBox],
    *,
    depth_tolerance: float = 0.03,
) -> OcclusionEstimate:
    """Return how much of ``target`` is covered by nearer boxes in ``others``.

    A box counts as an occluder only when its bottom edge is lower than the target's by more
    than ``depth_tolerance`` of the target's height, so side-by-side neighbours with similar
    ground contact do not occlude each other.
    """
    box = target.box
    cell = (np.arange(_GRID) + 0.5) / _GRID
    xs = box.left + cell * box.width
    ys = box.top + cell * box.height
    covered = np.zeros((_GRID, _GRID), dtype=bool)
    occluders: list[str] = []
    for other in others:
        if other.id == target.id:
            continue
        nearer = other.box
        if nearer.bottom <= box.bottom + depth_tolerance * box.height:
            continue
        hidden = np.outer(
            (ys >= nearer.top) & (ys <= nearer.bottom),
            (xs >= nearer.left) & (xs <= nearer.right),
        )
        if hidden.any():
            covered |= hidden
            occluders.append(other.id)
    return OcclusionEstimate(target.id, float(covered.mean()), tuple(occluders))


def estimate_occlusion(
    boxes: Sequence[AuditBox],
    *,
    depth_tolerance: float = 0.03,
) -> dict[str, OcclusionEstimate]:
    """Return the hidden fraction of every box, keyed by box id."""
    return {
        box.id: hidden_fraction(box, boxes, depth_tolerance=depth_tolerance) for box in boxes
    }
