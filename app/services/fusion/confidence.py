"""Confidence-related fusion rules."""

from __future__ import annotations

from collections.abc import Sequence


def confidence_spread(confidences: Sequence[float]) -> float:
    """Return the difference between the highest and lowest confidence."""
    return max(confidences, default=0.0) - min(confidences, default=0.0)


def exceeds_confidence_difference(confidences: Sequence[float], threshold: float) -> bool:
    """Return whether confidence disagreement exceeds ``threshold``."""
    return confidence_spread(confidences) > threshold
