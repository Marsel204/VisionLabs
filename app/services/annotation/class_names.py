"""Canonical vehicle class names across detectors, datasets and annotation files."""

from __future__ import annotations

_CLASS_ALIASES: dict[str, frozenset[str]] = {
    "car": frozenset({"car", "mobil", "sedan", "suv", "vehicle", "automobile"}),
    "motorcycle": frozenset({"motorcycle", "motor", "bike", "motorbike", "scooter", "moped"}),
    "bus": frozenset({"bus", "bis", "minibus", "angkot"}),
    "truck": frozenset({"truck", "truk", "pickup"}),
    "person": frozenset({"person", "pedestrian", "orang"}),
}


def normalize_class_name(raw: str) -> str:
    """Map detector- or file-specific vehicle names to the canonical dataset class names."""
    lowered = raw.lower().strip()
    for canonical, aliases in _CLASS_ALIASES.items():
        if lowered in aliases:
            return canonical
    return raw
