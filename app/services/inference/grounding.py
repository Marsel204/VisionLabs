"""Grounding DINO prompt and tiled-inference helpers."""

from __future__ import annotations

import re


def prompt_variants(prompt: str) -> list[str]:
    """Return normalized individual prompts from a class-list prompt."""
    variants = [part.strip() for part in re.split(r"[,.;\n]+", prompt) if part.strip()]
    return [f"{variant}." for variant in variants] or ["motorcycle."]


def prompt_class(prompt: str) -> str | None:
    """Return the class explicitly requested by one prompt variant."""
    normalized = prompt.lower().strip(" .")
    for name in ("motorcycle", "motorbike", "scooter", "car", "bus", "truck"):
        if name in normalized:
            return "motorcycle" if name in {"motorbike", "scooter"} else name
    return None


def grounding_class(label: str) -> str | None:
    """Map Grounding DINO text phrases to supported application classes."""
    normalized = label.lower().strip(" .")
    for name in ("motorcycle", "motorbike", "scooter", "car", "bus", "truck"):
        if name in normalized:
            return "motorcycle" if name in {"motorbike", "scooter"} else name
    return None


def tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """Return stable crop starts that always include the far image edge."""
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    final = max(0, length - tile_size)
    if not positions or positions[-1] != final:
        positions.append(final)
    return positions
