"""Sharpening and local contrast for blurry, low-contrast traffic frames (display only)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class EnhanceOptions:
    """Enhancement steps, applied in this order: denoise, CLAHE, unsharp mask."""

    denoise: float = 0.0
    clahe_clip: float = 0.0
    sharpen_amount: float = 0.0
    sharpen_radius: float = 1.5

    @property
    def is_identity(self) -> bool:
        return self.denoise <= 0 and self.clahe_clip <= 0 and self.sharpen_amount <= 0


PRESETS: dict[str, EnhanceOptions] = {
    "none": EnhanceOptions(),
    "sharpen": EnhanceOptions(sharpen_amount=1.0),
    "contrast": EnhanceOptions(clahe_clip=2.0),
    "rain": EnhanceOptions(denoise=3.0, clahe_clip=2.0, sharpen_amount=0.8),
}


def resolve_preset(name: str | None) -> EnhanceOptions:
    """Return the options for a preset name; unknown names raise ``ValueError``."""
    key = (name or "none").lower()
    if key not in PRESETS:
        raise ValueError(f"unknown enhancement preset: {name}")
    return PRESETS[key]


def enhance(image: Image.Image, options: EnhanceOptions) -> Image.Image:
    """Return an enhanced RGB image of the same size (the input itself for the identity)."""
    if options.is_identity:
        return image
    pixels = np.asarray(image.convert("RGB"))
    if options.denoise > 0:
        pixels = cv2.fastNlMeansDenoisingColored(
            pixels, None, options.denoise, options.denoise, 5, 15
        )
    if options.clahe_clip > 0:
        # Equalize lightness only so colours are not shifted.
        lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=options.clahe_clip, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        pixels = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if options.sharpen_amount > 0:
        blurred = cv2.GaussianBlur(pixels, (0, 0), options.sharpen_radius)
        pixels = cv2.addWeighted(
            pixels, 1.0 + options.sharpen_amount, blurred, -options.sharpen_amount, 0
        )
    return Image.fromarray(pixels)
