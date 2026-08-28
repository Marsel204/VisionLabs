"""Automated unit test for visual verification and screenshot generation."""

from __future__ import annotations

import os
from pathlib import Path

from scripts.visual_verification import generate_visual_verification_screenshots


def test_visual_verification_generation(tmp_path: Path, qapp: object) -> None:
    """Verify that all visual verification screenshots are generated without error."""
    out_dir = tmp_path / "screenshots"
    screenshots = generate_visual_verification_screenshots(out_dir)

    assert len(screenshots) >= 3
    for path in screenshots:
        assert path.is_file(), f"Screenshot was not created: {path}"
        assert path.stat().st_size > 1000, f"Screenshot file is suspiciously small: {path}"
