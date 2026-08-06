"""Centralized application logging setup."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_root: Path, level: str = "INFO") -> None:
    """Configure console and rotating file logging exactly once per process."""
    log_root.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not any(
        getattr(handler, "name", None) == "traffic-console" for handler in root_logger.handlers
    ):
        console = logging.StreamHandler()
        console.name = "traffic-console"
        console.setFormatter(formatter)
        root_logger.addHandler(console)

    if not any(
        getattr(handler, "name", None) == "traffic-file" for handler in root_logger.handlers
    ):
        file_handler = logging.FileHandler(log_root / "traffic-annotator.log", encoding="utf-8")
        file_handler.name = "traffic-file"
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
