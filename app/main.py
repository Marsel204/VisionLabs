"""Application entry point and dependency composition root."""

from __future__ import annotations

import argparse
import io
import logging
import multiprocessing
import os
import sys
from pathlib import Path

# Enable expandable segments to avoid CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _apply_torch_grid_sample_patch() -> None:
    """Ensure torch.nn.functional.grid_sample aligns grid dtype to input dtype."""
    try:
        import torch.nn.functional as F

        if getattr(F.grid_sample, "_is_dtype_safe", False):
            return

        orig_grid_sample = F.grid_sample

        def _safe_grid_sample(input, grid, *args, **kwargs):  # type: ignore[no-untyped-def]
            if hasattr(input, "dtype") and hasattr(grid, "dtype") and input.dtype != grid.dtype:
                grid = grid.to(input.dtype)
            return orig_grid_sample(input, grid, *args, **kwargs)

        _safe_grid_sample._is_dtype_safe = True  # type: ignore[attr-defined]
        F.grid_sample = _safe_grid_sample
    except Exception:
        pass


_apply_torch_grid_sample_patch()


def get_base_dir() -> Path:
    """Return the application base directory, handling PyInstaller frozen binaries."""
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


_PROJECT_ROOT = get_base_dir()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure stdout and stderr exist when running as a windowed application on Windows
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

from app.configs.settings import AppSettings, load_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="VisionLab: Universal AI-assisted visual annotation application")
    parser.add_argument("--config", type=Path, help="path to a JSON configuration file")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Start the desktop application and return a process exit code."""
    multiprocessing.freeze_support()
    try:
        args = build_parser().parse_args(argv)
        settings = load_settings(args.config)
        if args.config is None:
            active_learning_path = get_base_dir() / "configs" / "active_learning.yaml"
            if active_learning_path.is_file():
                settings = AppSettings.from_active_learning_yaml(active_learning_path, settings)
        settings.ensure_directories()
        configure_logging(settings.paths.log_root, settings.log_level)

        from PySide6.QtWidgets import QApplication

        from app.ui.main_window import MainWindow

        application = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
        application.setStyleSheet(_dark_stylesheet())
        window = MainWindow(settings.fusion, settings.active_learning)
        window.show()
        return application.exec()
    except Exception:
        LOGGER.exception("application startup failed")
        return 1


def _dark_stylesheet() -> str:
    from app.ui.theme import get_dark_stylesheet

    return get_dark_stylesheet()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())

