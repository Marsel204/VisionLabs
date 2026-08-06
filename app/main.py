"""Application entry point and dependency composition root."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.configs.settings import AppSettings, load_settings
from app.core.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="AI-assisted traffic annotation application")
    parser.add_argument("--config", type=Path, help="path to a JSON configuration file")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Start the desktop application and return a process exit code."""
    try:
        args = build_parser().parse_args(argv)
        settings = load_settings(args.config)
        if args.config is None:
            active_learning_path = (
                Path(__file__).resolve().parents[1] / "configs" / "active_learning.yaml"
            )
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
    return """
    QWidget { background: #202124; color: #e8eaed; font-size: 13px; }
    QMainWindow, QDockWidget { background: #202124; }
    QDockWidget::title { background: #292a2d; padding: 8px; font-weight: 600; }
    QTreeWidget { background: #292a2d; border: 1px solid #3c4043; padding: 4px; }
    QToolBar {
        background: #17181a;
        border-bottom: 1px solid #303134;
        padding: 6px 8px;
        spacing: 8px;
    }
    QToolButton {
        background: #292a2d;
        border: 1px solid #3c4043;
        border-radius: 4px;
        padding: 6px 10px;
    }
    QToolButton:hover { background: #333438; border-color: #5f6368; }
    QToolButton:pressed { background: #3c4043; }
    QLineEdit, QDoubleSpinBox {
        background: #292a2d;
        border: 1px solid #3c4043;
        border-radius: 4px;
        padding: 5px 8px;
        selection-background-color: #394457;
    }
    QMenuBar { background: #17181a; border-bottom: 1px solid #303134; }
    QMenuBar::item { padding: 6px 10px; border-radius: 4px; }
    QMenuBar::item:selected { background: #292a2d; }
    QMenu {
        background: #202124;
        border: 1px solid #3c4043;
        padding: 6px;
    }
    QMenu::item { padding: 6px 18px; border-radius: 4px; }
    QMenu::item:selected { background: #394457; }
    QStatusBar { background: #292a2d; color: #9aa0a6; }
    #welcomeLabel { color: #9aa0a6; font-size: 18px; }
    """


if __name__ == "__main__":
    raise SystemExit(main())
