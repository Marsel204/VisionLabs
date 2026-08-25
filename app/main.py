"""Application entry point and dependency composition root."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
    * {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    QWidget {
        background: #131417;
        color: #e2e4e9;
        font-size: 13px;
    }
    QMainWindow {
        background: #131417;
    }
    QDockWidget {
        background: #131417;
        color: #e2e4e9;
        font-weight: 600;
    }
    QDockWidget::title {
        background: #18191e;
        color: #c9ccd6;
        padding: 9px 12px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.3px;
        border-bottom: 1px solid #262830;
    }
    QScrollArea {
        background: transparent;
        border: none;
    }
    QScrollArea > QWidget > QWidget {
        background: transparent;
    }
    QTreeWidget, QListWidget {
        background: #18191e;
        border: 1px solid #262830;
        border-radius: 8px;
        padding: 6px;
        outline: none;
    }
    QTreeWidget::item, QListWidget::item {
        padding: 8px 10px;
        border-radius: 6px;
        margin: 2px 0;
        color: #e2e4e9;
        font-size: 13px;
        font-weight: 500;
    }
    QTreeWidget::item:hover, QListWidget::item:hover {
        background: #23252e;
    }
    QTreeWidget::item:selected, QListWidget::item:selected {
        background: #2b364a;
        border: 1px solid #4a72b2;
        color: #ffffff;
        font-weight: 600;
    }
    QListWidget#imageBrowser {
        background: #18191e;
        border: 1px solid #262830;
        border-radius: 8px;
        padding: 6px;
        outline: none;
    }
    QListWidget#imageBrowser::item {
        border-radius: 8px;
        margin: 3px;
        padding: 2px;
        border: 2px solid transparent;
    }
    QListWidget#imageBrowser::item:hover {
        background: #242733;
        border: 2px solid #3d4252;
    }
    QListWidget#imageBrowser::item:selected {
        background: #28364e;
        border: 2px solid #5f8dd3;
    }
    QToolButton {
        background: #1f2026;
        border: 1px solid #2c2e38;
        border-radius: 6px;
        padding: 5px 8px;
        min-height: 28px;
        color: #d8dade;
        font-size: 12px;
        font-weight: 500;
    }
    QToolButton:hover {
        background: #272932;
        border-color: #3e4250;
        color: #ffffff;
    }
    QToolButton:pressed {
        background: #313440;
        border-color: #4a4f60;
    }
    QToolButton:checked {
        background: #26344a;
        border: 1px solid #4d74b3;
        font-weight: 600;
        color: #9ec2ff;
    }
    QToolButton:disabled {
        background: #17181c;
        border-color: #212228;
        color: #4e525c;
    }
    QGroupBox {
        background: #18191e;
        border: 1px solid #262830;
        border-radius: 8px;
        margin-top: 18px;
        padding-top: 14px;
        padding-bottom: 8px;
        padding-left: 8px;
        padding-right: 8px;
        font-weight: 600;
        font-size: 12px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 6px;
        color: #9da3af;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    QLineEdit, QDoubleSpinBox, QAbstractSpinBox {
        background: #1f2026;
        border: 1px solid #2c2e38;
        border-radius: 6px;
        padding: 6px 10px;
        color: #e2e4e9;
        selection-background-color: #2b364a;
    }
    QLineEdit:focus, QDoubleSpinBox:focus {
        border-color: #5f8dd3;
    }
    QMenuBar {
        background: #131417;
        border-bottom: 1px solid #262830;
        padding: 2px 6px;
    }
    QMenuBar::item {
        background: transparent;
        padding: 6px 10px;
        border-radius: 5px;
        color: #c9ccd6;
        font-weight: 500;
    }
    QMenuBar::item:selected {
        background: #202228;
        color: #ffffff;
    }
    QMenu {
        background: #18191e;
        border: 1px solid #2c2e38;
        border-radius: 6px;
        padding: 6px;
    }
    QMenu::item {
        padding: 6px 20px;
        border-radius: 4px;
        color: #d8dade;
    }
    QMenu::item:selected {
        background: #2b364a;
        color: #ffffff;
    }
    QMenu::separator {
        height: 1px;
        background: #2c2e38;
        margin: 4px 6px;
    }
    QStatusBar {
        background: #131417;
        color: #838896;
        border-top: 1px solid #262830;
        padding: 4px 8px;
        font-size: 12px;
    }
    QScrollBar:vertical {
        background: #131417;
        width: 8px;
        margin: 0;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background: #2a2c36;
        min-height: 24px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical:hover {
        background: #3b3e4c;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
    }
    QScrollBar:horizontal {
        background: #131417;
        height: 8px;
        margin: 0;
        border-radius: 4px;
    }
    QScrollBar::handle:horizontal {
        background: #2a2c36;
        min-width: 24px;
        border-radius: 4px;
    }
    QScrollBar::handle:horizontal:hover {
        background: #3b3e4c;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0;
    }
    QSplitter::handle {
        background: #1c1d22;
        width: 2px;
        height: 2px;
    }
    QSplitter::handle:hover {
        background: #5f8dd3;
    }
    QDialog {
        background: #18191e;
        border: 1px solid #2c2e38;
        border-radius: 8px;
    }
    QDialog QLabel {
        color: #e2e4e9;
        padding: 4px 2px;
    }
    QProgressBar {
        background: #131417;
        border: 1px solid #2c2e38;
        border-radius: 4px;
        min-height: 12px;
        text-align: center;
        color: #e2e4e9;
        font-size: 11px;
    }
    QProgressBar::chunk {
        background: #4f78b8;
        border-radius: 3px;
    }
    QPushButton {
        background: #26344a;
        border: 1px solid #436294;
        border-radius: 6px;
        padding: 6px 14px;
        color: #ffffff;
        font-weight: 600;
    }
    QPushButton:hover {
        background: #30425e;
        border-color: #557cb8;
    }
    QPushButton:pressed {
        background: #394f70;
    }
    #welcomeLabel {
        color: #697082;
        font-size: 18px;
        font-weight: 500;
    }
    """


if __name__ == "__main__":
    raise SystemExit(main())
