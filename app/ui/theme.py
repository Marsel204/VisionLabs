"""Modern design system and unified Dark Slate theme for Traffic Annotator."""

from __future__ import annotations

# Semantic Palette Tokens
PALETTE = {
    # Surfaces
    "bg_base": "#0d0f14",
    "bg_dock": "#131620",
    "bg_card": "#181d2a",
    "bg_card_elevated": "#1e2434",
    "bg_control": "#23293a",
    "bg_control_hover": "#2c344a",
    "bg_control_active": "#37415d",
    # Borders
    "border_subtle": "#1f2535",
    "border_medium": "#2d354b",
    "border_highlight": "#414d6b",
    "border_focus": "#6366f1",
    # Text
    "text_primary": "#f8fafc",
    "text_secondary": "#cbd5e1",
    "text_muted": "#64748b",
    # Accents & Semantic
    "accent_primary": "#6366f1",
    "accent_primary_hover": "#4f46e5",
    "accent_cyan": "#0ea5e9",
    "accent_emerald": "#10b981",
    "accent_amber": "#f59e0b",
    "accent_rose": "#f43f5e",
    "accent_violet": "#a855f7",
}

CLASS_COLORS = {
    "motorcycle": "#f59e0b",
    "car": "#0ea5e9",
    "bus": "#10b981",
    "truck": "#f43f5e",
}


def get_dark_stylesheet() -> str:
    """Return the modernized, polished Dark Slate stylesheet for Traffic Annotator."""
    return f"""
    * {{
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}
    QWidget {{
        background: {PALETTE["bg_base"]};
        color: {PALETTE["text_secondary"]};
        font-size: 13px;
        selection-background-color: {PALETTE["bg_control_active"]};
        selection-color: {PALETTE["text_primary"]};
    }}
    QMainWindow {{
        background: {PALETTE["bg_base"]};
    }}
    QDockWidget {{
        background: {PALETTE["bg_dock"]};
        color: {PALETTE["text_primary"]};
        font-weight: 600;
        border: 1px solid {PALETTE["border_subtle"]};
    }}
    QDockWidget::title {{
        background: {PALETTE["bg_dock"]};
        color: {PALETTE["text_primary"]};
        padding: 8px 12px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        border-bottom: 1px solid {PALETTE["border_subtle"]};
    }}
    QDockWidget::close-button, QDockWidget::float-button {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 2px;
    }}
    QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
        background: {PALETTE["bg_control_hover"]};
        border: 1px solid {PALETTE["border_medium"]};
    }}
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    QScrollArea > QWidget > QWidget {{
        background: transparent;
    }}
    QTreeWidget, QListWidget {{
        background: {PALETTE["bg_dock"]};
        border: 1px solid {PALETTE["border_subtle"]};
        border-radius: 8px;
        padding: 4px;
        outline: none;
    }}
    QTreeWidget::item, QListWidget::item {{
        padding: 7px 10px;
        border-radius: 6px;
        margin: 2px 0;
        color: {PALETTE["text_secondary"]};
        font-size: 13px;
        font-weight: 500;
        border: 1px solid transparent;
    }}
    QTreeWidget::item:hover, QListWidget::item:hover {{
        background: {PALETTE["bg_control"]};
        border-color: {PALETTE["border_subtle"]};
    }}
    QTreeWidget::item:selected, QListWidget::item:selected {{
        background: {PALETTE["bg_control_active"]};
        border: 1px solid {PALETTE["accent_primary"]};
        color: {PALETTE["text_primary"]};
        font-weight: 600;
    }}
    QListWidget#imageBrowser {{
        background: {PALETTE["bg_dock"]};
        border: 1px solid {PALETTE["border_subtle"]};
        border-radius: 8px;
        padding: 6px;
        outline: none;
    }}
    QListWidget#imageBrowser::item {{
        border-radius: 8px;
        margin: 3px;
        padding: 3px;
        border: 2px solid transparent;
        background: {PALETTE["bg_card"]};
    }}
    QListWidget#imageBrowser::item:hover {{
        background: {PALETTE["bg_control"]};
        border: 2px solid {PALETTE["border_highlight"]};
    }}
    QListWidget#imageBrowser::item:selected {{
        background: {PALETTE["bg_control_active"]};
        border: 2px solid {PALETTE["accent_primary"]};
    }}
    QToolButton {{
        background: {PALETTE["bg_control"]};
        border: 1px solid {PALETTE["border_medium"]};
        border-radius: 6px;
        padding: 6px 10px;
        min-height: 26px;
        color: {PALETTE["text_secondary"]};
        font-size: 12px;
        font-weight: 500;
    }}
    QToolButton:hover {{
        background: {PALETTE["bg_control_hover"]};
        border-color: {PALETTE["border_highlight"]};
        color: {PALETTE["text_primary"]};
    }}
    QToolButton:pressed {{
        background: {PALETTE["bg_control_active"]};
        border-color: {PALETTE["accent_primary"]};
        color: {PALETTE["text_primary"]};
    }}
    QToolButton:checked {{
        background: rgba(99, 102, 241, 0.22);
        border: 1px solid {PALETTE["accent_primary"]};
        font-weight: 600;
        color: #a5b4fc;
    }}
    QToolButton:disabled {{
        background: {PALETTE["bg_dock"]};
        border-color: {PALETTE["border_subtle"]};
        color: {PALETTE["text_muted"]};
    }}
    QGroupBox {{
        background: {PALETTE["bg_card"]};
        border: 1px solid {PALETTE["border_subtle"]};
        border-radius: 8px;
        margin-top: 18px;
        padding-top: 14px;
        padding-bottom: 8px;
        padding-left: 8px;
        padding-right: 8px;
        font-weight: 600;
        font-size: 12px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 6px;
        color: {PALETTE["text_secondary"]};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}
    QLineEdit, QDoubleSpinBox, QAbstractSpinBox {{
        background: {PALETTE["bg_control"]};
        border: 1px solid {PALETTE["border_medium"]};
        border-radius: 6px;
        padding: 6px 10px;
        color: {PALETTE["text_primary"]};
        selection-background-color: {PALETTE["bg_control_active"]};
    }}
    QLineEdit:focus, QDoubleSpinBox:focus, QAbstractSpinBox:focus {{
        border: 1px solid {PALETTE["accent_primary"]};
        background: {PALETTE["bg_control_hover"]};
    }}
    QMenuBar {{
        background: {PALETTE["bg_base"]};
        border-bottom: 1px solid {PALETTE["border_subtle"]};
        padding: 2px 6px;
    }}
    QMenuBar::item {{
        background: transparent;
        padding: 5px 10px;
        border-radius: 5px;
        color: {PALETTE["text_secondary"]};
        font-weight: 500;
    }}
    QMenuBar::item:selected {{
        background: {PALETTE["bg_control_hover"]};
        color: {PALETTE["text_primary"]};
    }}
    QMenu {{
        background: {PALETTE["bg_card"]};
        border: 1px solid {PALETTE["border_medium"]};
        border-radius: 8px;
        padding: 6px;
    }}
    QMenu::item {{
        padding: 6px 20px;
        border-radius: 5px;
        color: {PALETTE["text_secondary"]};
    }}
    QMenu::item:selected {{
        background: {PALETTE["bg_control_active"]};
        color: {PALETTE["text_primary"]};
    }}
    QMenu::separator {{
        height: 1px;
        background: {PALETTE["border_subtle"]};
        margin: 4px 6px;
    }}
    QStatusBar {{
        background: {PALETTE["bg_dock"]};
        color: {PALETTE["text_muted"]};
        border-top: 1px solid {PALETTE["border_subtle"]};
        padding: 4px 8px;
        font-size: 12px;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {PALETTE["border_medium"]};
        min-height: 24px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {PALETTE["border_highlight"]};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 8px;
        margin: 0;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {PALETTE["border_medium"]};
        min-width: 24px;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {PALETTE["border_highlight"]};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}
    QSplitter::handle {{
        background: {PALETTE["border_subtle"]};
        width: 2px;
        height: 2px;
    }}
    QSplitter::handle:hover {{
        background: {PALETTE["accent_primary"]};
    }}
    QDialog {{
        background: {PALETTE["bg_dock"]};
        border: 1px solid {PALETTE["border_medium"]};
        border-radius: 10px;
    }}
    QDialog QLabel {{
        color: {PALETTE["text_secondary"]};
        padding: 2px;
    }}
    QProgressBar {{
        background: {PALETTE["bg_control"]};
        border: 1px solid {PALETTE["border_medium"]};
        border-radius: 6px;
        min-height: 12px;
        text-align: center;
        color: {PALETTE["text_primary"]};
        font-size: 11px;
        font-weight: 600;
    }}
    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {PALETTE["accent_primary"]}, stop:1 {PALETTE["accent_cyan"]});
        border-radius: 5px;
    }}
    QPushButton {{
        background: {PALETTE["bg_control"]};
        border: 1px solid {PALETTE["border_medium"]};
        border-radius: 6px;
        padding: 6px 14px;
        color: {PALETTE["text_primary"]};
        font-weight: 600;
    }}
    QPushButton:hover {{
        background: {PALETTE["bg_control_hover"]};
        border-color: {PALETTE["border_highlight"]};
    }}
    QPushButton:pressed {{
        background: {PALETTE["bg_control_active"]};
        border-color: {PALETTE["accent_primary"]};
    }}
    QPushButton:disabled {{
        background: {PALETTE["bg_dock"]};
        border-color: {PALETTE["border_subtle"]};
        color: {PALETTE["text_muted"]};
    }}
    #welcomeLabel {{
        color: {PALETTE["text_muted"]};
        font-size: 18px;
        font-weight: 500;
    }}
    """
