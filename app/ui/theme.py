"""Modern design system and unified Dark Slate theme (Option 4: VisionForge AI / macOS Pro style)."""

from __future__ import annotations

# Semantic Palette Tokens
PALETTE = {
    # Surfaces
    "bg_base": "#0c0e14",
    "bg_dock": "#11141c",
    "bg_card": "#161922",
    "bg_card_elevated": "#1c212d",
    "bg_control": "#202534",
    "bg_control_hover": "#293042",
    "bg_control_active": "#323a50",
    # Borders
    "border_subtle": "#1a1e2a",
    "border_medium": "#262c3e",
    "border_highlight": "#38415a",
    "border_focus": "#4f46e5",
    # Text
    "text_primary": "#f8fafc",
    "text_secondary": "#cbd5e1",
    "text_muted": "#64748b",
    # Accents & Semantic
    "accent_primary": "#3b82f6",
    "accent_primary_dark": "#2563eb",
    "accent_purple": "#7c3aed",
    "accent_indigo": "#4f46e5",
    "accent_cyan": "#0ea5e9",
    "accent_emerald": "#10b981",
    "accent_amber": "#f59e0b",
    "accent_rose": "#f43f5e",
}

CLASS_COLORS = {
    "motorcycle": "#f59e0b",
    "car": "#0ea5e9",
    "bus": "#10b981",
    "truck": "#f43f5e",
}


def get_dark_stylesheet() -> str:
    """Return the modernized VisionForge AI (Option 4) Dark Slate stylesheet."""
    return f"""
    * {{
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "SF Pro Display", "Segoe UI", Roboto, sans-serif;
    }}
    QWidget {{
        background: {PALETTE["bg_base"]};
        color: {PALETTE["text_secondary"]};
        font-size: 12px;
        selection-background-color: {PALETTE["bg_control_active"]};
        selection-color: {PALETTE["text_primary"]};
    }}
    QMainWindow {{
        background: {PALETTE["bg_base"]};
    }}
    QToolBar#topActionBar {{
        background: {PALETTE["bg_dock"]};
        border-bottom: 1px solid {PALETTE["border_subtle"]};
        padding: 4px 12px;
        spacing: 8px;
    }}
    QToolBar#topActionBar QToolButton {{
        background: {PALETTE["bg_card"]};
        border: 1px solid {PALETTE["border_medium"]};
        border-radius: 6px;
        padding: 5px 12px;
        color: {PALETTE["text_secondary"]};
        font-weight: 600;
        font-size: 12px;
    }}
    QToolBar#topActionBar QToolButton:hover {{
        background: {PALETTE["bg_control_hover"]};
        border-color: {PALETTE["border_highlight"]};
        color: {PALETTE["text_primary"]};
    }}
    #activityRail {{
        background: #0a0c10;
        border-right: 1px solid {PALETTE["border_subtle"]};
        min-width: 42px;
        max-width: 42px;
    }}
    #activityRail QToolButton {{
        background: transparent;
        border: none;
        border-radius: 6px;
        padding: 8px 4px;
        color: #64748b;
        font-size: 16px;
    }}
    #activityRail QToolButton:hover {{
        background: {PALETTE["bg_control_hover"]};
        color: {PALETTE["text_primary"]};
    }}
    #activityRail QToolButton:checked {{
        background: #1e2434;
        color: #818cf8;
        border: 1px solid #4338ca;
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
        padding: 7px 10px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.3px;
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
        padding: 6px 10px;
        border-radius: 6px;
        margin: 2px 0;
        color: {PALETTE["text_secondary"]};
        font-size: 12px;
        font-weight: 500;
        border: 1px solid transparent;
    }}
    QTreeWidget::item:hover, QListWidget::item:hover {{
        background: {PALETTE["bg_card"]};
        border-color: {PALETTE["border_subtle"]};
    }}
    QTreeWidget::item:selected, QListWidget::item:selected {{
        background: #1c2230;
        border: 1px solid #3b82f6;
        color: {PALETTE["text_primary"]};
        font-weight: 600;
    }}
    QListWidget#imageBrowser {{
        background: {PALETTE["bg_dock"]};
        border: 1px solid {PALETTE["border_subtle"]};
        border-radius: 8px;
        padding: 4px;
        outline: none;
    }}
    QListWidget#imageBrowser::item {{
        border-radius: 6px;
        margin: 2px;
        padding: 2px;
        border: 1px solid {PALETTE["border_subtle"]};
        background: {PALETTE["bg_card"]};
    }}
    QListWidget#imageBrowser::item:hover {{
        background: {PALETTE["bg_control"]};
        border: 1px solid {PALETTE["border_highlight"]};
    }}
    QListWidget#imageBrowser::item:selected {{
        background: #1c2230;
        border: 1px solid #3b82f6;
    }}
    QToolButton {{
        background: {PALETTE["bg_control"]};
        border: 1px solid {PALETTE["border_medium"]};
        border-radius: 6px;
        padding: 5px 9px;
        min-height: 24px;
        color: {PALETTE["text_secondary"]};
        font-size: 11px;
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
        background: rgba(59, 130, 246, 0.22);
        border: 1px solid {PALETTE["accent_primary"]};
        font-weight: 600;
        color: #93c5fd;
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
        margin-top: 14px;
        padding-top: 12px;
        padding-bottom: 8px;
        padding-left: 8px;
        padding-right: 8px;
        font-weight: 600;
        font-size: 11px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 8px;
        padding: 0 4px;
        color: #94a3b8;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }}
    QLineEdit, QDoubleSpinBox, QAbstractSpinBox {{
        background: #0f1218;
        border: 1px solid {PALETTE["border_medium"]};
        border-radius: 6px;
        padding: 5px 8px;
        color: {PALETTE["text_primary"]};
        font-size: 11px;
        selection-background-color: {PALETTE["bg_control_active"]};
    }}
    QLineEdit:focus, QDoubleSpinBox:focus, QAbstractSpinBox:focus {{
        border: 1px solid {PALETTE["accent_primary"]};
        background: {PALETTE["bg_control"]};
    }}
    QMenuBar {{
        background: {PALETTE["bg_base"]};
        border-bottom: 1px solid {PALETTE["border_subtle"]};
        padding: 2px 6px;
    }}
    QMenuBar::item {{
        background: transparent;
        padding: 4px 8px;
        border-radius: 4px;
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
        padding: 4px;
    }}
    QMenu::item {{
        padding: 5px 18px;
        border-radius: 4px;
        color: {PALETTE["text_secondary"]};
    }}
    QMenu::item:selected {{
        background: {PALETTE["bg_control_active"]};
        color: {PALETTE["text_primary"]};
    }}
    QStatusBar {{
        background: {PALETTE["bg_dock"]};
        color: {PALETTE["text_muted"]};
        border-top: 1px solid {PALETTE["border_subtle"]};
        padding: 3px 8px;
        font-size: 11px;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 6px;
        margin: 0;
        border-radius: 3px;
    }}
    QScrollBar::handle:vertical {{
        background: {PALETTE["border_medium"]};
        min-height: 20px;
        border-radius: 3px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {PALETTE["border_highlight"]};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 6px;
        margin: 0;
        border-radius: 3px;
    }}
    QScrollBar::handle:horizontal {{
        background: {PALETTE["border_medium"]};
        min-width: 20px;
        border-radius: 3px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {PALETTE["border_highlight"]};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}
    QDialog {{
        background: {PALETTE["bg_dock"]};
        border: 1px solid {PALETTE["border_medium"]};
        border-radius: 10px;
    }}
    QPushButton {{
        background: {PALETTE["bg_control"]};
        border: 1px solid {PALETTE["border_medium"]};
        border-radius: 6px;
        padding: 6px 12px;
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
    """
