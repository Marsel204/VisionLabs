"""Virtualized-friendly image path browser widget."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QListView, QListWidget, QListWidgetItem


_IMAGE_BROWSER_CACHE: dict[Path, QPixmap] = {}


class ImageBrowser(QListWidget):
    """Paged image list that emits the selected path with modern thumbnail card presentation."""

    image_selected = Signal(Path)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("imageBrowser")
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setIconSize(QSize(62, 62))
        self.setGridSize(QSize(74, 74))
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setSpacing(4)
        self.setWordWrap(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.itemSelectionChanged.connect(self._emit_selection)

    @staticmethod
    def _create_thumbnail_icon(path: Path) -> QIcon:
        """Create a thumbnail preview or sleek placeholder icon with memory caching."""
        cached = _IMAGE_BROWSER_CACHE.get(path)
        if cached is not None:
            return QIcon(cached)

        target_size = 60
        pixmap = QPixmap(target_size, target_size)
        pixmap.fill(Qt.GlobalColor.transparent)

        reader_image = QPixmap(str(path))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if not reader_image.isNull():
            scaled = reader_image.scaled(
                target_size,
                target_size,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x_off = max(0, (scaled.width() - target_size) // 2)
            y_off = max(0, (scaled.height() - target_size) // 2)
            cropped = scaled.copy(x_off, y_off, target_size, target_size)

            painter.setBrush(QBrush(cropped))
            painter.setPen(QColor("#2c2f3b"))
            painter.drawRoundedRect(1, 1, target_size - 2, target_size - 2, 6, 6)
        else:
            painter.setBrush(QBrush(QColor("#20222a")))
            painter.setPen(QColor("#2c2f3b"))
            painter.drawRoundedRect(1, 1, target_size - 2, target_size - 2, 6, 6)
            painter.setPen(QColor("#697082"))
            font = QFont("sans-serif", 16)
            painter.setFont(font)
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "🖼")
        painter.end()
        _IMAGE_BROWSER_CACHE[path] = pixmap
        return QIcon(pixmap)

    def set_paths(self, paths: list[Path]) -> None:
        """Replace visible paths with thumbnail cards."""
        self.clear()
        for path in paths:
            item = QListWidgetItem()
            item.setIcon(self._create_thumbnail_icon(path))
            item.setToolTip(path.name)
            item.setData(256, str(path))
            self.addItem(item)

    def _emit_selection(self) -> None:
        item = self.currentItem()
        if item is not None:
            self.image_selected.emit(Path(item.data(256)))
