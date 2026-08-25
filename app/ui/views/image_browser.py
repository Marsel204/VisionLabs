"""Virtualized-friendly image path browser widget with annotation status badges."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QImageReader,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QListView, QListWidget, QListWidgetItem

_IMAGE_BROWSER_CACHE: dict[tuple[Path, int], QPixmap] = {}
_MAX_IMAGE_BROWSER_CACHE_SIZE = 300


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
        self._annotation_counts: dict[Path, int] = {}

    @staticmethod
    def _create_thumbnail_icon(path: Path, count: int = 0) -> QIcon:
        """Create a thumbnail preview or sleek placeholder icon with memory caching."""
        cache_key = (path, count)
        cached = _IMAGE_BROWSER_CACHE.get(cache_key)
        if cached is not None:
            return QIcon(cached)

        target_size = 60
        pixmap = QPixmap(target_size, target_size)
        pixmap.fill(Qt.GlobalColor.transparent)

        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        orig_size = reader.size()

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if orig_size.isValid() and orig_size.width() > 0 and orig_size.height() > 0:
            scale_ratio = max(target_size / orig_size.width(), target_size / orig_size.height())
            scaled_w = max(1, int(round(orig_size.width() * scale_ratio)))
            scaled_h = max(1, int(round(orig_size.height() * scale_ratio)))
            reader.setScaledSize(QSize(scaled_w, scaled_h))
            qimg = reader.read()
            if not qimg.isNull():
                reader_image = QPixmap.fromImage(qimg)
                x_off = max(0, (reader_image.width() - target_size) // 2)
                y_off = max(0, (reader_image.height() - target_size) // 2)
                cropped = reader_image.copy(x_off, y_off, target_size, target_size)

                painter.setBrush(QBrush(cropped))
                if count > 0:
                    painter.setPen(QPen(QColor("#059669"), 1.8))
                else:
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
        else:
            painter.setBrush(QBrush(QColor("#20222a")))
            painter.setPen(QColor("#2c2f3b"))
            painter.drawRoundedRect(1, 1, target_size - 2, target_size - 2, 6, 6)
            painter.setPen(QColor("#697082"))
            font = QFont("sans-serif", 16)
            painter.setFont(font)
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "🖼")

        if count > 0:
            badge_text = str(count) if count < 100 else "99+"
            badge_w = 20 if len(badge_text) > 1 else 15
            badge_rect = QRect(target_size - badge_w - 3, 3, badge_w, 13)
            painter.setBrush(QBrush(QColor("#059669")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(badge_rect, 3, 3)

            painter.setPen(QColor("#ffffff"))
            font = QFont("sans-serif", 7)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

        painter.end()

        if len(_IMAGE_BROWSER_CACHE) >= _MAX_IMAGE_BROWSER_CACHE_SIZE:
            _IMAGE_BROWSER_CACHE.clear()

        _IMAGE_BROWSER_CACHE[cache_key] = pixmap
        return QIcon(pixmap)

    def set_paths(
        self, paths: list[Path], annotation_counts: dict[Path, int] | None = None
    ) -> None:
        """Replace visible paths with thumbnail cards and optional annotation counts."""
        self._annotation_counts = annotation_counts or {}
        self.clear()
        for path in paths:
            count = self._annotation_counts.get(path, 0)
            item = QListWidgetItem()
            item.setIcon(self._create_thumbnail_icon(path, count))
            ann_note = f" ({count} annotations)" if count > 0 else " (unannotated)"
            item.setToolTip(f"{path.name}{ann_note}")
            item.setData(256, str(path))
            self.addItem(item)

    def _emit_selection(self) -> None:
        item = self.currentItem()
        if item is not None:
            self.image_selected.emit(Path(item.data(256)))
