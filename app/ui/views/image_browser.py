"""Virtualized-friendly image path browser widget with annotation status badges."""

from __future__ import annotations

from collections import OrderedDict, deque
from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QImageReader,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QShowEvent,
)
from PySide6.QtWidgets import QListView, QListWidget, QListWidgetItem, QMenu

_BASE_IMAGE_CACHE: OrderedDict[Path, QPixmap] = OrderedDict()
_MAX_BASE_CACHE_SIZE = 2000
_ICON_CACHE: OrderedDict[tuple[Path, int], QIcon] = OrderedDict()
_MAX_ICON_CACHE_SIZE = 2000
_DEFAULT_ICON: QIcon | None = None


def _get_default_icon(target_size: int = 60) -> QIcon:
    """Singleton default placeholder icon to avoid repeated rendering on large datasets."""
    global _DEFAULT_ICON
    if _DEFAULT_ICON is not None:
        return _DEFAULT_ICON
    pixmap = QPixmap(target_size, target_size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QBrush(QColor("#20222a")))
    painter.setPen(QColor("#2c2f3b"))
    painter.drawRoundedRect(1, 1, target_size - 2, target_size - 2, 6, 6)
    painter.setPen(QColor("#697082"))
    font = QFont("sans-serif", 16)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "🖼")
    painter.end()
    _DEFAULT_ICON = QIcon(pixmap)
    return _DEFAULT_ICON


class ImageBrowser(QListWidget):
    """Paged image list that emits the selected path with modern thumbnail card presentation."""

    image_selected = Signal(Path)
    delete_requested = Signal(Path)

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
        self._items_by_path: dict[Path, QListWidgetItem] = {}
        self._rendered_paths: set[Path] = set()
        self._unrendered_queue: deque[Path] = deque()
        self._batch_timer: QTimer | None = None
        self.verticalScrollBar().valueChanged.connect(self._on_scroll_changed)

    @staticmethod
    def _get_base_thumbnail(path: Path, target_size: int = 60) -> QPixmap:
        """Retrieve or compute a base unbadged 60x60 thumbnail pixmap with LRU memory caching."""
        if path in _BASE_IMAGE_CACHE:
            _BASE_IMAGE_CACHE.move_to_end(path)
            return _BASE_IMAGE_CACHE[path]

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

        painter.end()

        while len(_BASE_IMAGE_CACHE) >= _MAX_BASE_CACHE_SIZE:
            _BASE_IMAGE_CACHE.popitem(last=False)

        _BASE_IMAGE_CACHE[path] = pixmap
        return pixmap

    @classmethod
    def _create_thumbnail_icon(cls, path: Path, count: int = 0) -> QIcon:
        """Create a thumbnail preview or sleek placeholder icon with layered memory caching."""
        cache_key = (path, count)
        if cache_key in _ICON_CACHE:
            _ICON_CACHE.move_to_end(cache_key)
            return _ICON_CACHE[cache_key]

        target_size = 60
        base = cls._get_base_thumbnail(path, target_size)
        pixmap = QPixmap(base)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if count > 0:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#059669"), 1.8))
            painter.drawRoundedRect(1, 1, target_size - 2, target_size - 2, 6, 6)

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

        icon = QIcon(pixmap)
        while len(_ICON_CACHE) >= _MAX_ICON_CACHE_SIZE:
            _ICON_CACHE.popitem(last=False)
        _ICON_CACHE[cache_key] = icon
        return icon

    def _ensure_item_thumbnail(self, item: QListWidgetItem) -> None:
        """Ensure thumbnail icon is rendered and set for the given item."""
        path_str = item.data(256)
        if not path_str:
            return
        path = Path(path_str)
        if path in self._rendered_paths:
            return
        self._rendered_paths.add(path)
        count = self._annotation_counts.get(path, 0)
        item.setIcon(self._create_thumbnail_icon(path, count))

    def _load_visible_thumbnails(self) -> None:
        """Immediately render thumbnails for all items currently visible in viewport + margin."""
        if not self._items_by_path:
            return

        viewport_rect = self.viewport().rect()
        if not viewport_rect.isValid() or viewport_rect.isEmpty():
            for idx in range(min(40, self.count())):
                item = self.item(idx)
                if item is not None:
                    self._ensure_item_thumbnail(item)
            return

        count = self.count()
        first_visible = -1
        last_visible = -1

        for row in range(count):
            rect = self.visualRect(self.model().index(row, 0))
            if rect.bottom() >= 0 and rect.top() <= viewport_rect.bottom():
                if first_visible == -1:
                    first_visible = row
                last_visible = row
            elif first_visible != -1 and rect.top() > viewport_rect.bottom():
                break

        if first_visible == -1:
            first_visible = 0
            last_visible = min(count - 1, 40)

        start_row = max(0, first_visible - 10)
        end_row = min(count, last_visible + 15)

        for row in range(start_row, end_row):
            item = self.item(row)
            if item is not None:
                self._ensure_item_thumbnail(item)

    def _start_background_loader(self) -> None:
        """Progressively render remaining unrendered images in idle batches."""
        if not self._unrendered_queue:
            return
        if self._batch_timer is None:
            self._batch_timer = QTimer(self)
            self._batch_timer.setInterval(10)
            self._batch_timer.timeout.connect(self._process_background_batch)
        if not self._batch_timer.isActive():
            self._batch_timer.start()

    def _process_background_batch(self) -> None:
        """Process a small batch of unrendered thumbnails in the background."""
        if not self._unrendered_queue:
            if self._batch_timer and self._batch_timer.isActive():
                self._batch_timer.stop()
            return

        batch_size = 20
        processed = 0
        while self._unrendered_queue and processed < batch_size:
            path = self._unrendered_queue.popleft()
            if path in self._rendered_paths:
                continue
            item = self._items_by_path.get(path)
            if item is not None:
                self._ensure_item_thumbnail(item)
                processed += 1

        if not self._unrendered_queue and self._batch_timer and self._batch_timer.isActive():
            self._batch_timer.stop()

    def _on_scroll_changed(self, _value: int) -> None:
        self._load_visible_thumbnails()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self._load_visible_thumbnails()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._load_visible_thumbnails()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._load_visible_thumbnails()

    def set_paths(
        self, paths: list[Path], annotation_counts: dict[Path, int] | None = None
    ) -> None:
        """Replace visible paths with thumbnail cards without blocking on massive datasets."""
        if self._batch_timer and self._batch_timer.isActive():
            self._batch_timer.stop()

        self._annotation_counts = annotation_counts or {}
        self._items_by_path.clear()
        self._rendered_paths.clear()
        self._unrendered_queue = deque(paths)

        self.setUpdatesEnabled(False)
        self.blockSignals(True)
        self.clear()

        default_icon = _get_default_icon()
        eager_limit = 20

        for idx, path in enumerate(paths):
            count = self._annotation_counts.get(path, 0)
            item = QListWidgetItem()
            if path in _BASE_IMAGE_CACHE or idx < eager_limit:
                item.setIcon(self._create_thumbnail_icon(path, count))
                self._rendered_paths.add(path)
            else:
                item.setIcon(default_icon)
            ann_note = f" ({count} annotations)" if count > 0 else " (unannotated)"
            item.setToolTip(f"{path.name}{ann_note}")
            item.setData(256, str(path))
            self.addItem(item)
            self._items_by_path[path] = item

        self.blockSignals(False)
        self.setUpdatesEnabled(True)

        self._load_visible_thumbnails()
        self._start_background_loader()

    def update_annotation_count(self, path: Path, count: int) -> None:
        """In-place O(1) thumbnail badge update for a single image without resetting the list."""
        if self._annotation_counts.get(path) == count:
            return
        self._annotation_counts[path] = count
        item = self._items_by_path.get(path)
        if item is not None:
            self._rendered_paths.add(path)
            item.setIcon(self._create_thumbnail_icon(path, count))
            ann_note = f" ({count} annotations)" if count > 0 else " (unannotated)"
            item.setToolTip(f"{path.name}{ann_note}")

    def remove_path(self, path: Path) -> None:
        """In-place removal of a single image from the browser without resetting the entire list."""
        item = self._items_by_path.pop(path, None)
        if item is not None:
            row = self.row(item)
            self.takeItem(row)
        self._annotation_counts.pop(path, None)
        self._rendered_paths.discard(path)
        if path in self._unrendered_queue:
            self._unrendered_queue = deque(p for p in self._unrendered_queue if p != path)

    def contextMenuEvent(self, event: Any) -> None:
        """Show context menu on right-click for the image item."""
        item = self.itemAt(event.pos())
        if item is None:
            return
        path_str = item.data(256)
        if not path_str:
            return
        path = Path(path_str)
        menu = QMenu(self)
        delete_action = menu.addAction("🗑 Delete Picture from Database")
        delete_action.triggered.connect(lambda: self.delete_requested.emit(path))
        menu.exec(event.globalPos())

    def keyPressEvent(self, event: Any) -> None:
        """Handle Delete and Backspace keyboard shortcuts on the selected image."""
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            item = self.currentItem()
            if item is not None:
                path_str = item.data(256)
                if path_str:
                    self.delete_requested.emit(Path(path_str))
                    return
        super().keyPressEvent(event)

    def _emit_selection(self) -> None:
        item = self.currentItem()
        if item is not None:
            path_str = item.data(256)
            if path_str:
                path = Path(path_str)
                self._ensure_item_thumbnail(item)
                self.image_selected.emit(path)

