"""Roboflow-style Auto Label dialog with unified premium command bar, 4-sample grid preview, and approved batch application."""

from __future__ import annotations

import logging
import random
from collections import OrderedDict, deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image
from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QImageReader,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.services.annotation.domain import AnnotationDocument
from app.services.auto_label.engine import AutoLabelEngine
from app.services.auto_label.models import (
    DEFAULT_AUTO_LABEL_CLASSES,
    AutoLabelClass,
    AutoLabelConfig,
    AutoLabelDetection,
    AutoLabelPipelineMode,
    AutoLabelResult,
)

LOGGER = logging.getLogger(__name__)

CLASS_PALETTE = [
    "#ef5350",  # Red
    "#ff9800",  # Orange
    "#29b6f6",  # Light Blue
    "#66bb6a",  # Green
    "#ab47bc",  # Purple
    "#ec407a",  # Pink
    "#26a69a",  # Teal
    "#ffa726",  # Amber
    "#5c6bc0",  # Indigo
    "#78909c",  # Blue Grey
]


class ClassCardWidget(QFrame):
    """A single class item with name input, color badge, visual description prompt, and delete button."""

    deleted = Signal(object)
    changed = Signal()

    def __init__(self, class_item: AutoLabelClass, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.class_item = class_item
        self.setObjectName("ClassCard")
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 6)
        layout.setSpacing(3)

        # Top Row: Color Dot + Class Name Input + Delete Button
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)

        self.color_btn = QPushButton()
        self.color_btn.setObjectName("ClassColorBtn")
        self.color_btn.setFixedSize(14, 14)
        self.color_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.color_btn.setToolTip("Change class color")
        self.color_btn.clicked.connect(self._pick_color)
        self._update_color_btn()

        self.name_edit = QLineEdit(self.class_item.name)
        self.name_edit.setPlaceholderText("Class name")
        self.name_edit.setObjectName("ClassNameInput")
        self.name_edit.textChanged.connect(self._on_name_changed)

        self.delete_btn = QPushButton("✕")
        self.delete_btn.setFixedSize(16, 16)
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.setObjectName("ClassDeleteBtn")
        self.delete_btn.setToolTip("Remove class")
        self.delete_btn.clicked.connect(lambda: self.deleted.emit(self))

        top_row.addWidget(self.color_btn, 0)
        top_row.addWidget(self.name_edit, 1)
        top_row.addWidget(self.delete_btn, 0)

        # Bottom Row: Visual Description Prompt (clean, full width)
        self.prompt_edit = QLineEdit(self.class_item.prompt)
        self.prompt_edit.setPlaceholderText("Visual prompt (e.g. city bus, delivery truck)")
        self.prompt_edit.setObjectName("ClassPromptInput")
        self.prompt_edit.setCursorPosition(0)
        self.prompt_edit.textChanged.connect(self._on_prompt_changed)

        layout.addLayout(top_row)
        layout.addWidget(self.prompt_edit)

    def _on_name_changed(self, text: str) -> None:
        self.class_item.name = text.strip()
        self.changed.emit()

    def _on_prompt_changed(self, text: str) -> None:
        self.class_item.prompt = text.strip()
        self.changed.emit()

    def _pick_color(self) -> None:
        new_color = QColorDialog.getColor(QColor(self.class_item.color), self, "Select Class Color")
        if new_color.isValid():
            self.class_item.color = new_color.name()
            self._update_color_btn()
            self.changed.emit()

    def _update_color_btn(self) -> None:
        c = self.class_item.color
        self.color_btn.setStyleSheet(
            f"""
            QPushButton#ClassColorBtn {{
                background-color: {c};
                border: 1px solid rgba(255, 255, 255, 0.4);
                border-radius: 7px;
            }}
            QPushButton#ClassColorBtn:hover {{
                border: 2px solid #ffffff;
            }}
            """
        )


class AutoLabelPreviewCanvas(QGraphicsView):
    """Canvas for rendering preview images with overlaid polygons, masks, and boxes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor("#111827")))
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._image_item: QGraphicsPixmapItem | None = None
        self._items: list[Any] = []

    def set_image(self, image_path: Path) -> None:
        """Display raw image without any detection overlays."""
        self._scene.clear()
        self._items.clear()
        self._image_item = None

        if not image_path.is_file():
            return

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return

        self._image_item = self._scene.addPixmap(pixmap)
        self._image_item.setZValue(0)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_result(self, image_path: Path, result: AutoLabelResult) -> None:
        """Render the image and detection overlays."""
        self.set_image(image_path)
        if not self._image_item or not result or not result.detections:
            return

        pixmap = self._image_item.pixmap()
        img_w = float(pixmap.width())
        img_h = float(pixmap.height())

        font = QFont("Inter, Segoe UI, Sans-Serif", 9, QFont.Weight.Bold)

        for det in result.detections:
            color = QColor(det.color)
            fill_color = QColor(color)
            fill_color.setAlpha(85)

            # Draw polygon if available
            if det.polygon_pixels and len(det.polygon_pixels) >= 3:
                poly_f = QPolygonF([QPointF(pt[0], pt[1]) for pt in det.polygon_pixels])
                poly_item = QGraphicsPolygonItem(poly_f)
                poly_item.setBrush(QBrush(fill_color))
                poly_item.setPen(QPen(color, 2.0, Qt.PenStyle.SolidLine))
                poly_item.setZValue(1)
                self._scene.addItem(poly_item)
                self._items.append(poly_item)
            elif det.polygon_normalized and len(det.polygon_normalized) >= 3:
                poly_f = QPolygonF(
                    [QPointF(pt[0] * img_w, pt[1] * img_h) for pt in det.polygon_normalized]
                )
                poly_item = QGraphicsPolygonItem(poly_f)
                poly_item.setBrush(QBrush(fill_color))
                poly_item.setPen(QPen(color, 2.0, Qt.PenStyle.SolidLine))
                poly_item.setZValue(1)
                self._scene.addItem(poly_item)
                self._items.append(poly_item)
            else:
                # Draw bounding box
                box_rect = QRectF(
                    det.box.left * img_w,
                    det.box.top * img_h,
                    det.box.width * img_w,
                    det.box.height * img_h,
                )
                rect_item = QGraphicsRectItem(box_rect)
                rect_item.setBrush(QBrush(fill_color))
                rect_item.setPen(QPen(color, 2.0, Qt.PenStyle.SolidLine))
                rect_item.setZValue(1)
                self._scene.addItem(rect_item)
                self._items.append(rect_item)

            # Draw label tag with confidence
            label_text = f"{det.class_name} {int(round(det.confidence * 100))}%"
            tag_x = det.box.left * img_w
            tag_y = max(0.0, det.box.top * img_h - 20)

            text_item = QGraphicsSimpleTextItem(label_text)
            text_item.setFont(font)
            text_item.setBrush(QBrush(QColor("#ffffff")))
            text_bounds = text_item.boundingRect()

            bg_rect = QRectF(tag_x, tag_y, text_bounds.width() + 8, text_bounds.height() + 3)
            bg_item = QGraphicsRectItem(bg_rect)
            bg_item.setBrush(QBrush(color))
            bg_item.setPen(QPen(Qt.PenStyle.NoPen))
            bg_item.setZValue(2)
            self._scene.addItem(bg_item)
            self._items.append(bg_item)

            text_item.setPos(tag_x + 4, tag_y + 1)
            text_item.setZValue(3)
            self._scene.addItem(text_item)
            self._items.append(text_item)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if self._scene and self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class PreviewCardWidget(QFrame):
    """Card widget representing one of the 4 preview images in 2x2 grid view."""

    clicked = Signal(Path)

    def __init__(self, slot_index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.slot_index = slot_index
        self.image_path: Path | None = None
        self.setObjectName("PreviewCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Header Row
        header_row = QHBoxLayout()
        header_row.setContentsMargins(6, 4, 6, 2)
        header_row.setSpacing(6)

        self.title_label = QLabel(f"Sample {slot_index + 1}")
        self.title_label.setObjectName("CardTitle")

        self.count_badge = QLabel("Ready")
        self.count_badge.setObjectName("CardBadge")

        self.zoom_btn = QPushButton("🔍 Zoom")
        self.zoom_btn.setObjectName("CardZoomBtn")
        self.zoom_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.zoom_btn.clicked.connect(self._on_zoom_clicked)

        header_row.addWidget(self.title_label, 1)
        header_row.addWidget(self.count_badge, 0)
        header_row.addWidget(self.zoom_btn, 0)

        layout.addLayout(header_row)

        self.canvas = AutoLabelPreviewCanvas()
        self.canvas.setObjectName("CardCanvas")
        layout.addWidget(self.canvas, 1)

    def set_data(self, path: Path, result: AutoLabelResult | None) -> None:
        self.image_path = path
        short_name = path.name if len(path.name) <= 22 else f"{path.name[:10]}...{path.name[-8:]}"
        self.title_label.setText(f"Sample {self.slot_index + 1}: {short_name}")
        self.title_label.setToolTip(str(path))
        if result is not None:
            self.canvas.set_result(path, result)
            self.count_badge.setText(f"⚡ {result.count} obj")
        else:
            self.canvas.set_image(path)
            self.count_badge.setText("Ready")

    def mousePressEvent(self, event: Any) -> None:
        super().mousePressEvent(event)
        if self.image_path is not None:
            self.clicked.emit(self.image_path)

    def _on_zoom_clicked(self) -> None:
        if self.image_path is not None:
            self.clicked.emit(self.image_path)


class CherryPickDialog(QDialog):
    """Modal dialog allowing the user to search and cherry-pick up to 4 images for preview."""

    def __init__(
        self,
        all_images: list[Path],
        selected_images: list[Path],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cherry-Pick Preview Images")
        self.setFixedSize(580, 500)
        self.all_images = all_images
        self.selected_images: list[Path] = list(selected_images)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel("Select up to 4 sample images to preview:")
        title.setStyleSheet("font-size: 14px; font-weight: 700; color: #f8fafc;")
        layout.addWidget(title)

        # Search Bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Filter images by filename...")
        self.search_input.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; "
            "padding: 6px 10px; color: #f8fafc; font-size: 12px;"
        )
        self.search_input.textChanged.connect(self._filter_list)
        layout.addWidget(self.search_input)

        # Image List Widget with Checkboxes
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget { background-color: #0f172a; border: 1px solid #334155; "
            "border-radius: 8px; padding: 6px; color: #f8fafc; } "
            "QListWidget::item { padding: 6px; border-radius: 4px; } "
            "QListWidget::item:selected { background-color: #312e81; color: #ffffff; }"
        )
        self.list_widget.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget, 1)

        self._populate_list()

        # Bottom row: Count & Buttons
        bottom_row = QHBoxLayout()
        self.count_label = QLabel(f"Selected: {len(self.selected_images)} / 4")
        self.count_label.setStyleSheet("color: #a5b4fc; font-weight: 600; font-size: 12px;")

        random_btn = QPushButton("🎲 Pick 4 Random")
        random_btn.setStyleSheet(
            "background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; "
            "color: #f8fafc; padding: 6px 12px; font-weight: 600;"
        )
        random_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        random_btn.clicked.connect(self._pick_random)

        apply_btn = QPushButton("✔ Apply Selection")
        apply_btn.setStyleSheet(
            "background-color: #6366f1; border: none; border-radius: 6px; "
            "color: #ffffff; padding: 6px 16px; font-weight: 700;"
        )
        apply_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        apply_btn.clicked.connect(self.accept)

        bottom_row.addWidget(self.count_label)
        bottom_row.addStretch(1)
        bottom_row.addWidget(random_btn)
        bottom_row.addWidget(apply_btn)
        layout.addLayout(bottom_row)

    def _populate_list(self, filter_text: str = "") -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        filter_lower = filter_text.lower()
        for p in self.all_images:
            if filter_lower and filter_lower not in p.name.lower():
                continue
            item = QListWidgetItem(f"📄 {p.name}")
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            if p in self.selected_images:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    def _filter_list(self, text: str) -> None:
        self._populate_list(text)

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        path_str = item.data(Qt.ItemDataRole.UserRole)
        p = Path(path_str)
        if item.checkState() == Qt.CheckState.Checked:
            if p not in self.selected_images:
                if len(self.selected_images) >= 4:
                    removed = self.selected_images.pop(0)
                    for idx in range(self.list_widget.count()):
                        it = self.list_widget.item(idx)
                        if it.data(Qt.ItemDataRole.UserRole) == str(removed):
                            it.setCheckState(Qt.CheckState.Unchecked)
                self.selected_images.append(p)
        else:
            if p in self.selected_images:
                self.selected_images.remove(p)
        self.count_label.setText(f"Selected: {len(self.selected_images)} / 4")

    def _pick_random(self) -> None:
        self.selected_images = random.sample(self.all_images, min(4, len(self.all_images)))
        self._populate_list(self.search_input.text())
        self.count_label.setText(f"Selected: {len(self.selected_images)} / 4")


_THUMBNAIL_BASE_CACHE: OrderedDict[Path, QPixmap] = OrderedDict()
_MAX_THUMBNAIL_CACHE_SIZE = 2000
_DEFAULT_AUTOLABEL_ICON: QIcon | None = None


def _get_default_autolabel_icon(target_size: int = 60) -> QIcon:
    """Singleton default placeholder icon to avoid repeated rendering on large datasets."""
    global _DEFAULT_AUTOLABEL_ICON
    if _DEFAULT_AUTOLABEL_ICON is not None:
        return _DEFAULT_AUTOLABEL_ICON
    pixmap = QPixmap(target_size, target_size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QBrush(QColor("#1e2230")))
    painter.setPen(QColor("#2c2f3b"))
    painter.drawRoundedRect(1, 1, target_size - 2, target_size - 2, 6, 6)
    painter.setPen(QColor("#697082"))
    font = QFont("sans-serif", 16)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "🖼")
    cb_rect = QRectF(4, 4, 15, 15)
    painter.setBrush(QBrush(QColor(15, 23, 42, 190)))
    painter.setPen(QPen(QColor("#64748b"), 1.2))
    painter.drawRoundedRect(cb_rect, 3, 3)
    painter.end()
    _DEFAULT_AUTOLABEL_ICON = QIcon(pixmap)
    return _DEFAULT_AUTOLABEL_ICON


def _get_base_thumbnail_pixmap(path: Path, target_size: int = 60) -> QPixmap:
    """Retrieve or compute a cached 60x60 cropped pixmap for lightning-fast rendering with zero redundant disk reads."""
    if path in _THUMBNAIL_BASE_CACHE:
        _THUMBNAIL_BASE_CACHE.move_to_end(path)
        return _THUMBNAIL_BASE_CACHE[path]

    base = QPixmap(target_size, target_size)
    base.fill(Qt.GlobalColor.transparent)

    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    orig_size = reader.size()

    painter = QPainter(base)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    if orig_size.isValid() and orig_size.width() > 0 and orig_size.height() > 0:
        scale_ratio = max(target_size / orig_size.width(), target_size / orig_size.height())
        scaled_w = max(1, int(round(orig_size.width() * scale_ratio)))
        scaled_h = max(1, int(round(orig_size.height() * scale_ratio)))
        reader.setScaledSize(QSize(scaled_w, scaled_h))
        qimg = reader.read()
        if not qimg.isNull():
            reader_pixmap = QPixmap.fromImage(qimg)
            x_off = max(0, (reader_pixmap.width() - target_size) // 2)
            y_off = max(0, (reader_pixmap.height() - target_size) // 2)
            cropped = reader_pixmap.copy(x_off, y_off, target_size, target_size)
            painter.setBrush(QBrush(cropped))
            painter.setPen(QColor("#2c2f3b"))
            painter.drawRoundedRect(1, 1, target_size - 2, target_size - 2, 6, 6)
        else:
            painter.setBrush(QBrush(QColor("#1e2230")))
            painter.setPen(QColor("#2c2f3b"))
            painter.drawRoundedRect(1, 1, target_size - 2, target_size - 2, 6, 6)
            painter.setPen(QColor("#697082"))
            font = QFont("sans-serif", 16)
            painter.setFont(font)
            painter.drawText(base.rect(), Qt.AlignmentFlag.AlignCenter, "🖼")
    else:
        painter.setBrush(QBrush(QColor("#1e2230")))
        painter.setPen(QColor("#2c2f3b"))
        painter.drawRoundedRect(1, 1, target_size - 2, target_size - 2, 6, 6)
        painter.setPen(QColor("#697082"))
        font = QFont("sans-serif", 16)
        painter.setFont(font)
        painter.drawText(base.rect(), Qt.AlignmentFlag.AlignCenter, "🖼")

    painter.end()

    while len(_THUMBNAIL_BASE_CACHE) >= _MAX_THUMBNAIL_CACHE_SIZE:
        _THUMBNAIL_BASE_CACHE.popitem(last=False)

    _THUMBNAIL_BASE_CACHE[path] = base
    return base


class AutoLabelPreviewThread(QThread):
    """Background worker generating preview detections without freezing Qt GUI."""

    sample_processed = Signal(int, int, Path, object)
    preview_finished = Signal(object)
    preview_failed = Signal(str)

    def __init__(
        self,
        engine: AutoLabelEngine,
        images: list[Path],
        config: AutoLabelConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.images = images
        self.config = config
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            results: dict[Path, AutoLabelResult] = {}
            for idx, img_path in enumerate(self.images):
                if self._cancel_requested:
                    break
                res = self.engine.run_preview(img_path, self.config)
                results[img_path] = res
                self.sample_processed.emit(idx + 1, len(self.images), img_path, res)
            self.preview_finished.emit(results)
        except Exception as err:
            LOGGER.exception("AutoLabel preview thread error")
            self.preview_failed.emit(str(err))


class AutoLabelBatchThread(QThread):
    """Background worker executing dataset batch auto-annotation."""

    batch_progress = Signal(int, int, Path, object)
    batch_finished = Signal(object)
    batch_failed = Signal(str)

    def __init__(
        self,
        engine: AutoLabelEngine,
        documents: list[AnnotationDocument],
        config: AutoLabelConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.documents = documents
        self.config = config
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        try:
            def _progress(cur: int, tot: int, p: Path, res: AutoLabelResult) -> None:
                self.batch_progress.emit(cur, tot, p, res)

            def _is_cancelled() -> bool:
                return self._cancel_requested

            updated = self.engine.run_batch(
                self.documents,
                self.config,
                progress_callback=_progress,
                is_cancelled=_is_cancelled,
            )
            self.batch_finished.emit(updated)
        except Exception as err:
            LOGGER.exception("AutoLabel batch thread error")
            self.batch_failed.emit(str(err))


class AutoLabelDialog(QDialog):
    """Roboflow-style Auto Label modal with clean unified top bar, direct 4-sample grid, and approved batch application."""

    preview_applied = Signal(AutoLabelResult)
    batch_completed = Signal(object)

    def __init__(
        self,
        image_paths: Sequence[Path],
        current_image_path: Path | None = None,
        engine: AutoLabelEngine | None = None,
        initial_classes: Sequence[AutoLabelClass] | None = None,
        ground_truth: dict[Path, AnnotationDocument] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("AutoLabelDialog")
        self.setWindowTitle("Auto Label")
        self.resize(1320, 840)
        self.setMinimumSize(1060, 660)

        self.image_paths = list(image_paths)
        self.ground_truth = ground_truth or {}

        # Initialize primary image + up to 3 diverse sample images (prioritizing annotated ground truth)
        if self.image_paths:
            annotated_in_dataset = [
                p
                for p in self.image_paths
                if p in self.ground_truth and self.ground_truth[p].annotations
            ]

            if current_image_path and current_image_path in self.image_paths:
                primary = current_image_path
            elif annotated_in_dataset:
                primary = annotated_in_dataset[0]
            else:
                primary = self.image_paths[0]

            other_samples = [p for p in self.image_paths if p != primary]
            annotated_others = [p for p in other_samples if p in annotated_in_dataset]
            unannotated_others = [p for p in other_samples if p not in annotated_in_dataset]

            selected_others = list(annotated_others[:3])
            needed = 3 - len(selected_others)
            if needed > 0 and unannotated_others:
                selected_others += random.sample(
                    unannotated_others, min(needed, len(unannotated_others))
                )

            self.preview_image_paths = [primary] + selected_others
            self.current_image_path = primary
        else:
            self.preview_image_paths = []
            self.current_image_path = None

        self.engine = engine or AutoLabelEngine()
        raw_model = getattr(self.engine, "_yolo_model_name", "yolo11n.pt")
        initial_yolo = raw_model if isinstance(raw_model, str) and raw_model else "yolo11n.pt"
        self._active_yolo_models: list[str] = [initial_yolo]

        self.classes: list[AutoLabelClass] = [
            AutoLabelClass(c.name, c.prompt, c.color, c.enabled)
            for c in (initial_classes or DEFAULT_AUTO_LABEL_CLASSES)
        ]
        self._class_cards: list[ClassCardWidget] = []
        self._latest_result: AutoLabelResult | None = None
        self._latest_results: dict[Path, AutoLabelResult] = {}
        self._is_previewing = False
        self._preview_thread: AutoLabelPreviewThread | None = None
        self._batch_thread: AutoLabelBatchThread | None = None

        self._init_ui()
        self._render_initial_images()
        self._set_preview_view_mode(0)

    def _init_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ----------------------------------------------------------------------
        # Top Header Bar (Breadcrumb & Title)
        # ----------------------------------------------------------------------
        header = QFrame()
        header.setObjectName("DialogHeader")
        header.setFixedHeight(50)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)
        header_layout.setSpacing(12)

        title_label = QLabel("Auto Label")
        title_label.setObjectName("HeaderTitle")

        header_layout.addWidget(title_label)
        header_layout.addStretch(1)

        close_btn = QPushButton("✕")
        close_btn.setObjectName("HeaderCloseBtn")
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.reject)
        header_layout.addWidget(close_btn)

        root_layout.addWidget(header)

        # ----------------------------------------------------------------------
        # Main Splitter: Left Sidebar & Right Stage
        # ----------------------------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("MainSplitter")
        splitter.setHandleWidth(1)

        # --- LEFT PANEL: Classes & Prompts + Preview Images ---
        left_panel = QFrame()
        left_panel.setObjectName("LeftPanel")
        left_panel.setMinimumWidth(330)
        left_panel.setMaximumWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 14, 16, 14)
        left_layout.setSpacing(10)

        # Classes Header Row
        classes_header_row = QHBoxLayout()
        classes_header_row.setContentsMargins(0, 0, 0, 0)
        classes_header_row.setSpacing(8)

        self.classes_title = QLabel("Classes")
        self.classes_title.setObjectName("SectionTitle")

        self.class_count_badge = QLabel(str(len(self.classes)))
        self.class_count_badge.setObjectName("CountBadge")

        classes_header_row.addWidget(self.classes_title)
        classes_header_row.addWidget(self.class_count_badge)
        classes_header_row.addStretch(1)

        left_layout.addLayout(classes_header_row)

        info_sub = QLabel(
            "Use custom prompts to specify descriptions (e.g. 'sedan, suv' or 'delivery truck')."
        )
        info_sub.setObjectName("InfoSubtitle")
        info_sub.setWordWrap(True)
        left_layout.addWidget(info_sub)

        # Actions Row: Clear All & Add Class
        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        clear_all_btn = QPushButton("✕ Clear All")
        clear_all_btn.setObjectName("ClearAllBtn")
        clear_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_all_btn.clicked.connect(self._clear_all_classes)

        add_class_btn = QPushButton("+ Add Class")
        add_class_btn.setObjectName("AddClassBtn")
        add_class_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_class_btn.clicked.connect(self._add_new_class)

        actions_row.addWidget(clear_all_btn)
        actions_row.addStretch(1)
        actions_row.addWidget(add_class_btn)
        left_layout.addLayout(actions_row)

        # Scrollable Classes Container
        self.classes_scroll = QScrollArea()
        self.classes_scroll.setObjectName("ClassesScrollArea")
        self.classes_scroll.setWidgetResizable(True)
        self.classes_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.classes_container = QWidget()
        self.classes_container.setObjectName("ClassesContainer")
        self.classes_vbox = QVBoxLayout(self.classes_container)
        self.classes_vbox.setContentsMargins(0, 0, 4, 0)
        self.classes_vbox.setSpacing(6)
        self.classes_vbox.addStretch(1)

        self.classes_scroll.setWidget(self.classes_container)
        left_layout.addWidget(self.classes_scroll, 1)

        # Preview Samples Section
        prev_header_row = QHBoxLayout()
        prev_header_row.setContentsMargins(0, 0, 0, 0)
        prev_header_row.setSpacing(6)

        prev_header = QLabel("Preview Samples (4)")
        prev_header = QLabel("Preview Samples (4)")
        prev_header.setObjectName("SectionTitle")

        self.shuffle_btn = QPushButton("🎲 Random 4")
        self.shuffle_btn.setObjectName("PreviewActionBtn")
        self.shuffle_btn.setToolTip("Randomize 4 sample images from the batch")
        self.shuffle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.shuffle_btn.clicked.connect(self._randomize_preview_samples)

        prev_header_row.addWidget(prev_header)
        prev_header_row.addStretch(1)
        prev_header_row.addWidget(self.shuffle_btn)
        left_layout.addLayout(prev_header_row)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("ImageSearchInput")
        self.search_input.setPlaceholderText("🔍 Filter batch images...")
        self.search_input.textChanged.connect(self._filter_image_list)
        left_layout.addWidget(self.search_input)

        self.image_list = QListWidget()
        self.image_list.setObjectName("PreviewImageList")
        self.image_list.setViewMode(QListView.ViewMode.IconMode)
        self.image_list.setIconSize(QSize(62, 62))
        self.image_list.setGridSize(QSize(74, 74))
        self.image_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.image_list.setMovement(QListView.Movement.Static)
        self.image_list.setSpacing(4)
        self.image_list.setWordWrap(False)
        self.image_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_list.setMinimumHeight(150)
        self.image_list.verticalScrollBar().valueChanged.connect(self._load_visible_autolabel_thumbnails)
        self._populate_image_list()
        self.image_list.itemClicked.connect(self._on_preview_image_clicked)
        self.image_list.itemSelectionChanged.connect(self._on_preview_image_selected)
        left_layout.addWidget(self.image_list, 1)

        splitter.addWidget(left_panel)

        # --- RIGHT PANEL: Direct Images Stage + Model Mix Controls ---
        right_panel = QFrame()
        right_panel.setObjectName("RightPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 14, 16, 14)
        right_layout.setSpacing(10)

        # ----------------------------------------------------------------------
        # Unified Model Command Bar
        # ----------------------------------------------------------------------
        control_card = QFrame()
        control_card.setObjectName("TopControlCard")
        control_layout = QVBoxLayout(control_card)
        control_layout.setContentsMargins(14, 10, 14, 10)
        control_layout.setSpacing(8)

        # Row 1: Model Preset, Weights, Sliders
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(10)

        self.model_combo = QComboBox()
        self.model_combo.setObjectName("ModelCombo")
        for mode in AutoLabelPipelineMode:
            self.model_combo.addItem(f"⚡ {mode.display_name}", mode.value)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)

        self.model_badge = QLabel("Mask labels")
        self.model_badge.setObjectName("ModelBadge")

        self.yolo_weights_btn = QPushButton("📦 YOLO (1 model)")
        self.yolo_weights_btn.setObjectName("YoloWeightsBtn")
        self.yolo_weights_btn.setToolTip("Configure up to 3 simultaneous YOLO models (click to manage)")
        self.yolo_weights_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.yolo_weights_btn.clicked.connect(self._open_multi_yolo_menu)
        self.yolo_weights_btn.setVisible(False)
        self._update_yolo_button_ui()

        # Confidence Threshold Slider
        conf_container = QHBoxLayout()
        conf_container.setContentsMargins(0, 0, 0, 0)
        conf_container.setSpacing(4)
        conf_title = QLabel("Conf:")
        conf_title.setObjectName("ParamTitle")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setObjectName("ConfSlider")
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(35)
        self.conf_slider.setFixedWidth(65)
        self.conf_val_label = QLabel("35%")
        self.conf_val_label.setObjectName("ParamVal")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_val_label.setText(f"{v}%")
        )
        conf_container.addWidget(conf_title)
        conf_container.addWidget(self.conf_slider)
        conf_container.addWidget(self.conf_val_label)

        # IoU Deduplication Slider
        iou_container = QHBoxLayout()
        iou_container.setContentsMargins(0, 0, 0, 0)
        iou_container.setSpacing(4)
        iou_title = QLabel("IoU Dedup:")
        iou_title.setObjectName("ParamTitle")
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setObjectName("IouSlider")
        self.iou_slider.setRange(10, 95)
        self.iou_slider.setValue(45)
        self.iou_slider.setFixedWidth(65)
        self.iou_val_label = QLabel("45%")
        self.iou_val_label.setObjectName("ParamVal")
        self.iou_slider.valueChanged.connect(
            lambda v: self.iou_val_label.setText(f"{v}%")
        )

        iou_container.addWidget(iou_title)
        iou_container.addWidget(self.iou_slider)
        iou_container.addWidget(self.iou_val_label)

        self.preview_mix_btn = QPushButton("⚡ Preview Mix")
        self.preview_mix_btn.setObjectName("PreviewMixTopBtn")
        self.preview_mix_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.preview_mix_btn.clicked.connect(self._run_single_preview)

        self.ai_tuner_btn = QPushButton("✨ AI Auto-Tune")
        self.ai_tuner_btn.setObjectName("AITunerTopBtn")
        self.ai_tuner_btn.setToolTip("Autonomous Prompt & Setting Optimizer using Vision LLM")
        self.ai_tuner_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ai_tuner_btn.clicked.connect(self._open_ai_tuner_dialog)

        row1.addWidget(self.model_combo, 0)
        row1.addWidget(self.model_badge, 0)
        row1.addWidget(self.yolo_weights_btn, 0)
        row1.addStretch(1)
        row1.addLayout(conf_container)
        row1.addLayout(iou_container)
        row1.addWidget(self.preview_mix_btn, 0)
        row1.addWidget(self.ai_tuner_btn, 0)

        # Row 2: Detector Pill Checkboxes
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(10)

        detector_title = QLabel("⚡ Detectors:")
        detector_title.setObjectName("EnsembleTitle")

        self.dino_chk = QCheckBox("Grounding DINO")
        self.dino_chk.setObjectName("EnsembleCheck")
        self.dino_chk.setChecked(True)
        self.dino_chk.toggled.connect(self._on_detector_toggled)

        self.yolo_chk = QCheckBox("YOLO")
        self.yolo_chk.setObjectName("EnsembleCheck")
        self.yolo_chk.setChecked(False)
        self.yolo_chk.toggled.connect(self._on_detector_toggled)

        self.florence_chk = QCheckBox("Florence-2 VLM")
        self.florence_chk.setObjectName("EnsembleCheck")
        self.florence_chk.setChecked(False)
        self.florence_chk.toggled.connect(self._on_detector_toggled)

        ensemble_sep = QFrame()
        ensemble_sep.setFrameShape(QFrame.Shape.VLine)
        ensemble_sep.setObjectName("EnsembleSep")

        self.sam2_chk = QCheckBox("SAM 2 Polygon Masks")
        self.sam2_chk.setObjectName("EnsembleCheckSam2")
        self.sam2_chk.setChecked(True)
        self.sam2_chk.toggled.connect(self._on_sam2_toggled)

        row2.addWidget(detector_title, 0)
        row2.addWidget(self.dino_chk, 0)
        row2.addWidget(self.yolo_chk, 0)
        row2.addWidget(self.florence_chk, 0)
        row2.addWidget(ensemble_sep, 0)
        row2.addWidget(self.sam2_chk, 0)
        row2.addStretch(1)

        control_layout.addLayout(row1)
        control_layout.addLayout(row2)
        right_layout.addWidget(control_card, 0)

        # ----------------------------------------------------------------------
        # Image First Canvas Stage (2x2 Grid View + Focused Single View)
        # ----------------------------------------------------------------------
        stage_frame = QFrame()
        stage_frame.setObjectName("StageFrame")
        stage_layout = QVBoxLayout(stage_frame)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(0)

        # View Switcher Bar
        view_switcher_frame = QFrame()
        view_switcher_frame.setObjectName("ViewSwitcherFrame")
        view_switcher_layout = QHBoxLayout(view_switcher_frame)
        view_switcher_layout.setContentsMargins(10, 6, 10, 6)
        view_switcher_layout.setSpacing(8)

        self.grid_view_btn = QPushButton("🔲 4-Grid View (2x2)")
        self.grid_view_btn.setObjectName("ViewModeBtnActive")
        self.grid_view_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.grid_view_btn.clicked.connect(lambda: self._set_preview_view_mode(0))

        self.single_view_btn = QPushButton("🔍 Focused View")
        self.single_view_btn.setObjectName("ViewModeBtn")
        self.single_view_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.single_view_btn.clicked.connect(lambda: self._set_preview_view_mode(1))

        view_switcher_layout.addWidget(self.grid_view_btn)
        view_switcher_layout.addWidget(self.single_view_btn)
        view_switcher_layout.addStretch(1)

        # Sample tabs (active in focused view)
        self.sample_tab_buttons: list[QPushButton] = []
        for i in range(4):
            btn = QPushButton(f"Sample {i + 1}")
            btn.setObjectName("SampleTabBtn")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, idx=i: self._on_sample_tab_clicked(idx))
            self.sample_tab_buttons.append(btn)
            view_switcher_layout.addWidget(btn)

        stage_layout.addWidget(view_switcher_frame, 0)

        # Stacked Views (0: 2x2 Grid, 1: Single Canvas)
        self.preview_views_stack = QStackedWidget()
        self.preview_views_stack.setObjectName("PreviewViewsStack")

        # View 0: 2x2 Grid View
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(6, 6, 6, 6)
        self.grid_layout.setSpacing(8)

        self.preview_cards: list[PreviewCardWidget] = []
        for i in range(4):
            card = PreviewCardWidget(slot_index=i)
            card.clicked.connect(self._on_card_zoomed)
            row = i // 2
            col = i % 2
            self.grid_layout.addWidget(card, row, col)
            self.preview_cards.append(card)

        self.preview_views_stack.addWidget(self.grid_container)

        # View 1: Single Focused Canvas
        self.single_container = QWidget()
        single_layout = QVBoxLayout(self.single_container)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.setSpacing(0)
        self.preview_canvas = AutoLabelPreviewCanvas()
        single_layout.addWidget(self.preview_canvas, 1)

        self.preview_views_stack.addWidget(self.single_container)
        stage_layout.addWidget(self.preview_views_stack, 1)

        # Bottom Results, Stats & Approval Bar
        bottom_frame = QFrame()
        bottom_frame.setObjectName("PrevBottomFrame")
        bottom_bar = QHBoxLayout(bottom_frame)
        bottom_bar.setContentsMargins(14, 8, 14, 8)
        bottom_bar.setSpacing(10)

        self.result_stats_label = QLabel("⚡ 4 Sample Images Loaded. Click 'Preview Mix' to evaluate detections.")
        self.result_stats_label.setObjectName("ResultStatsLabel")

        # Inline progress indicator
        self.inline_progress = QProgressBar()
        self.inline_progress.setObjectName("InlineProgressBar")
        self.inline_progress.setFixedWidth(130)
        self.inline_progress.setTextVisible(False)
        self.inline_progress.setVisible(False)

        self.apply_btn = QPushButton("✔ Apply to Samples")
        self.apply_btn.setObjectName("ApplyBtn")
        self.apply_btn.setToolTip("Apply predicted annotations to current preview sample(s)")
        self.apply_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.apply_btn.clicked.connect(self._apply_preview_to_image)

        self.auto_label_btn = QPushButton(f"⚡ Auto Label Entire Batch ({len(self.image_paths)})")
        self.auto_label_btn.setObjectName("AutoLabelPrimaryBtn")
        self.auto_label_btn.setToolTip("Run this approved model mix on every image in the dataset batch")
        self.auto_label_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_label_btn.clicked.connect(self._run_batch_auto_label)

        bottom_bar.addWidget(self.result_stats_label, 1)
        bottom_bar.addWidget(self.inline_progress, 0)
        bottom_bar.addWidget(self.apply_btn, 0)
        bottom_bar.addWidget(self.auto_label_btn, 0)

        stage_layout.addWidget(bottom_frame, 0)
        right_layout.addWidget(stage_frame, 1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root_layout.addWidget(splitter, 1)

        self._rebuild_class_cards()
        self._apply_stylesheet()

    def _render_initial_images(self) -> None:
        """Render the 4 sample images on the cards immediately without blocking."""
        for i, card in enumerate(self.preview_cards):
            if i < len(self.preview_image_paths):
                p = self.preview_image_paths[i]
                card.set_data(p, None)
                card.setVisible(True)
            else:
                card.setVisible(False)

        for i, btn in enumerate(self.sample_tab_buttons):
            if i < len(self.preview_image_paths):
                p = self.preview_image_paths[i]
                short_name = p.name if len(p.name) <= 14 else f"{p.name[:6]}...{p.name[-5:]}"
                btn.setText(f"Sample {i + 1}: {short_name}")
                btn.setToolTip(str(p))
                btn.setVisible(True)
            else:
                btn.setVisible(False)

        if self.current_image_path and self.current_image_path.is_file():
            self.preview_canvas.set_image(self.current_image_path)
        self._update_sample_tabs_ui()

    @staticmethod
    def _create_thumbnail_icon(path: Path, is_preview: bool = False, slot_num: int = 0) -> QIcon:
        """Create a thumbnail preview card from in-memory cached pixmap instantly."""
        target_size = 60
        base_pixmap = _get_base_thumbnail_pixmap(path, target_size)
        pixmap = QPixmap(base_pixmap)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if is_preview:
            # Highlight border
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#818cf8"), 2))
            painter.drawRoundedRect(1, 1, target_size - 2, target_size - 2, 6, 6)

            # Checked checkbox: vibrant indigo background with white checkmark
            cb_rect = QRectF(4, 4, 15, 15)
            painter.setBrush(QBrush(QColor("#6366f1")))
            painter.setPen(QPen(QColor("#a5b4fc"), 1))
            painter.drawRoundedRect(cb_rect, 3, 3)

            painter.setPen(
                QPen(QColor("#ffffff"), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            )
            painter.drawLine(int(cb_rect.left() + 3), int(cb_rect.top() + 7), int(cb_rect.left() + 6), int(cb_rect.bottom() - 3))
            painter.drawLine(int(cb_rect.left() + 6), int(cb_rect.bottom() - 3), int(cb_rect.right() - 3), int(cb_rect.top() + 4))

            # Sample slot badge (top-right) if slot number available
            if slot_num > 0:
                badge_rect = QRectF(target_size - 24, 4, 20, 14)
                painter.setBrush(QBrush(QColor("#4338ca")))
                painter.setPen(QPen(QColor("#c7d2fe"), 1))
                painter.drawRoundedRect(badge_rect, 3, 3)
                painter.setPen(QColor("#ffffff"))
                font = QFont("sans-serif", 8, QFont.Weight.Bold)
                painter.setFont(font)
                painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, f"S{slot_num}")
        else:
            # Unchecked checkbox: sleek dark translucent box with subtle border
            cb_rect = QRectF(4, 4, 15, 15)
            painter.setBrush(QBrush(QColor(15, 23, 42, 190)))
            painter.setPen(QPen(QColor("#64748b"), 1.2))
            painter.drawRoundedRect(cb_rect, 3, 3)

        painter.end()
        return QIcon(pixmap)

    def _ensure_autolabel_item_thumbnail(self, item: QListWidgetItem) -> None:
        """Ensure thumbnail icon is rendered and set for the given item."""
        path_str = item.data(Qt.ItemDataRole.UserRole)
        if not path_str:
            return
        p = Path(path_str)
        if p in self._autolabel_rendered_paths:
            return
        self._autolabel_rendered_paths.add(p)
        is_preview = p in self.preview_image_paths
        slot = (self.preview_image_paths.index(p) + 1) if is_preview else 0
        item.setIcon(self._create_thumbnail_icon(p, is_preview, slot))

    def _load_visible_autolabel_thumbnails(self) -> None:
        """Immediately render thumbnails for all items currently visible in preview list viewport."""
        if not hasattr(self, "image_list") or not hasattr(self, "_autolabel_items_by_path") or not self._autolabel_items_by_path:
            return
        viewport_rect = self.image_list.viewport().rect()
        if not viewport_rect.isValid() or viewport_rect.isEmpty():
            for idx in range(min(40, self.image_list.count())):
                item = self.image_list.item(idx)
                if item is not None:
                    self._ensure_autolabel_item_thumbnail(item)
            return

        count = self.image_list.count()
        first_visible = -1
        last_visible = -1
        for row in range(count):
            rect = self.image_list.visualRect(self.image_list.model().index(row, 0))
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
            item = self.image_list.item(row)
            if item is not None:
                self._ensure_autolabel_item_thumbnail(item)

    def _start_autolabel_background_loader(self) -> None:
        """Progressively render remaining unrendered images in preview list."""
        if not hasattr(self, "_autolabel_unrendered_queue") or not self._autolabel_unrendered_queue:
            return
        if not hasattr(self, "_autolabel_batch_timer") or self._autolabel_batch_timer is None:
            self._autolabel_batch_timer = QTimer(self)
            self._autolabel_batch_timer.setInterval(10)
            self._autolabel_batch_timer.timeout.connect(self._process_autolabel_background_batch)
        if not self._autolabel_batch_timer.isActive():
            self._autolabel_batch_timer.start()

    def _process_autolabel_background_batch(self) -> None:
        """Process a small batch of unrendered thumbnails in the background."""
        if not self._autolabel_unrendered_queue:
            if self._autolabel_batch_timer and self._autolabel_batch_timer.isActive():
                self._autolabel_batch_timer.stop()
            return
        batch_size = 20
        processed = 0
        while self._autolabel_unrendered_queue and processed < batch_size:
            path = self._autolabel_unrendered_queue.popleft()
            if path in self._autolabel_rendered_paths:
                continue
            item = self._autolabel_items_by_path.get(path)
            if item is not None:
                self._ensure_autolabel_item_thumbnail(item)
                processed += 1
        if not self._autolabel_unrendered_queue and self._autolabel_batch_timer and self._autolabel_batch_timer.isActive():
            self._autolabel_batch_timer.stop()

    def _populate_image_list(self, filter_text: str = "") -> None:
        if hasattr(self, "_autolabel_batch_timer") and self._autolabel_batch_timer and self._autolabel_batch_timer.isActive():
            self._autolabel_batch_timer.stop()

        self._autolabel_rendered_paths: set[Path] = set()
        self._autolabel_items_by_path: dict[Path, QListWidgetItem] = {}
        self._autolabel_unrendered_queue: deque[Path] = deque()

        self.image_list.setUpdatesEnabled(False)
        self.image_list.blockSignals(True)
        self.image_list.clear()
        filter_lower = filter_text.lower()
        selected_idx = 0
        visible_idx = 0

        # Pin selected preview images at the top, followed by other annotated images, then unannotated
        pinned_selected = [p for p in self.preview_image_paths if p in self.image_paths]
        remaining = [p for p in self.image_paths if p not in self.preview_image_paths]
        other_annotated = [
            p for p in remaining if p in self.ground_truth and self.ground_truth[p].annotations
        ]
        unannotated = [
            p for p in remaining if not (p in self.ground_truth and self.ground_truth[p].annotations)
        ]
        ordered_paths = pinned_selected + other_annotated + unannotated

        default_icon = _get_default_autolabel_icon()
        eager_limit = 40

        for idx, img_path in enumerate(ordered_paths):
            if filter_lower and filter_lower not in img_path.name.lower():
                continue
            self._autolabel_unrendered_queue.append(img_path)
            is_preview = img_path in self.preview_image_paths
            slot = (self.preview_image_paths.index(img_path) + 1) if is_preview else 0
            item = QListWidgetItem()
            if is_preview or idx < eager_limit or img_path in _THUMBNAIL_BASE_CACHE:
                item.setIcon(self._create_thumbnail_icon(img_path, is_preview, slot))
                self._autolabel_rendered_paths.add(img_path)
            else:
                item.setIcon(default_icon)
            prefix = f"★ [Sample {slot}] " if is_preview else ""
            item.setToolTip(f"{prefix}{img_path.name}")
            item.setData(Qt.ItemDataRole.UserRole, str(img_path))
            self.image_list.addItem(item)
            self._autolabel_items_by_path[img_path] = item
            if self.current_image_path and img_path == self.current_image_path:
                selected_idx = visible_idx
            visible_idx += 1

        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(selected_idx)
        self.image_list.blockSignals(False)
        self.image_list.setUpdatesEnabled(True)

        self._load_visible_autolabel_thumbnails()
        self._start_autolabel_background_loader()

    def _filter_image_list(self, text: str) -> None:
        self._populate_image_list(text)

    def _on_preview_image_clicked(self, item: QListWidgetItem) -> None:
        if item is None:
            return
        path_str = item.data(Qt.ItemDataRole.UserRole)
        if not path_str:
            return
        p = Path(path_str)
        self.current_image_path = p
        if p in self.preview_image_paths:
            if len(self.preview_image_paths) > 1:
                self.preview_image_paths.remove(p)
                self.current_image_path = self.preview_image_paths[0]
        else:
            if len(self.preview_image_paths) >= 4:
                self.preview_image_paths.pop(0)
            self.preview_image_paths.append(p)

        self._populate_image_list(self.search_input.text() if hasattr(self, "search_input") else "")
        self._render_initial_images()
        self._update_sample_tabs_ui()

    def _on_preview_image_selected(self) -> None:
        item = self.image_list.currentItem()
        if item is not None:
            path_str = item.data(Qt.ItemDataRole.UserRole)
            if path_str:
                p = Path(path_str)
                self.current_image_path = p
                is_preview = p in self.preview_image_paths
                slot = (self.preview_image_paths.index(p) + 1) if is_preview else 0
                self._ensure_autolabel_item_thumbnail(item)
                if is_preview:
                    self._on_card_zoomed(p)

    def _randomize_preview_samples(self) -> None:
        if not self.image_paths:
            return
        self.preview_image_paths = random.sample(self.image_paths, min(4, len(self.image_paths)))
        if self.preview_image_paths:
            self.current_image_path = self.preview_image_paths[0]
        self._populate_image_list(self.search_input.text() if hasattr(self, "search_input") else "")
        self._render_initial_images()
        self._run_single_preview()

    def _open_cherry_pick_dialog(self) -> None:
        dialog = CherryPickDialog(self.image_paths, self.preview_image_paths, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.selected_images:
                self.preview_image_paths = dialog.selected_images
                self.current_image_path = self.preview_image_paths[0]
                self._populate_image_list(self.search_input.text() if hasattr(self, "search_input") else "")
                self._render_initial_images()
                self._run_single_preview()

    def _set_preview_view_mode(self, mode: int) -> None:
        """Mode 0: 4-Grid View, Mode 1: Focused Single View."""
        self.preview_views_stack.setCurrentIndex(mode)
        num_samples = len(self.preview_image_paths)
        if mode == 0:
            self.grid_view_btn.setObjectName("ViewModeBtnActive")
            self.single_view_btn.setObjectName("ViewModeBtn")
            for btn in self.sample_tab_buttons:
                btn.setVisible(False)
            if num_samples <= 1:
                self.apply_btn.setText("✔ Apply to Current Image")
                self.apply_btn.setToolTip("Apply annotations to the active sample image")
            else:
                self.apply_btn.setText(f"✔ Apply to {num_samples} Samples")
                self.apply_btn.setToolTip(f"Apply annotations across all {num_samples} sample images in the preview grid")
        else:
            self.grid_view_btn.setObjectName("ViewModeBtn")
            self.single_view_btn.setObjectName("ViewModeBtnActive")
            for i, btn in enumerate(self.sample_tab_buttons):
                btn.setVisible(i < num_samples)
            self.apply_btn.setText("✔ Apply to Current Image")
            self.apply_btn.setToolTip("Apply annotations ONLY to this single focused sample image")
        self.grid_view_btn.setStyle(self.grid_view_btn.style())
        self.single_view_btn.setStyle(self.single_view_btn.style())

    def _on_card_zoomed(self, img_path: Path) -> None:
        self.current_image_path = img_path
        if img_path in self._latest_results:
            self._latest_result = self._latest_results[img_path]
            self.preview_canvas.set_result(img_path, self._latest_result)
        else:
            self.preview_canvas.set_image(img_path)
        self._update_sample_tabs_ui()
        self._set_preview_view_mode(1)

    def _on_sample_tab_clicked(self, index: int) -> None:
        if index < len(self.preview_image_paths):
            p = self.preview_image_paths[index]
            self.current_image_path = p
            if p in self._latest_results:
                self._latest_result = self._latest_results[p]
                self.preview_canvas.set_result(p, self._latest_result)
            else:
                self.preview_canvas.set_image(p)
            self._update_sample_tabs_ui()

    def _update_sample_tabs_ui(self) -> None:
        for i, btn in enumerate(self.sample_tab_buttons):
            if i < len(self.preview_image_paths):
                p = self.preview_image_paths[i]
                is_active = (p == self.current_image_path)
                btn.setObjectName("SampleTabBtnActive" if is_active else "SampleTabBtn")
                btn.setStyle(btn.style())

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            QDialog#AutoLabelDialog {
                background-color: #0b0f19;
                color: #f8fafc;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            QLabel {
                background: transparent;
                background-color: transparent;
                border: none;
                padding: 0px;
                margin: 0px;
            }
            QFrame#DialogHeader {
                background-color: #111827;
                border-bottom: 1px solid #1f293d;
            }
            QLabel#BreadcrumbLabel {
                color: #94a3b8;
                font-size: 13px;
                font-weight: 500;
                background: transparent;
                border: none;
                padding: 0px;
            }
            QLabel#BreadcrumbLabel:hover {
                color: #ffffff;
            }
            QLabel#BreadcrumbSep {
                color: #64748b;
                font-size: 14px;
                font-weight: 400;
                background: transparent;
            }
            QLabel#HeaderTitle {
                color: #ffffff;
                font-size: 14px;
                font-weight: 700;
                background: transparent;
            }
            QPushButton#HeaderCloseBtn {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton#HeaderCloseBtn:hover {
                background-color: #ef4444;
                border-color: #ef4444;
                color: #ffffff;
            }
            QSplitter#MainSplitter::handle {
                background-color: #1e293b;
            }

            /* --- Left Sidebar --- */
            QFrame#LeftPanel {
                background-color: #0f172a;
                border-right: 1px solid #1e293b;
            }
            QLabel#SectionTitle {
                font-size: 13px;
                font-weight: 700;
                color: #f8fafc;
                background: transparent;
                border: none;
                padding: 0px;
            }
            QLabel#CountBadge {
                background-color: #1e1b4b;
                border: 1px solid #3730a3;
                color: #a5b4fc;
                font-size: 10px;
                font-weight: 700;
                border-radius: 9px;
                padding: 1px 7px;
            }
            QLabel#InfoSubtitle {
                font-size: 11px;
                color: #94a3b8;
                line-height: 1.35;
                background: transparent;
                padding: 0px;
            }
            QPushButton#ClearAllBtn {
                background: transparent;
                border: none;
                color: #64748b;
                font-size: 11px;
                font-weight: 600;
                padding: 2px 4px;
            }
            QPushButton#ClearAllBtn:hover {
                color: #ef4444;
            }
            QPushButton#AddClassBtn {
                background-color: #312e81;
                border: 1px solid #4338ca;
                border-radius: 6px;
                color: #c7d2fe;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 12px;
            }
            QPushButton#AddClassBtn:hover {
                background-color: #3730a3;
                border-color: #6366f1;
                color: #ffffff;
            }
            QPushButton#PreviewActionBtn {
                background: transparent;
                border: 1px solid #283654;
                border-radius: 5px;
                font-size: 11px;
                font-weight: 600;
                color: #c7d2fe;
                padding: 3px 8px;
            }
            QPushButton#PreviewActionBtn:hover {
                background-color: #1e293b;
                border-color: #6366f1;
                color: #ffffff;
            }
            QLineEdit#ImageSearchInput {
                background-color: #141c2e;
                border: 1px solid #283654;
                border-radius: 6px;
                padding: 5px 10px;
                font-size: 11px;
                color: #cbd5e1;
            }
            QLineEdit#ImageSearchInput:focus {
                border-color: #6366f1;
                background-color: #1a2540;
                color: #ffffff;
            }

            /* --- Classes Scroll Container (Transparent, no gray box) --- */
            QScrollArea#ClassesScrollArea,
            QWidget#ClassesContainer,
            QScrollArea#ClassesScrollArea > QWidget,
            QScrollArea#ClassesScrollArea > QWidget > QWidget {
                background: transparent;
                background-color: transparent;
                border: none;
            }

            /* --- Class Card --- */
            QFrame#ClassCard {
                background: transparent;
                background-color: transparent;
                border: none;
                border-bottom: 1px solid #1e293b;
                border-radius: 0px;
                padding-bottom: 4px;
            }
            QFrame#ClassCard:hover {
                background-color: rgba(255, 255, 255, 0.02);
            }
            QLineEdit#ClassNameInput {
                background: transparent;
                background-color: transparent;
                border: none;
                padding: 0px 2px;
                font-size: 13px;
                font-weight: 600;
                color: #f8fafc;
            }
            QLineEdit#ClassNameInput:focus {
                background-color: #1e293b;
                border: 1px solid #6366f1;
                border-radius: 3px;
            }
            QLineEdit#ClassPromptInput {
                background: transparent;
                background-color: transparent;
                border: none;
                padding: 0px 2px;
                font-size: 11px;
                color: #8b9dc3;
            }
            QLineEdit#ClassPromptInput:focus {
                background-color: #1e293b;
                border: 1px solid #6366f1;
                border-radius: 3px;
                color: #ffffff;
            }
            QPushButton#ClassDeleteBtn {
                background: transparent;
                border: none;
                border-radius: 3px;
                font-size: 11px;
                color: #64748b;
            }
            QPushButton#ClassDeleteBtn:hover {
                background-color: #3b1820;
                color: #ef4444;
            }

            /* --- Preview List / Thumbnail Grid --- */
            QListWidget#PreviewImageList {
                background-color: #0c1120;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 6px;
                outline: none;
            }
            QListWidget#PreviewImageList::item {
                border-radius: 8px;
                margin: 3px;
                padding: 2px;
                border: 2px solid transparent;
            }
            QListWidget#PreviewImageList::item:hover {
                background-color: #1a2236;
                border: 2px solid #4338ca;
            }
            QListWidget#PreviewImageList::item:selected {
                background-color: #2e2a72;
                border: 2px solid #818cf8;
            }

            /* --- Right Stage Panel & Unified Command Bar --- */
            QFrame#RightPanel {
                background-color: #0b0f19;
            }
            QFrame#TopControlCard {
                background-color: #111827;
                border: 1px solid #1f293d;
                border-radius: 10px;
            }
            QLabel#EnsembleTitle {
                font-size: 11px;
                font-weight: 700;
                color: #c7d2fe;
                background: transparent;
            }
            QCheckBox#EnsembleCheck, QCheckBox#EnsembleCheckSam2 {
                color: #f1f5f9;
                font-size: 11px;
                font-weight: 600;
                spacing: 5px;
                background: transparent;
            }
            QCheckBox#EnsembleCheck::indicator, QCheckBox#EnsembleCheckSam2::indicator {
                width: 14px;
                height: 14px;
                border-radius: 3px;
                border: 1px solid #475569;
                background-color: #1e293b;
            }
            QCheckBox#EnsembleCheck::indicator:checked {
                background-color: #6366f1;
                border-color: #818cf8;
            }
            QCheckBox#EnsembleCheckSam2::indicator:checked {
                background-color: #10b981;
                border-color: #34d399;
            }
            QFrame#EnsembleSep {
                background-color: #334155;
                width: 1px;
                max-width: 1px;
            }
            QComboBox#ModelCombo {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 12px;
                font-weight: 600;
                color: #ffffff;
                min-width: 200px;
            }
            QComboBox#ModelCombo QAbstractItemView {
                background-color: #111827;
                border: 1px solid #334155;
                color: #ffffff;
                selection-background-color: #312e81;
                padding: 4px;
            }
            QLabel#ModelBadge {
                background-color: #312e81;
                border: 1px solid #4338ca;
                border-radius: 6px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 600;
                color: #a5b4fc;
            }
            QPushButton#YoloWeightsBtn {
                background-color: #1e293b;
                border: 1px solid #6366f1;
                border-radius: 6px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 600;
                color: #c7d2fe;
                max-width: 140px;
            }
            QPushButton#YoloWeightsBtn:hover {
                background-color: #312e81;
                color: #ffffff;
            }
            QLabel#ParamTitle {
                font-size: 10px;
                font-weight: 600;
                color: #94a3b8;
                background: transparent;
            }
            QLabel#ParamVal {
                font-size: 11px;
                font-weight: 700;
                color: #e2e8f0;
                min-width: 26px;
                background: transparent;
            }
            QSlider, QSlider#ConfSlider, QSlider#IouSlider {
                background: transparent;
                background-color: transparent;
                border: none;
                height: 18px;
            }
            QSlider#ConfSlider::groove:horizontal, QSlider#IouSlider::groove:horizontal {
                height: 4px;
                background: #1e293b;
                border-radius: 2px;
            }
            QSlider#ConfSlider::sub-page:horizontal, QSlider#IouSlider::sub-page:horizontal {
                background: #6366f1;
                border-radius: 2px;
            }
            QSlider#ConfSlider::handle:horizontal, QSlider#IouSlider::handle:horizontal {
                background: #ffffff;
                border: 2px solid #6366f1;
                width: 10px;
                height: 10px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 6px;
            }
            QSlider#ConfSlider::handle:horizontal:hover, QSlider#IouSlider::handle:horizontal:hover {
                background: #ffffff;
                border: 2px solid #818cf8;
            }
            QPushButton#PreviewMixTopBtn {
                background-color: #4f46e5;
                border: none;
                border-radius: 6px;
                color: #ffffff;
                font-size: 11px;
                font-weight: 700;
                padding: 5px 12px;
            }
            QPushButton#PreviewMixTopBtn:hover {
                background-color: #4338ca;
            }
            QPushButton#AITunerTopBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8b5cf6, stop:1 #ec4899);
                border: none;
                border-radius: 6px;
                color: #ffffff;
                font-size: 11px;
                font-weight: 700;
                padding: 5px 12px;
            }
            QPushButton#AITunerTopBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #db2777);
            }

            /* --- Stage Frame & View Switcher --- */
            QFrame#StageFrame {
                background-color: #0f172a;
                border: 1px solid #1f293d;
                border-radius: 10px;
            }
            QFrame#ViewSwitcherFrame {
                background-color: #111827;
                border-bottom: 1px solid #1f293d;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
            }
            QPushButton#ViewModeBtn {
                background: transparent;
                border: 1px solid #283654;
                border-radius: 5px;
                color: #94a3b8;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 10px;
            }
            QPushButton#ViewModeBtn:hover {
                background-color: #1e293b;
                border-color: #6366f1;
                color: #ffffff;
            }
            QPushButton#ViewModeBtnActive {
                background-color: #312e81;
                border: 1px solid #6366f1;
                border-radius: 5px;
                color: #ffffff;
                font-size: 11px;
                font-weight: 700;
                padding: 4px 10px;
            }
            QPushButton#SampleTabBtn {
                background: transparent;
                border: 1px solid #283654;
                border-radius: 4px;
                color: #94a3b8;
                font-size: 10px;
                font-weight: 600;
                padding: 3px 10px;
            }
            QPushButton#SampleTabBtn:hover {
                background-color: #1e293b;
                border-color: #6366f1;
                color: #ffffff;
            }
            QPushButton#SampleTabBtnActive {
                background-color: #4338ca;
                border: 1px solid #818cf8;
                border-radius: 4px;
                color: #ffffff;
                font-size: 10px;
                font-weight: 700;
                padding: 3px 10px;
            }

            /* --- 2x2 Grid Cards --- */
            QFrame#PreviewCard {
                background-color: #111827;
                border: 1px solid #1f293d;
                border-radius: 8px;
            }
            QFrame#PreviewCard:hover {
                border-color: #6366f1;
                background-color: #151d30;
            }
            QLabel#CardTitle {
                font-size: 11px;
                font-weight: 600;
                color: #f1f5f9;
                background: transparent;
            }
            QLabel#CardBadge {
                background-color: #312e81;
                color: #a5b4fc;
                font-size: 10px;
                font-weight: 700;
                border-radius: 4px;
                padding: 2px 6px;
            }
            QPushButton#CardZoomBtn {
                background: transparent;
                border: none;
                font-size: 11px;
                font-weight: 600;
                color: #94a3b8;
                padding: 2px 6px;
            }
            QPushButton#CardZoomBtn:hover {
                color: #ffffff;
            }

            /* --- Bottom Approval Actions Bar --- */
            QFrame#PrevBottomFrame {
                background-color: #111827;
                border-top: 1px solid #1f293d;
                border-bottom-left-radius: 10px;
                border-bottom-right-radius: 10px;
            }
            QLabel#ResultStatsLabel {
                font-size: 12px;
                font-weight: 600;
                color: #cbd5e1;
                background: transparent;
                border: none;
                padding: 0px;
            }
            QPushButton#ApplyBtn {
                background-color: #10b981;
                border: none;
                border-radius: 6px;
                color: #ffffff;
                font-size: 11px;
                font-weight: 700;
                padding: 6px 14px;
            }
            QPushButton#ApplyBtn:hover {
                background-color: #059669;
            }
            QPushButton#AutoLabelPrimaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #8b5cf6);
                border: none;
                border-radius: 6px;
                color: #ffffff;
                font-size: 11px;
                font-weight: 700;
                padding: 6px 14px;
            }
            QPushButton#AutoLabelPrimaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #7c3aed);
            }
            QProgressBar#InlineProgressBar {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 3px;
                height: 5px;
            }
            QProgressBar#InlineProgressBar::chunk {
                background-color: #8b5cf6;
                border-radius: 2px;
            }
            """
        )

    def _on_model_changed(self) -> None:
        if getattr(self, "_syncing_models", False):
            return
        self._syncing_models = True
        try:
            mode_val = self.model_combo.currentData()
            mode = AutoLabelPipelineMode(mode_val)
            self.model_badge.setText(mode.badge_label)

            if hasattr(self, "sam2_chk"):
                self.sam2_chk.setChecked(mode.produces_masks)

            if not mode.is_ensemble:
                if hasattr(self, "dino_chk"):
                    self.dino_chk.setChecked(
                        mode in (AutoLabelPipelineMode.DINO_SAM2_MASKS, AutoLabelPipelineMode.DINO_BOXES)
                    )
                if hasattr(self, "yolo_chk"):
                    self.yolo_chk.setChecked(
                        mode in (AutoLabelPipelineMode.YOLO_SAM2_MASKS, AutoLabelPipelineMode.YOLO_BOXES)
                    )
                if hasattr(self, "florence_chk"):
                    self.florence_chk.setChecked(
                        mode in (AutoLabelPipelineMode.VLM_SAM2_MASKS, AutoLabelPipelineMode.VLM_BOXES)
                    )
            else:
                active_count = sum(
                    1
                    for chk in (self.dino_chk, self.yolo_chk, self.florence_chk)
                    if hasattr(self, "dino_chk") and chk.isChecked()
                )
                if active_count < 2 and hasattr(self, "dino_chk") and hasattr(self, "yolo_chk"):
                    self.dino_chk.setChecked(True)
                    self.yolo_chk.setChecked(True)
                    active_count = 2
                self.model_badge.setText(f"⚡ Fused ({active_count} models)")

            if hasattr(self, "yolo_weights_btn"):
                self.yolo_weights_btn.setVisible(
                    self.yolo_chk.isChecked() if hasattr(self, "yolo_chk") else mode.uses_yolo
                )
        except Exception:
            pass
        finally:
            self._syncing_models = False

    def _on_detector_toggled(self) -> None:
        if getattr(self, "_syncing_models", False):
            return
        self._syncing_models = True
        try:
            enabled_detectors = sum(
                1
                for chk in (self.dino_chk, self.yolo_chk, self.florence_chk)
                if chk.isChecked()
            )
            if enabled_detectors == 0:
                self.dino_chk.setChecked(True)
                enabled_detectors = 1

            if hasattr(self, "yolo_weights_btn"):
                self.yolo_weights_btn.setVisible(self.yolo_chk.isChecked())

            sam2_enabled = self.sam2_chk.isChecked()

            if enabled_detectors > 1:
                target_mode = (
                    AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS
                    if sam2_enabled
                    else AutoLabelPipelineMode.ENSEMBLE_FUSION_BOXES
                )
                self.model_badge.setText(f"⚡ Fused ({enabled_detectors} models)")
            else:
                if self.dino_chk.isChecked():
                    target_mode = (
                        AutoLabelPipelineMode.DINO_SAM2_MASKS
                        if sam2_enabled
                        else AutoLabelPipelineMode.DINO_BOXES
                    )
                elif self.yolo_chk.isChecked():
                    target_mode = (
                        AutoLabelPipelineMode.YOLO_SAM2_MASKS
                        if sam2_enabled
                        else AutoLabelPipelineMode.YOLO_BOXES
                    )
                else:
                    target_mode = (
                        AutoLabelPipelineMode.VLM_SAM2_MASKS
                        if sam2_enabled
                        else AutoLabelPipelineMode.VLM_BOXES
                    )
                self.model_badge.setText("Mask labels" if sam2_enabled else "Box labels")

            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == target_mode.value:
                    self.model_combo.setCurrentIndex(i)
                    break
        finally:
            self._syncing_models = False

    def _on_sam2_toggled(self) -> None:
        if getattr(self, "_syncing_models", False):
            return
        self._on_detector_toggled()

    def _update_yolo_button_ui(self) -> None:
        count = len(getattr(self, "_active_yolo_models", []))
        if count <= 0:
            self._active_yolo_models = ["yolo11n.pt"]
            count = 1

        if count == 1:
            name = Path(self._active_yolo_models[0]).name
            self.yolo_weights_btn.setText(f"📦 {name}")
        elif count == 2:
            n1 = Path(self._active_yolo_models[0]).name
            n2 = Path(self._active_yolo_models[1]).name
            self.yolo_weights_btn.setText(f"⚡ 2 YOLO: {n1} + {n2}")
        else:
            n1 = Path(self._active_yolo_models[0]).name
            self.yolo_weights_btn.setText(f"⚡ 3 YOLO Models ({n1} + {count - 1})")

        details = "\n".join(f"• Model {i+1}: {m}" for i, m in enumerate(self._active_yolo_models))
        self.yolo_weights_btn.setToolTip(
            f"Active YOLO Models ({count}/3 simultaneous):\n{details}\n\nClick to add, remove, or swap models."
        )

    def _open_multi_yolo_menu(self) -> None:
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e222d;
                color: #e0e0e0;
                border: 1px solid #3b4252;
                border-radius: 6px;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px 6px 12px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #29b6f6;
                color: #000000;
            }
            QMenu::separator {
                height: 1px;
                background-color: #3b4252;
                margin: 4px 6px;
            }
        """)

        header_action = menu.addAction(f"── Active YOLO Models ({len(self._active_yolo_models)}/3) ──")
        header_action.setEnabled(False)

        for i, model in enumerate(self._active_yolo_models):
            display = Path(model).name
            sub_menu = menu.addMenu(f"📦 Model {i+1}: {display}")

            replace_action = sub_menu.addAction("📂 Change Weights (*.pt)...")
            replace_action.triggered.connect(lambda _, idx=i: self._replace_yolo_model(idx))

            presets_menu = sub_menu.addMenu("⚡ Switch to Preset...")
            for preset in ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8m.pt", "yolov8x.pt"]:
                p_act = presets_menu.addAction(preset)
                p_act.triggered.connect(lambda _, idx=i, p=preset: self._set_yolo_model_preset(idx, p))

            if len(self._active_yolo_models) > 1:
                remove_action = sub_menu.addAction("🗑 Remove Model")
                remove_action.triggered.connect(lambda _, idx=i: self._remove_yolo_model(idx))

        menu.addSeparator()

        if len(self._active_yolo_models) < 3:
            add_menu = menu.addMenu("➕ Add Another YOLO Model (up to 3)...")
            for preset in ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8m.pt", "yolov8x.pt"]:
                if preset not in self._active_yolo_models:
                    act = add_menu.addAction(f"⚡ {preset}")
                    act.triggered.connect(lambda _, p=preset: self._add_yolo_model(p))

            custom_act = add_menu.addAction("📂 + Custom Weights (*.pt)...")
            custom_act.triggered.connect(self._add_custom_yolo_weights)
        else:
            limit_act = menu.addAction("⚠️ Maximum 3 YOLO models reached")
            limit_act.setEnabled(False)

        menu.addSeparator()
        reset_act = menu.addAction("🔄 Reset to Default (yolo11n.pt)")
        reset_act.triggered.connect(self._reset_yolo_models)

        menu.exec(self.yolo_weights_btn.mapToGlobal(QPoint(0, self.yolo_weights_btn.height())))

    def _add_yolo_model(self, model: str) -> None:
        if len(self._active_yolo_models) < 3 and model not in self._active_yolo_models:
            self._active_yolo_models.append(model)
            self._update_yolo_button_ui()
            if not self.yolo_chk.isChecked():
                self.yolo_chk.setChecked(True)

    def _add_custom_yolo_weights(self) -> None:
        if len(self._active_yolo_models) >= 3:
            QMessageBox.information(self, "Limit Reached", "You can run up to 3 YOLO models simultaneously.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Pretrained YOLO Weights",
            "",
            "YOLO Weights (*.pt *.onnx *.engine);;All Files (*)",
        )
        if file_path:
            path = Path(file_path)
            try:
                from ultralytics import YOLO

                loaded_yolo = YOLO(str(path))
                if hasattr(self.engine, "_yolo_detectors") and isinstance(self.engine._yolo_detectors, dict):
                    self.engine._yolo_detectors[str(path)] = loaded_yolo
                    self.engine._yolo_detectors[path.name] = loaded_yolo
                self.engine._yolo_detector = loaded_yolo
                self.engine._yolo_model_name = path.name
                self._add_yolo_model(str(path))
            except Exception as err:
                LOGGER.exception("Failed to load custom YOLO model weights")
                QMessageBox.critical(
                    self,
                    "Loading Error",
                    f"Failed to load YOLO model from {path.name}:\n{err}",
                )

    def _replace_yolo_model(self, index: int) -> None:
        if index < 0 or index >= len(self._active_yolo_models):
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Pretrained YOLO Weights",
            "",
            "YOLO Weights (*.pt *.onnx *.engine);;All Files (*)",
        )
        if file_path:
            path = Path(file_path)
            try:
                from ultralytics import YOLO

                loaded_yolo = YOLO(str(path))
                if hasattr(self.engine, "_yolo_detectors") and isinstance(self.engine._yolo_detectors, dict):
                    self.engine._yolo_detectors[str(path)] = loaded_yolo
                    self.engine._yolo_detectors[path.name] = loaded_yolo
                self.engine._yolo_detector = loaded_yolo
                self.engine._yolo_model_name = path.name
                self._active_yolo_models[index] = str(path)
                self._update_yolo_button_ui()
            except Exception as err:
                LOGGER.exception("Failed to load custom YOLO model weights")
                QMessageBox.critical(
                    self,
                    "Loading Error",
                    f"Failed to load YOLO model from {path.name}:\n{err}",
                )

    def _set_yolo_model_preset(self, index: int, preset: str) -> None:
        if 0 <= index < len(self._active_yolo_models):
            self._active_yolo_models[index] = preset
            self._update_yolo_button_ui()

    def _remove_yolo_model(self, index: int) -> None:
        if 0 <= index < len(self._active_yolo_models) and len(self._active_yolo_models) > 1:
            self._active_yolo_models.pop(index)
            self._update_yolo_button_ui()

    def _reset_yolo_models(self) -> None:
        self._active_yolo_models = ["yolo11n.pt"]
        self._update_yolo_button_ui()

    def _choose_custom_yolo_weights(self) -> None:
        """Single model custom weights loader (legacy & testing compatibility)."""
        self._replace_yolo_model(0)

    def _rebuild_class_cards(self) -> None:
        for card in self._class_cards:
            self.classes_vbox.removeWidget(card)
            card.deleteLater()
        self._class_cards.clear()

        for cls_item in self.classes:
            card = ClassCardWidget(cls_item, self.classes_container)
            card.deleted.connect(self._remove_class_card)
            self.classes_vbox.insertWidget(self.classes_vbox.count() - 1, card)
            self._class_cards.append(card)

        self.class_count_badge.setText(str(len(self.classes)))

    def _add_new_class(self) -> None:
        palette_idx = len(self.classes) % len(CLASS_PALETTE)
        new_color = CLASS_PALETTE[palette_idx]
        new_cls = AutoLabelClass(
            name=f"class_{len(self.classes) + 1}",
            prompt="",
            color=new_color,
        )
        self.classes.append(new_cls)
        self._rebuild_class_cards()

    def _remove_class_card(self, card: ClassCardWidget) -> None:
        if card in self._class_cards:
            self.classes.remove(card.class_item)
            self.classes_vbox.removeWidget(card)
            self._class_cards.remove(card)
            card.deleteLater()
            self.class_count_badge.setText(str(len(self.classes)))

    def _clear_all_classes(self) -> None:
        self.classes.clear()
        self._rebuild_class_cards()

    def _get_current_config(self) -> AutoLabelConfig:
        mode_val = self.model_combo.currentData()
        mode = AutoLabelPipelineMode(mode_val) if mode_val else AutoLabelPipelineMode.DINO_SAM2_MASKS
        conf = float(self.conf_slider.value()) / 100.0
        iou = float(self.iou_slider.value()) / 100.0 if hasattr(self, "iou_slider") else 0.45

        dino_enabled = self.dino_chk.isChecked() if hasattr(self, "dino_chk") else True
        yolo_enabled = self.yolo_chk.isChecked() if hasattr(self, "yolo_chk") else False
        florence_enabled = self.florence_chk.isChecked() if hasattr(self, "florence_chk") else False
        sam2_enabled = self.sam2_chk.isChecked() if hasattr(self, "sam2_chk") else True

        active_yolos = list(getattr(self, "_active_yolo_models", ["yolo11n.pt"]))
        if not active_yolos:
            active_yolos = ["yolo11n.pt"]

        return AutoLabelConfig(
            mode=mode,
            confidence_threshold=conf,
            text_threshold=max(0.15, conf - 0.10),
            box_iou_threshold=iou,
            classes=self.classes,
            yolo_model_name=active_yolos[0],
            yolo_models=active_yolos[:3],
            enable_grounding_dino=dino_enabled,
            enable_yolo=yolo_enabled,
            enable_florence2=florence_enabled,
            enable_sam2_masks=sam2_enabled,
        )

    def _run_single_preview(self, wait: bool = False) -> None:
        """Run preview across the active sample images (all 4 in Grid View, or current image only in Focused View)."""
        if self._is_previewing:
            return

        if not self.preview_image_paths:
            if self.image_paths:
                self.preview_image_paths = random.sample(self.image_paths, min(4, len(self.image_paths)))
                self.current_image_path = self.preview_image_paths[0]
            else:
                QMessageBox.warning(self, "No Images", "Please select at least one image from the batch.")
                return

        if not self.classes:
            QMessageBox.warning(self, "No Classes", "Please add at least one class before previewing.")
            return

        is_focused_view = self.preview_views_stack.currentIndex() == 1
        if is_focused_view and self.current_image_path:
            target_images = [self.current_image_path]
        else:
            target_images = list(self.preview_image_paths)

        self._is_previewing = True
        config = self._get_current_config()

        self.inline_progress.setVisible(True)
        self.inline_progress.setRange(0, len(target_images))
        self.inline_progress.setValue(0)
        self.result_stats_label.setText(f"⚡ Running preview on {len(target_images)} image(s)...")

        self._preview_thread = AutoLabelPreviewThread(
            engine=self.engine,
            images=target_images,
            config=config,
            parent=self,
        )
        self._preview_thread.sample_processed.connect(self._on_preview_sample_processed)
        self._preview_thread.preview_finished.connect(self._on_preview_finished)
        self._preview_thread.preview_failed.connect(self._on_preview_failed)
        self._preview_thread.start()

        if wait or not self.isVisible():
            self._preview_thread.wait(15000)
            from PySide6.QtWidgets import QApplication

            QApplication.processEvents()

    def _on_preview_sample_processed(
        self, cur: int, total: int, img_path: Path, result: AutoLabelResult
    ) -> None:
        self._latest_results[img_path] = result
        if img_path in self.preview_image_paths:
            card_idx = self.preview_image_paths.index(img_path)
            if card_idx < len(self.preview_cards):
                self.preview_cards[card_idx].set_data(img_path, result)
        self.inline_progress.setValue(cur)
        if total == 1:
            self.result_stats_label.setText(
                f"⚡ Running {self.model_combo.currentText()} on {img_path.name[:24]}..."
            )
        else:
            self.result_stats_label.setText(
                f"⚡ Processed sample {cur}/{total}: {img_path.name[:20]}..."
            )

    def _on_preview_finished(self, results: dict[Path, AutoLabelResult]) -> None:
        self._latest_results.update(results)
        is_focused_view = self.preview_views_stack.currentIndex() == 1

        total_detections = sum(r.count for r in results.values())
        total_time = sum(r.elapsed_seconds for r in results.values())
        class_counts: dict[str, int] = {}
        for r in results.values():
            for det in r.detections:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

        if self.current_image_path and self.current_image_path in self._latest_results:
            self._latest_result = self._latest_results[self.current_image_path]
        elif self.preview_image_paths:
            self.current_image_path = self.preview_image_paths[0]
            self._latest_result = self._latest_results.get(self.current_image_path)

        if self.current_image_path and self._latest_result:
            self.preview_canvas.set_result(self.current_image_path, self._latest_result)

        if class_counts:
            breakdown = ", ".join(
                f"{count} {c}{'s' if count > 1 else ''}" for c, count in class_counts.items()
            )
            if is_focused_view and self.current_image_path:
                summary_str = f"Found {total_detections} objects ({breakdown}) on {self.current_image_path.name[:20]} in {total_time:.2f}s"
            else:
                summary_str = f"Found {total_detections} objects ({breakdown}) in {total_time:.2f}s"
        else:
            if is_focused_view and self.current_image_path:
                summary_str = f"Found 0 objects on {self.current_image_path.name[:20]} ({total_time:.2f}s)"
            else:
                summary_str = f"Found 0 objects across {len(results)} samples ({total_time:.2f}s)"

        self.result_stats_label.setText(f"⚡ {summary_str}")
        self._update_sample_tabs_ui()
        self._is_previewing = False
        self.inline_progress.setVisible(False)

    def _on_preview_failed(self, error_msg: str) -> None:
        self._is_previewing = False
        self.inline_progress.setVisible(False)
        LOGGER.exception("Preview generation failed: %s", error_msg)
        QMessageBox.critical(self, "Preview Error", f"Failed to generate preview: {error_msg}")
        self.result_stats_label.setText("⚡ Preview error occurred.")

    def _open_ai_tuner_dialog(self) -> None:
        """Open the interactive AI Auto-Tuner modal."""
        if not self.preview_image_paths:
            QMessageBox.warning(self, "No Images", "Please load at least one sample image to tune.")
            return

        from app.ui.dialogs.ai_tuner_dialog import AITunerDialog

        dialog = AITunerDialog(
            sample_images=self.preview_image_paths,
            ground_truth=self.ground_truth,
            current_config=self._get_current_config(),
            engine=self.engine,
            parent=self,
        )
        dialog.tuning_applied.connect(self._apply_tuned_config)
        dialog.exec()

    def _apply_tuned_config(self, tuned_config: AutoLabelConfig) -> None:
        """Apply tuned prompts and hyperparameters to the UI and re-run preview."""
        self.classes = [
            AutoLabelClass(c.name, c.prompt, c.color, c.enabled)
            for c in tuned_config.classes
        ]
        self._rebuild_class_cards()

        conf_val = int(round(tuned_config.confidence_threshold * 100))
        iou_val = int(round(tuned_config.box_iou_threshold * 100))
        self.conf_slider.setValue(conf_val)
        self.iou_slider.setValue(iou_val)

        self.result_stats_label.setText("✨ AI Tuned Settings applied! Re-evaluating sample preview...")
        self._run_single_preview()

    def _apply_preview_to_image(self) -> None:
        """Approve and apply annotations from the preview.
        In Focused View: applies ONLY to the currently focused sample image.
        In 4-Grid View: applies to all previewed sample images.
        """
        try:
            if not self._latest_results and not self._latest_result:
                QMessageBox.information(
                    self,
                    "No Preview Ready",
                    "Please click '⚡ Preview Mix' first to preview before approving.",
                )
                return

            from app.services.annotation.domain import (
                TARGET_CLASSES,
                Annotation,
                AnnotationDocument,
                AnnotationSource,
                BoundingBox,
            )
            from app.services.auto_label.engine import compute_box_iou

            is_focused_view = self.preview_views_stack.currentIndex() == 1

            if is_focused_view:
                if not self.current_image_path:
                    return
                res = self._latest_results.get(self.current_image_path, self._latest_result)
                if res is None:
                    QMessageBox.information(
                        self,
                        "No Preview",
                        f"No preview result available for {self.current_image_path.name}.",
                    )
                    return
                results_to_apply = {self.current_image_path: res}
            else:
                results_to_apply = dict(self._latest_results)
                if not results_to_apply and self._latest_result and self.current_image_path:
                    results_to_apply[self.current_image_path] = self._latest_result

            if not results_to_apply:
                QMessageBox.information(
                    self,
                    "No Preview Ready",
                    "Please click '⚡ Preview Mix' first to preview before approving.",
                )
                return

            updated_docs: dict[Path, AnnotationDocument] = {}
            total_applied_boxes = 0

            for img_path, res in results_to_apply.items():
                existing_doc = self.ground_truth.get(img_path)
                existing_anns = list(existing_doc.annotations) if existing_doc else []
                new_anns = list(existing_anns)

                for det in res.detections:
                    if det.class_name not in TARGET_CLASSES:
                        continue
                    # Safely validate and clamp bounding box
                    try:
                        box = det.box
                        left = max(0.0, min(1.0, float(box.left)))
                        top = max(0.0, min(1.0, float(box.top)))
                        right = max(0.0, min(1.0, float(box.right)))
                        bottom = max(0.0, min(1.0, float(box.bottom)))
                        if left >= right or top >= bottom:
                            continue
                        safe_box = BoundingBox(left, top, right, bottom)
                    except Exception:
                        continue

                    # Avoid duplicate annotations
                    if any(
                        ex.class_name == det.class_name
                        and compute_box_iou(ex.box, safe_box) >= 0.50
                        for ex in new_anns
                    ):
                        continue

                    conf = det.confidence
                    if conf is not None:
                        conf = max(0.0, min(1.0, float(conf)))

                    source = (
                        AnnotationSource.SAM2
                        if det.polygon_normalized
                        else AnnotationSource.GROUNDING_DINO
                    )
                    new_anns.append(
                        Annotation(
                            class_name=det.class_name,
                            box=safe_box,
                            confidence=conf,
                            source=source,
                        )
                    )
                    total_applied_boxes += 1

                w = res.image_width or (existing_doc.image_width if existing_doc else 640)
                h = res.image_height or (existing_doc.image_height if existing_doc else 480)
                try:
                    with Image.open(img_path) as im:
                        w, h = im.width, im.height
                except Exception:
                    pass

                updated_doc = AnnotationDocument(
                    image_path=img_path,
                    image_width=w,
                    image_height=h,
                    annotations=tuple(new_anns),
                )
                updated_docs[img_path] = updated_doc

            # Update in-memory ground truth dictionary in dialog
            self.ground_truth.update(updated_docs)

            # Notify MainWindow to update its dataset documents and canvas
            self.batch_completed.emit(updated_docs)

            # Emit preview_applied for current image if available
            if self.current_image_path in results_to_apply:
                self.preview_applied.emit(results_to_apply[self.current_image_path])

            if is_focused_view or len(results_to_apply) == 1:
                target_name = (
                    self.current_image_path.name
                    if self.current_image_path
                    else next(iter(results_to_apply.keys())).name
                )
                self.result_stats_label.setText(
                    f"✔ Applied {total_applied_boxes} annotations to {target_name}!"
                )
                QMessageBox.information(
                    self,
                    "Applied to Current Image",
                    f"Successfully applied {total_applied_boxes} annotation(s) to:\n"
                    f"{target_name}\n\n"
                    "Saved as Ground Truth reference for AI Auto-Tuning.",
                )
            else:
                self.result_stats_label.setText(
                    f"✔ Applied {total_applied_boxes} annotations to {len(updated_docs)} sample images!"
                )
                QMessageBox.information(
                    self,
                    "Applied to Samples",
                    f"Successfully applied {total_applied_boxes} annotation(s) across "
                    f"{len(updated_docs)} sample image(s)!\n\n"
                    "These samples are now stored as Ground Truth references for AI Auto-Tuning.",
                )
        except Exception as err:
            LOGGER.exception("Failed to apply preview annotations: %s", err)
            QMessageBox.critical(
                self,
                "Apply Error",
                f"An error occurred while applying annotations:\n{err}",
            )

    def _run_batch_auto_label(self, wait: bool = False) -> None:
        """Run full batch auto-annotation across every image in the dataset."""
        if not self.image_paths:
            QMessageBox.warning(self, "Empty Batch", "No images in current batch.")
            return

        if not self.classes:
            QMessageBox.warning(self, "No Classes", "Please add at least one class before annotating.")
            return

        confirm = QMessageBox.question(
            self,
            "Confirm Auto Label Batch",
            f"Run {self.model_combo.currentText()} on all {len(self.image_paths)} images in this batch?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        config = self._get_current_config()
        self.result_stats_label.setText(f"⚡ Auto Labeling all {len(self.image_paths)} images...")
        self.inline_progress.setVisible(True)
        self.inline_progress.setRange(0, len(self.image_paths))
        self.inline_progress.setValue(0)

        docs: list[AnnotationDocument] = []
        for p in self.image_paths:
            try:
                with Image.open(p) as img:
                    w, h = img.width, img.height
                docs.append(AnnotationDocument(image_path=p, image_width=w, image_height=h))
            except Exception:
                docs.append(AnnotationDocument(image_path=p, image_width=640, image_height=640))

        self._batch_thread = AutoLabelBatchThread(
            engine=self.engine,
            documents=docs,
            config=config,
            parent=self,
        )
        self._batch_thread.batch_progress.connect(self._on_batch_progress_update)
        self._batch_thread.batch_finished.connect(self._on_batch_finished)
        self._batch_thread.batch_failed.connect(self._on_batch_failed)
        self._batch_thread.start()

        if wait or not self.isVisible():
            self._batch_thread.wait(60000)
            from PySide6.QtWidgets import QApplication

            QApplication.processEvents()

    def _on_batch_progress_update(self, cur: int, tot: int, p: Path, res: AutoLabelResult) -> None:
        self.inline_progress.setValue(cur)
        self.result_stats_label.setText(f"⚡ Processing {cur}/{tot}: {p.name[:20]}...")

    def _on_batch_finished(self, updated_docs: dict[Path, AnnotationDocument]) -> None:
        self.inline_progress.setVisible(False)
        self.batch_completed.emit(updated_docs)
        QMessageBox.information(
            self,
            "Auto Label Complete",
            f"Successfully annotated {len(updated_docs)} images with {self.model_combo.currentText()}.",
        )
        self.accept()

    def _on_batch_failed(self, error_msg: str) -> None:
        self.inline_progress.setVisible(False)
        LOGGER.exception("Batch Auto Label failed: %s", error_msg)
        QMessageBox.critical(self, "Batch Error", f"Failed to run batch auto label: {error_msg}")
