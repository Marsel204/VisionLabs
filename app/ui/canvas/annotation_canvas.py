"""Interactive image canvas for viewing, navigating, and annotating images."""

from __future__ import annotations

from enum import Enum, auto

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from app.services.annotation.domain import Annotation, AnnotationDocument, BoundingBox
from app.services.fusion.fusion_models import FusionStatus


class CanvasMode(Enum):
    """Active tool mode for the annotation canvas."""

    DRAW = auto()
    PAN = auto()


class AnnotationCanvas(QGraphicsView):
    """Zoomable and pannable annotation view with multi-mode navigation."""

    box_created = Signal(BoundingBox)
    box_selected = Signal(object)
    box_resized = Signal(object, BoundingBox)
    box_deleted = Signal(object)
    mode_changed = Signal(object)

    CLASS_COLORS = {
        "motorcycle": "#ff9800",
        "car": "#29b6f6",
        "bus": "#66bb6a",
        "truck": "#ef5350",
    }

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(QBrush(QColor("#111215")))
        self.setStyleSheet("border: none; background: #111215;")
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        self._mode = CanvasMode.DRAW
        self._panning = False
        self._pan_button: Qt.MouseButton | None = None
        self._last_pan_pos = QPointF()
        self._space_pressed = False

        self._zoom = 1.0
        self._image_item: QGraphicsPixmapItem | None = None
        self._document: AnnotationDocument | None = None
        self._drawing = False
        self._draw_start = QPointF()
        self._preview: QGraphicsRectItem | None = None
        self._annotation_items: list[tuple[QGraphicsRectItem, Annotation]] = []
        self._resizing: tuple[Annotation, QGraphicsRectItem, str, QRectF] | None = None
        self._moving: tuple[Annotation, QGraphicsRectItem, QPointF, QRectF] | None = None
        self._selected: Annotation | None = None
        self._fusion_statuses: dict[object, FusionStatus] = {}
        self._status_filter: set[FusionStatus] | None = None
        self._show_fusion_colors = True

    @property
    def mode(self) -> CanvasMode:
        """Return the current tool mode."""
        return self._mode

    def set_mode(self, mode: CanvasMode) -> None:
        """Set the active canvas mode (DRAW or PAN)."""
        if self._mode != mode:
            self._mode = mode
            self.mode_changed.emit(mode)
            self._update_hover_cursor()

    def set_draw_mode(self) -> None:
        """Switch to bounding box drawing mode."""
        self.set_mode(CanvasMode.DRAW)

    def set_pan_mode(self) -> None:
        """Switch to pan / hand tool mode."""
        self.set_mode(CanvasMode.PAN)

    def zoom_in(self, factor: float = 1.15) -> None:
        """Zoom into the canvas around the mouse cursor or center."""
        if self._zoom * factor <= 8.0:
            self._zoom *= factor
            self.scale(factor, factor)

    def zoom_out(self, factor: float = 1.15) -> None:
        """Zoom out of the canvas."""
        if self._zoom / factor >= 0.2:
            self._zoom /= factor
            self.scale(1 / factor, 1 / factor)

    def zoom_actual_size(self) -> None:
        """Reset zoom to 100% (1:1 pixel scale)."""
        self.resetTransform()
        self._zoom = 1.0
        if self._image_item is not None:
            self.centerOn(self._image_item)

    def pan_by(self, dx: int, dy: int) -> None:
        """Pan the canvas viewport by the specified delta in pixels."""
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + dx)
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + dy)

    def reset_view(self) -> None:
        """Fit the image to the available canvas."""
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = 1.0

    def set_document(self, document: AnnotationDocument) -> None:
        """Load one image and render its current boxes."""
        image = QImage(str(document.image_path))
        if image.isNull():
            raise ValueError(f"could not load image: {document.image_path}")
        self._document = document
        self._scene.clear()
        self._annotation_items.clear()
        self._selected = None
        self._image_item = self._scene.addPixmap(QPixmap.fromImage(image))
        self._image_item.setZValue(-1)
        for annotation in document.annotations:
            fusion_status = self._fusion_statuses.get(annotation.annotation_id)
            if self._status_filter is not None and fusion_status not in self._status_filter:
                continue
            box = annotation.box
            item = QGraphicsRectItem(
                box.left * document.image_width,
                box.top * document.image_height,
                box.width * document.image_width,
                box.height * document.image_height,
            )
            if fusion_status is None or not self._show_fusion_colors:
                color = self.CLASS_COLORS.get(annotation.class_name, "#4fc3f7")
            else:
                color = {
                    FusionStatus.ACCEPTED: "#43a047",
                    FusionStatus.NEEDS_REVIEW: "#ab47bc",
                    FusionStatus.CONFLICT: "#e53935",
                    FusionStatus.REJECTED: "#757575",
                }[fusion_status]
            qcol = QColor(color)
            item.setPen(QPen(qcol, 2))
            fill_col = QColor(qcol)
            fill_col.setAlpha(30)
            item.setBrush(QBrush(fill_col))
            status_text = fusion_status.value.replace("_", " ").title() if fusion_status else ""
            suffix = f" | {status_text}" if status_text else ""
            item.setToolTip(
                f"{annotation.class_name} ({annotation.confidence or 1.0:.2f}){suffix}"
            )
            self._scene.addItem(item)
            self._annotation_items.append((item, annotation))
        self._scene.setSceneRect(self._scene.itemsBoundingRect())
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 1.0
        self._update_hover_cursor()

    def set_fusion_statuses(self, statuses: dict[object, FusionStatus]) -> None:
        """Set fusion colors for the current document and repaint the canvas."""
        self._fusion_statuses = statuses
        if self._document is not None:
            self.set_document(self._document)

    def clear_fusion_statuses(self) -> None:
        """Return rendering to normal class colors and show all annotations."""
        self._fusion_statuses = {}
        self._status_filter = None

    def set_status_filter(self, statuses: set[FusionStatus] | None) -> None:
        """Show only annotations whose fusion status is included in ``statuses``."""
        self._status_filter = statuses
        if self._document is not None:
            self.set_document(self._document)

    def set_fusion_colors_enabled(self, enabled: bool) -> None:
        """Enable or disable status colors while retaining fusion metadata."""
        self._show_fusion_colors = enabled
        if self._document is not None:
            self.set_document(self._document)

    def _cursor_for_edge(self, edge: str) -> Qt.CursorShape:
        """Return the appropriate resize cursor for the given box edge."""
        if ("l" in edge and "t" in edge) or ("r" in edge and "b" in edge):
            return Qt.CursorShape.SizeFDiagCursor
        if ("r" in edge and "t" in edge) or ("l" in edge and "b" in edge):
            return Qt.CursorShape.SizeBDiagCursor
        if "l" in edge or "r" in edge:
            return Qt.CursorShape.SizeHorCursor
        if "t" in edge or "b" in edge:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.CrossCursor

    def _update_hover_cursor(self, view_point: QPointF | None = None) -> None:
        """Update viewport cursor according to current tool, space key, and hover target."""
        if self._panning:
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if self._space_pressed or self._mode == CanvasMode.PAN:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return

        if view_point is None:
            global_pos = self.cursor().pos()
            view_point = QPointF(self.viewport().mapFromGlobal(global_pos))

        point = self.mapToScene(view_point.toPoint())

        # In DRAW mode: inspect hover targets
        for item, _ in reversed(self._annotation_items):
            rect = item.rect()
            edge = self._resize_edge(rect, point)
            if edge:
                self.viewport().setCursor(self._cursor_for_edge(edge))
                return
            if rect.contains(point):
                self.viewport().setCursor(Qt.CursorShape.SizeAllCursor)
                return

        if self._image_item is not None and self._image_item.boundingRect().contains(point):
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """Handle mouse press for panning, resizing, moving, or drawing."""
        # 1. Middle-click or Right-click pan
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = True
            self._pan_button = event.button()
            self._last_pan_pos = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        # 2. Left-click
        if event.button() == Qt.MouseButton.LeftButton:
            # Space quick-pan or PAN tool mode
            if self._space_pressed or self._mode == CanvasMode.PAN:
                self._panning = True
                self._pan_button = Qt.MouseButton.LeftButton
                self._last_pan_pos = event.position()
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return

            # DRAW tool mode: select, resize, move, or draw
            if self._image_item is not None:
                point = self.mapToScene(event.position().toPoint())
                for item, annotation in reversed(self._annotation_items):
                    rect = item.rect()
                    edge = self._resize_edge(rect, point)
                    if edge:
                        self.setFocus()
                        self._selected = annotation
                        self.box_selected.emit(annotation.annotation_id)
                        self._resizing = (annotation, item, edge, rect)
                        return
                    if rect.contains(point):
                        self.setFocus()
                        self._selected = annotation
                        self.box_selected.emit(annotation.annotation_id)
                        self._moving = (annotation, item, point - rect.topLeft(), rect)
                        return
                if self._image_item.boundingRect().contains(point):
                    self._drawing = True
                    self._draw_start = point
                    self._preview = self._scene.addRect(
                        QRectF(point, point), QPen(QColor("#ffca28"), 2)
                    )
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """Handle mouse movement for active panning, moving, resizing, or drawing."""
        # Active pan
        if self._panning:
            current_pos = event.position()
            delta = current_pos - self._last_pan_pos
            self._last_pan_pos = current_pos
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() - delta.x())
            )
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() - delta.y())
            )
            event.accept()
            return

        # Active move
        if self._moving is not None:
            _, item, offset, _ = self._moving
            point = self.mapToScene(event.position().toPoint())
            rect = item.rect()
            left = point.x() - offset.x()
            top = point.y() - offset.y()
            if self._document is not None:
                left = max(0.0, min(left, self._document.image_width - rect.width()))
                top = max(0.0, min(top, self._document.image_height - rect.height()))
            item.setRect(QRectF(left, top, rect.width(), rect.height()))
            return

        # Active resize
        if self._resizing is not None:
            _, item, edge, original = self._resizing
            point = self.mapToScene(event.position().toPoint())
            rect = QRectF(original)
            if "l" in edge:
                rect.setLeft(min(point.x(), rect.right() - 3))
            if "r" in edge:
                rect.setRight(max(point.x(), rect.left() + 3))
            if "t" in edge:
                rect.setTop(min(point.y(), rect.bottom() - 3))
            if "b" in edge:
                rect.setBottom(max(point.y(), rect.top() + 3))
            if self._document is not None:
                rect = rect.intersected(
                    QRectF(0, 0, self._document.image_width, self._document.image_height)
                )
            item.setRect(rect)
            return

        # Active draw preview
        if self._drawing and self._preview is not None:
            point = self.mapToScene(event.position().toPoint())
            self._preview.setRect(QRectF(self._draw_start, point).normalized())
            return

        # Cursor hover update
        self._update_hover_cursor(event.position())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """Handle mouse release and finalize pan, move, resize, or box creation."""
        if self._panning and event.button() == self._pan_button:
            self._panning = False
            self._pan_button = None
            self._update_hover_cursor(event.position())
            event.accept()
            return

        if self._moving is not None:
            annotation, item, _, original = self._moving
            self._moving = None
            rect = item.rect()
            if self._document is not None and rect != original:
                self.box_resized.emit(annotation.annotation_id, BoundingBox(
                    rect.left() / self._document.image_width,
                    rect.top() / self._document.image_height,
                    rect.right() / self._document.image_width,
                    rect.bottom() / self._document.image_height,
                ))
            self._update_hover_cursor(event.position())
            return

        if self._resizing is not None:
            annotation, item, _, original = self._resizing
            self._resizing = None
            rect = item.rect()
            if self._document is not None and rect != original:
                self.box_resized.emit(annotation.annotation_id, BoundingBox(
                    rect.left() / self._document.image_width,
                    rect.top() / self._document.image_height,
                    rect.right() / self._document.image_width,
                    rect.bottom() / self._document.image_height,
                ))
            self._update_hover_cursor(event.position())
            return

        if self._drawing:
            self._drawing = False
            point = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._draw_start, point).normalized()
            if self._preview is not None:
                self._scene.removeItem(self._preview)
                self._preview = None
            if self._document is not None:
                bounds = QRectF(0, 0, self._document.image_width, self._document.image_height)
                rect = rect.intersected(bounds)
                if rect.width() > 2 and rect.height() > 2:
                    self.box_created.emit(BoundingBox(
                        rect.left() / self._document.image_width,
                        rect.top() / self._document.image_height,
                        rect.right() / self._document.image_width,
                        rect.bottom() / self._document.image_height,
                    ))
            self._update_hover_cursor(event.position())
            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """Handle Spacebar quick-pan, box deletion, and keyboard pan navigation."""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            if not self._panning:
                self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return

        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self._selected is not None:
            annotation = self._selected
            self._selected = None
            self.box_deleted.emit(annotation.annotation_id)
            return

        # Keyboard pan with Shift / Ctrl / Alt + Arrow keys
        if event.modifiers() & (
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
        ):
            step = 50
            if event.key() == Qt.Key.Key_Left:
                self.pan_by(-step, 0)
                event.accept()
                return
            if event.key() == Qt.Key.Key_Right:
                self.pan_by(step, 0)
                event.accept()
                return
            if event.key() == Qt.Key.Key_Up:
                self.pan_by(0, -step)
                event.accept()
                return
            if event.key() == Qt.Key.Key_Down:
                self.pan_by(0, step)
                event.accept()
                return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """Restore previous tool cursor upon Spacebar release."""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            if not self._panning:
                self._update_hover_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """Reset cursor when mouse leaves the canvas viewport."""
        if not self._panning:
            self.viewport().unsetCursor()
        super().leaveEvent(event)

    @staticmethod
    def _resize_edge(rect: QRectF, point: QPointF) -> str:
        """Return the box edges near a point, or an empty string."""
        tolerance = 8.0
        edge = ""
        if abs(point.x() - rect.left()) <= tolerance:
            edge += "l"
        if abs(point.x() - rect.right()) <= tolerance:
            edge += "r"
        if abs(point.y() - rect.top()) <= tolerance:
            edge += "t"
        if abs(point.y() - rect.bottom()) <= tolerance:
            edge += "b"
        return (
            edge
            if edge and rect.adjusted(-tolerance, -tolerance, tolerance, tolerance).contains(point)
            else ""
        )

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """Zoom around the cursor while preserving pan position."""
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        if 0.2 <= self._zoom * factor <= 8.0:
            self._zoom *= factor
            self.scale(factor, factor)
        event.accept()
