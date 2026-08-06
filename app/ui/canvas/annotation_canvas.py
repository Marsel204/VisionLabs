"""Interactive image canvas for viewing and navigating annotations."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView

from app.services.annotation.domain import Annotation, AnnotationDocument, BoundingBox
from app.services.fusion.fusion_models import FusionStatus


class AnnotationCanvas(QGraphicsView):
    """Zoomable and pannable annotation view."""

    box_created = Signal(BoundingBox)
    box_selected = Signal(object)
    box_resized = Signal(object, BoundingBox)
    box_deleted = Signal(object)
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
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
            item.setPen(QPen(QColor(color), 2))
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

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """Start drawing a box with the left mouse button."""
        if event.button() == Qt.MouseButton.LeftButton and self._image_item is not None:
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
        if self._drawing and self._preview is not None:
            point = self.mapToScene(event.position().toPoint())
            self._preview.setRect(QRectF(self._draw_start, point).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
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
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """Delete the selected box with Delete or Backspace."""
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self._selected is not None:
            annotation = self._selected
            self._selected = None
            self.box_deleted.emit(annotation.annotation_id)
            return
        super().keyPressEvent(event)

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

    def reset_view(self) -> None:
        """Fit the image to the available canvas."""
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = 1.0
