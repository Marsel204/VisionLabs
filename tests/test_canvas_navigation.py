"""Tests for canvas panning, zooming, and tool modes."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from app.services.annotation.domain import Annotation, AnnotationDocument, BoundingBox
from app.ui.canvas.annotation_canvas import AnnotationCanvas, CanvasMode


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Ensure a single QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def sample_document(tmp_path: Path) -> AnnotationDocument:
    """Create a temporary image file and an AnnotationDocument."""
    img_path = tmp_path / "test_image.png"
    image = QImage(800, 600, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    image.save(str(img_path))
    box = BoundingBox(0.2, 0.2, 0.6, 0.6)
    ann = Annotation(class_name="car", box=box)
    return AnnotationDocument(
        image_path=img_path,
        image_width=800,
        image_height=600,
        annotations=(ann,),
    )


def test_canvas_initial_mode_and_switching(qapp: QApplication) -> None:
    """Canvas should start in DRAW mode and allow toggling to PAN mode."""
    canvas = AnnotationCanvas()
    assert canvas.mode == CanvasMode.DRAW

    modes_emitted: list[CanvasMode] = []
    canvas.mode_changed.connect(modes_emitted.append)

    canvas.set_pan_mode()
    assert canvas.mode == CanvasMode.PAN
    assert modes_emitted == [CanvasMode.PAN]

    canvas.set_draw_mode()
    assert canvas.mode == CanvasMode.DRAW
    assert modes_emitted == [CanvasMode.PAN, CanvasMode.DRAW]


def test_canvas_zoom_and_reset_view(qapp: QApplication, sample_document: AnnotationDocument) -> None:
    """Zoom in, zoom out, actual size, and reset view should adjust zoom levels."""
    canvas = AnnotationCanvas()
    canvas.resize(400, 300)
    canvas.set_document(sample_document)
    assert canvas._zoom == 1.0

    canvas.zoom_in(1.5)
    assert canvas._zoom == pytest.approx(1.5)

    canvas.zoom_out(1.5)
    assert canvas._zoom == pytest.approx(1.0)

    canvas.zoom_actual_size()
    assert canvas._zoom == pytest.approx(1.0)

    canvas.zoom_in(2.0)
    assert canvas._zoom == pytest.approx(2.0)
    canvas.reset_view()
    assert canvas._zoom == pytest.approx(1.0)


def test_canvas_pan_by_delta(qapp: QApplication, sample_document: AnnotationDocument) -> None:
    """pan_by should adjust horizontal and vertical scrollbars."""
    canvas = AnnotationCanvas()
    canvas.resize(200, 200)
    canvas.set_document(sample_document)
    canvas.zoom_in(3.0)

    h_init = canvas.horizontalScrollBar().value()
    v_init = canvas.verticalScrollBar().value()

    canvas.pan_by(30, 40)
    assert canvas.horizontalScrollBar().value() == h_init + 30
    assert canvas.verticalScrollBar().value() == v_init + 40


def test_middle_mouse_drag_pans_canvas(qapp: QApplication, sample_document: AnnotationDocument) -> None:
    """Pressing and dragging with MiddleButton should pan the canvas smoothly."""
    canvas = AnnotationCanvas()
    canvas.resize(200, 200)
    canvas.set_document(sample_document)
    canvas.zoom_in(3.0)

    h_start = canvas.horizontalScrollBar().value()
    v_start = canvas.verticalScrollBar().value()

    # Mouse press with MiddleButton
    press_event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(100, 100),
        QPointF(100, 100),
        Qt.MouseButton.MiddleButton,
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mousePressEvent(press_event)
    assert canvas._panning is True
    assert canvas._pan_button == Qt.MouseButton.MiddleButton

    # Mouse move
    move_event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(70, 60),
        QPointF(70, 60),
        Qt.MouseButton.MiddleButton,
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseMoveEvent(move_event)
    # Dragging by (-30, -40) relative to start should scroll (+30, +40)
    assert canvas.horizontalScrollBar().value() == h_start + 30
    assert canvas.verticalScrollBar().value() == v_start + 40

    # Mouse release
    release_event = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(70, 60),
        QPointF(70, 60),
        Qt.MouseButton.MiddleButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseReleaseEvent(release_event)
    assert canvas._panning is False
    assert canvas._pan_button is None


def test_right_mouse_drag_pans_canvas(qapp: QApplication, sample_document: AnnotationDocument) -> None:
    """Pressing and dragging with RightButton should pan the canvas."""
    canvas = AnnotationCanvas()
    canvas.resize(200, 200)
    canvas.set_document(sample_document)
    canvas.zoom_in(3.0)

    h_start = canvas.horizontalScrollBar().value()
    v_start = canvas.verticalScrollBar().value()

    press_event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(100, 100),
        QPointF(100, 100),
        Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mousePressEvent(press_event)
    assert canvas._panning is True
    assert canvas._pan_button == Qt.MouseButton.RightButton

    move_event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(80, 80),
        QPointF(80, 80),
        Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseMoveEvent(move_event)
    assert canvas.horizontalScrollBar().value() == h_start + 20
    assert canvas.verticalScrollBar().value() == v_start + 20

    release_event = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(80, 80),
        QPointF(80, 80),
        Qt.MouseButton.RightButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseReleaseEvent(release_event)
    assert canvas._panning is False


def test_spacebar_quick_pan(qapp: QApplication, sample_document: AnnotationDocument) -> None:
    """Holding Spacebar enables quick-pan mode on left click drag."""
    canvas = AnnotationCanvas()
    canvas.resize(200, 200)
    canvas.set_document(sample_document)
    canvas.zoom_in(3.0)

    # Press spacebar
    key_press = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Space,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.keyPressEvent(key_press)
    assert canvas._space_pressed is True
    assert canvas.viewport().cursor().shape() == Qt.CursorShape.OpenHandCursor

    # Left click drag while space is held
    h_start = canvas.horizontalScrollBar().value()
    v_start = canvas.verticalScrollBar().value()

    mouse_press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(100, 100),
        QPointF(100, 100),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mousePressEvent(mouse_press)
    assert canvas._panning is True
    assert canvas.viewport().cursor().shape() == Qt.CursorShape.ClosedHandCursor

    mouse_move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(50, 50),
        QPointF(50, 50),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseMoveEvent(mouse_move)
    assert canvas.horizontalScrollBar().value() == h_start + 50
    assert canvas.verticalScrollBar().value() == v_start + 50

    mouse_release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(50, 50),
        QPointF(50, 50),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseReleaseEvent(mouse_release)
    assert canvas._panning is False

    # Release spacebar
    key_release = QKeyEvent(
        QEvent.Type.KeyRelease,
        Qt.Key.Key_Space,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.keyReleaseEvent(key_release)
    assert canvas._space_pressed is False


def test_pan_tool_mode_left_drag(qapp: QApplication, sample_document: AnnotationDocument) -> None:
    """When in PAN mode, left-click drag pans instead of creating/moving bounding boxes."""
    canvas = AnnotationCanvas()
    canvas.resize(200, 200)
    canvas.set_document(sample_document)
    canvas.zoom_in(3.0)
    canvas.set_pan_mode()

    h_start = canvas.horizontalScrollBar().value()
    v_start = canvas.verticalScrollBar().value()

    mouse_press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(100, 100),
        QPointF(100, 100),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mousePressEvent(mouse_press)
    assert canvas._panning is True

    mouse_move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(60, 70),
        QPointF(60, 70),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseMoveEvent(mouse_move)
    assert canvas.horizontalScrollBar().value() == h_start + 40
    assert canvas.verticalScrollBar().value() == v_start + 30

    mouse_release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(60, 70),
        QPointF(60, 70),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseReleaseEvent(mouse_release)
    assert canvas._panning is False


def test_keyboard_arrow_pan(qapp: QApplication, sample_document: AnnotationDocument) -> None:
    """Shift + Arrow keys should pan the canvas."""
    canvas = AnnotationCanvas()
    canvas.resize(200, 200)
    canvas.set_document(sample_document)
    canvas.zoom_in(3.0)

    h_start = canvas.horizontalScrollBar().value()
    v_start = canvas.verticalScrollBar().value()

    # Shift + Right
    canvas.keyPressEvent(QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Right,
        Qt.KeyboardModifier.ShiftModifier,
    ))
    assert canvas.horizontalScrollBar().value() == h_start + 50

    # Shift + Down
    canvas.keyPressEvent(QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Down,
        Qt.KeyboardModifier.ShiftModifier,
    ))
    assert canvas.verticalScrollBar().value() == v_start + 50


def test_edge_cursor_detection() -> None:
    """_cursor_for_edge should map edges to appropriate resize cursor shapes."""
    canvas = AnnotationCanvas()
    assert canvas._cursor_for_edge("l") == Qt.CursorShape.SizeHorCursor
    assert canvas._cursor_for_edge("r") == Qt.CursorShape.SizeHorCursor
    assert canvas._cursor_for_edge("t") == Qt.CursorShape.SizeVerCursor
    assert canvas._cursor_for_edge("b") == Qt.CursorShape.SizeVerCursor
    assert canvas._cursor_for_edge("lt") == Qt.CursorShape.SizeFDiagCursor
    assert canvas._cursor_for_edge("rb") == Qt.CursorShape.SizeFDiagCursor
    assert canvas._cursor_for_edge("rt") == Qt.CursorShape.SizeBDiagCursor
    assert canvas._cursor_for_edge("lb") == Qt.CursorShape.SizeBDiagCursor
