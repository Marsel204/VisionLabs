"""Virtualized-friendly image path browser widget."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QListWidget, QListWidgetItem


class ImageBrowser(QListWidget):
    """Paged image list that emits the selected path."""

    image_selected = Signal(Path)

    def __init__(self) -> None:
        super().__init__()
        self.itemSelectionChanged.connect(self._emit_selection)

    def set_paths(self, paths: list[Path]) -> None:
        """Replace visible paths without loading image pixels."""
        self.clear()
        for path in paths:
            item = QListWidgetItem(path.name)
            item.setData(256, str(path))
            self.addItem(item)

    def _emit_selection(self) -> None:
        item = self.currentItem()
        if item is not None:
            self.image_selected.emit(Path(item.data(256)))
