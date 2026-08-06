"""Command-based undo and redo for annotation edits."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from app.services.annotation.domain import Annotation, AnnotationDocument

LOGGER = logging.getLogger(__name__)


class AnnotationCommand(Protocol):
    """A reversible mutation of an annotation document."""

    def execute(self, document: AnnotationDocument) -> AnnotationDocument: ...

    def undo(self, document: AnnotationDocument) -> AnnotationDocument: ...


@dataclass(frozen=True, slots=True)
class AddAnnotationCommand:
    """Command that adds one annotation."""

    annotation: Annotation

    def execute(self, document: AnnotationDocument) -> AnnotationDocument:
        return document.add(self.annotation)

    def undo(self, document: AnnotationDocument) -> AnnotationDocument:
        return document.remove(self.annotation.annotation_id)


@dataclass(frozen=True, slots=True)
class RemoveAnnotationCommand:
    """Command that removes one annotation."""

    annotation: Annotation

    def execute(self, document: AnnotationDocument) -> AnnotationDocument:
        return document.remove(self.annotation.annotation_id)

    def undo(self, document: AnnotationDocument) -> AnnotationDocument:
        return document.add(self.annotation)


@dataclass(frozen=True, slots=True)
class UpdateAnnotationCommand:
    """Command that replaces one annotation and retains its previous value."""

    previous: Annotation
    updated: Annotation

    def execute(self, document: AnnotationDocument) -> AnnotationDocument:
        return document.update(self.updated)

    def undo(self, document: AnnotationDocument) -> AnnotationDocument:
        return document.update(self.previous)


@dataclass(frozen=True, slots=True)
class ReplaceDocumentCommand:
    """Command that replaces a complete document as one reversible edit."""

    previous: AnnotationDocument
    updated: AnnotationDocument

    def execute(self, document: AnnotationDocument) -> AnnotationDocument:
        return self.updated

    def undo(self, document: AnnotationDocument) -> AnnotationDocument:
        return self.previous


class AnnotationHistory:
    """Bounded in-memory command history for one active document."""

    def __init__(self, document: AnnotationDocument, max_size: int = 100) -> None:
        if max_size < 1:
            raise ValueError("max_size must be greater than zero")
        self._document = document
        self._max_size = max_size
        self._undo_stack: list[AnnotationCommand] = []
        self._redo_stack: list[AnnotationCommand] = []

    @property
    def document(self) -> AnnotationDocument:
        """Return the current immutable document state."""
        return self._document

    @property
    def can_undo(self) -> bool:
        """Whether an edit can be undone."""
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        """Whether an edit can be redone."""
        return bool(self._redo_stack)

    def execute(self, command: AnnotationCommand) -> AnnotationDocument:
        """Apply a command and clear redo history after a new edit."""
        self._document = command.execute(self._document)
        self._undo_stack.append(command)
        self._undo_stack = self._undo_stack[-self._max_size :]
        self._redo_stack.clear()
        LOGGER.debug("annotation command executed: %s", type(command).__name__)
        return self._document

    def undo(self) -> AnnotationDocument:
        """Undo the latest edit, raising ``IndexError`` when none exists."""
        if not self._undo_stack:
            raise IndexError("no annotation edit to undo")
        command = self._undo_stack.pop()
        self._document = command.undo(self._document)
        self._redo_stack.append(command)
        LOGGER.debug("annotation command undone: %s", type(command).__name__)
        return self._document

    def redo(self) -> AnnotationDocument:
        """Redo the latest undone edit, raising ``IndexError`` when none exists."""
        if not self._redo_stack:
            raise IndexError("no annotation edit to redo")
        command = self._redo_stack.pop()
        self._document = command.execute(self._document)
        self._undo_stack.append(command)
        LOGGER.debug("annotation command redone: %s", type(command).__name__)
        return self._document
