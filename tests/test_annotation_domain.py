from pathlib import Path

import pytest

from app.services.annotation import (
    AddAnnotationCommand,
    Annotation,
    AnnotationDocument,
    AnnotationHistory,
    AnnotationSource,
    AnnotationValidationError,
    BoundingBox,
    RemoveAnnotationCommand,
    ReviewStatus,
    UpdateAnnotationCommand,
)


def make_document(tmp_path: Path) -> AnnotationDocument:
    image_path = tmp_path / "traffic.jpg"
    image_path.touch()
    return AnnotationDocument(image_path, image_width=1920, image_height=1080)


def make_annotation() -> Annotation:
    return Annotation(
        class_name="motorcycle",
        box=BoundingBox(0.1, 0.2, 0.4, 0.6),
        confidence=0.91,
        source=AnnotationSource.FUSED,
    )


def test_box_converts_to_yolo_coordinates() -> None:
    box = BoundingBox(0.1, 0.2, 0.4, 0.6)
    assert box.to_yolo() == pytest.approx((0.25, 0.4, 0.3, 0.4))


def test_invalid_annotation_is_rejected() -> None:
    with pytest.raises(AnnotationValidationError, match="unsupported target class"):
        Annotation("person", BoundingBox(0.1, 0.2, 0.4, 0.6))


def test_history_supports_add_undo_and_redo(tmp_path: Path) -> None:
    history = AnnotationHistory(make_document(tmp_path))
    annotation = make_annotation()

    history.execute(AddAnnotationCommand(annotation))
    assert history.document.annotations == (annotation,)
    assert history.can_undo

    history.undo()
    assert not history.document.annotations
    assert history.can_redo

    history.redo()
    assert history.document.annotations[0].review_status is ReviewStatus.PENDING


def test_new_edit_clears_redo_history(tmp_path: Path) -> None:
    history = AnnotationHistory(make_document(tmp_path))
    history.execute(AddAnnotationCommand(make_annotation()))
    history.undo()
    history.execute(AddAnnotationCommand(make_annotation()))
    assert not history.can_redo


def test_history_restores_deleted_annotation(tmp_path: Path) -> None:
    annotation = make_annotation()
    history = AnnotationHistory(make_document(tmp_path).add(annotation))

    history.execute(RemoveAnnotationCommand(annotation))
    history.undo()

    assert history.document.annotations == (annotation,)


def test_history_restores_previous_box_after_update(tmp_path: Path) -> None:
    annotation = make_annotation()
    history = AnnotationHistory(make_document(tmp_path).add(annotation))
    updated = annotation.modify(BoundingBox(0.2, 0.3, 0.5, 0.7))

    history.execute(UpdateAnnotationCommand(annotation, updated))
    history.undo()

    assert history.document.annotations == (annotation,)


def test_dataset_annotation_preserves_manual_boxes(tmp_path: Path) -> None:
    from app.ui.main_window import _DatasetAnnotationTask

    manual = make_annotation()
    prediction = Annotation(
        "motorcycle",
        BoundingBox(0.11, 0.21, 0.39, 0.59),
        confidence=0.8,
        source=AnnotationSource.YOLO,
    )
    image_path = tmp_path / "traffic.jpg"
    image_path.touch()
    document = AnnotationDocument(
        image_path,
        1920,
        1080,
        (manual,),
    )

    additions = _DatasetAnnotationTask._merge_predictions(document, [prediction])

    assert additions == []
