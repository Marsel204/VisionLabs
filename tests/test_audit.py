import pytest
from PIL import Image

from app.services.annotation.domain import BoundingBox
from app.services.audit import (
    AuditBox,
    FindingKind,
    Prediction,
    audit_annotations,
    estimate_occlusion,
)

W, H = 1000, 1000


def box(left: float, top: float, right: float, bottom: float) -> BoundingBox:
    return BoundingBox(left, top, right, bottom)


def existing(box_id: str, cls: str, b: BoundingBox, **kwargs) -> AuditBox:
    return AuditBox(box_id, cls, b, **kwargs)


def kinds(report) -> list[FindingKind]:
    return [finding.kind for finding in report.findings]


def test_lower_box_occludes_the_one_behind_it() -> None:
    behind = existing("behind", "motorcycle", box(0.30, 0.30, 0.50, 0.60))
    front = existing("front", "car", box(0.25, 0.40, 0.55, 0.80))
    estimates = estimate_occlusion([behind, front])
    assert estimates["behind"].fraction == pytest.approx(0.667, abs=0.05)
    assert estimates["behind"].occluder_ids == ("front",)
    assert estimates["front"].fraction == 0.0


def test_side_by_side_neighbours_do_not_occlude() -> None:
    first = existing("a", "motorcycle", box(0.1, 0.4, 0.3, 0.6))
    second = existing("b", "motorcycle", box(0.25, 0.4, 0.45, 0.6))
    estimates = estimate_occlusion([first, second])
    assert estimates["a"].fraction == 0.0
    assert estimates["b"].fraction == 0.0


def test_matching_detection_produces_no_findings() -> None:
    labels = [existing("a", "motorcycle", box(0.2, 0.2, 0.4, 0.5))]
    preds = [Prediction("motorcycle", box(0.21, 0.2, 0.41, 0.5), 0.9)]
    assert audit_annotations(labels, preds, W, H).findings == ()


def test_unlabelled_detection_is_reported_as_missed() -> None:
    labels = [existing("a", "motorcycle", box(0.1, 0.1, 0.2, 0.3))]
    preds = [
        Prediction("motorcycle", box(0.1, 0.1, 0.2, 0.3), 0.9),
        Prediction("motorcycle", box(0.6, 0.5, 0.7, 0.7), 0.6),
    ]
    report = audit_annotations(labels, preds, W, H)
    assert kinds(report) == [FindingKind.POSSIBLY_MISSED]
    assert report.findings[0].box == box(0.6, 0.5, 0.7, 0.7)


def test_occluded_candidate_uses_relaxed_confidence_and_is_flagged() -> None:
    front = existing("front", "car", box(0.25, 0.40, 0.55, 0.80))
    hidden = Prediction("motorcycle", box(0.30, 0.30, 0.50, 0.60), 0.16)
    far_away = Prediction("motorcycle", box(0.80, 0.10, 0.90, 0.30), 0.16)
    report = audit_annotations(
        [front],
        [Prediction("car", front.box, 0.9), hidden, far_away],
        W, H,
    )
    assert kinds(report) == [FindingKind.POSSIBLY_MISSED]
    finding = report.findings[0]
    assert finding.box == hidden.box
    assert finding.likely_occluded
    assert finding.occlusion > 0.5


def test_smaller_object_inside_a_larger_same_class_box_is_not_redundant() -> None:
    big = existing("big", "motorcycle", box(0.2, 0.2, 0.6, 0.7))
    inner = Prediction("motorcycle", box(0.25, 0.25, 0.35, 0.40), 0.7)
    report = audit_annotations([big], [Prediction("motorcycle", big.box, 0.9), inner], W, H)
    assert kinds(report) == [FindingKind.POSSIBLY_MISSED]


def test_second_prediction_on_a_labelled_object_is_not_missed() -> None:
    labels = [existing("a", "motorcycle", box(0.2, 0.2, 0.4, 0.5))]
    preds = [
        Prediction("motorcycle", box(0.2, 0.2, 0.4, 0.5), 0.9),
        Prediction("motorcycle", box(0.21, 0.21, 0.41, 0.51), 0.6),
    ]
    assert audit_annotations(labels, preds, W, H).findings == ()


def test_weakly_overlapping_same_class_detection_is_misalignment_not_missed() -> None:
    labels = [existing("a", "motorcycle", box(0.20, 0.20, 0.40, 0.50))]
    preds = [Prediction("motorcycle", box(0.30, 0.25, 0.50, 0.55), 0.8)]
    report = audit_annotations(labels, preds, W, H)
    assert kinds(report) == [FindingKind.MISALIGNED]
    assert report.findings[0].annotation_id == "a"
    assert report.findings[0].box == preds[0].box


def test_class_mismatch_suggests_detector_class() -> None:
    labels = [existing("a", "car", box(0.2, 0.2, 0.4, 0.5))]
    preds = [Prediction("motorcycle", box(0.2, 0.2, 0.4, 0.5), 0.85)]
    report = audit_annotations(labels, preds, W, H)
    assert kinds(report) == [FindingKind.CLASS_MISMATCH]
    assert report.findings[0].class_name == "motorcycle"


def test_box_without_detector_support_is_unsupported() -> None:
    labels = [
        existing("a", "motorcycle", box(0.1, 0.1, 0.2, 0.3), source="yolo", confidence=0.4),
        existing("b", "motorcycle", box(0.6, 0.6, 0.7, 0.8)),
    ]
    preds = [Prediction("motorcycle", box(0.6, 0.6, 0.7, 0.8), 0.9)]
    report = audit_annotations(labels, preds, W, H)
    assert kinds(report) == [FindingKind.UNSUPPORTED]
    assert report.findings[0].annotation_id == "a"


def test_heavily_occluded_box_is_not_flagged_unsupported() -> None:
    hidden = existing("hidden", "motorcycle", box(0.30, 0.30, 0.50, 0.60), source="yolo")
    front = existing("front", "car", box(0.20, 0.30, 0.60, 0.80))
    preds = [Prediction("car", front.box, 0.9)]
    report = audit_annotations([hidden, front], preds, W, H)
    assert FindingKind.UNSUPPORTED not in kinds(report)


def test_no_predictions_gives_no_detector_based_findings() -> None:
    labels = [existing("a", "motorcycle", box(0.1, 0.1, 0.2, 0.3))]
    assert audit_annotations(labels, [], W, H).findings == ()


def test_duplicate_flags_the_lower_confidence_box() -> None:
    labels = [
        existing("keep", "motorcycle", box(0.2, 0.2, 0.4, 0.5), confidence=0.9),
        existing("drop", "motorcycle", box(0.205, 0.2, 0.405, 0.5), confidence=0.5),
    ]
    report = audit_annotations(labels, [], W, H)
    assert kinds(report) == [FindingKind.DUPLICATE]
    assert report.findings[0].annotation_id == "drop"


def test_occluded_neighbour_inside_bigger_box_is_not_a_duplicate() -> None:
    labels = [
        existing("big", "motorcycle", box(0.2, 0.2, 0.6, 0.7)),
        existing("small", "motorcycle", box(0.25, 0.25, 0.35, 0.40)),
    ]
    assert audit_annotations(labels, [], W, H).findings == ()


def test_tiny_box_is_implausible() -> None:
    labels = [existing("a", "motorcycle", box(0.5, 0.5, 0.502, 0.502))]
    assert kinds(audit_annotations(labels, [], W, H)) == [FindingKind.IMPLAUSIBLE_SHAPE]


def test_findings_are_ranked_by_severity() -> None:
    labels = [existing("a", "motorcycle", box(0.1, 0.1, 0.2, 0.3), source="yolo", confidence=0.4)]
    preds = [
        Prediction("motorcycle", box(0.6, 0.5, 0.7, 0.7), 0.35),
        Prediction("motorcycle", box(0.8, 0.5, 0.9, 0.7), 0.9),
        Prediction("motorcycle", box(0.4, 0.5, 0.5, 0.7), 0.5),
    ]
    severities = [f.severity for f in audit_annotations(labels, preds, W, H).findings]
    assert severities == sorted(severities, reverse=True)
    assert len(severities) == 4


class _Tensor(list):
    def tolist(self):
        return list(self)


class _FakeBox:
    def __init__(self, cls_id: int, conf: float, xyxy: tuple[float, float, float, float]):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [_Tensor(xyxy)]


class _FakeModel:
    names = {0: "motor", 1: "person"}

    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def __call__(self, image, **_kwargs):
        self.calls.append(image.size)
        width, height = image.size
        boxes = [_FakeBox(0, 0.5, (100, 100, 180, 220)), _FakeBox(1, 0.9, (10, 10, 40, 90))]
        if (width, height) != (2000, 1000):
            boxes.append(_FakeBox(0, 0.4, (0, 5, width - 1, height - 1)))  # cut by tile edge
        return [type("Result", (), {"boxes": boxes})()]


def test_tile_grid_covers_the_image_and_skips_small_images() -> None:
    from app.services.audit.detector import tile_grid

    assert tile_grid(600, 500, 640, 0.25) == []
    tiles = tile_grid(2000, 1000, 640, 0.25)
    assert max(t[2] for t in tiles) == 2000
    assert max(t[3] for t in tiles) == 1000
    assert all(t[2] - t[0] <= 640 and t[3] - t[1] <= 640 for t in tiles)


def test_detect_reference_normalizes_classes_filters_and_merges_tiles() -> None:
    from app.services.audit.detector import detect_reference

    model = _FakeModel()
    predictions = detect_reference(
        model, Image.new("RGB", (2000, 1000)), allowed_classes=["motorcycle"], views=("original",)
    )
    assert len(model.calls) > 1
    assert {p.class_name for p in predictions} == {"motorcycle"}
    # The same motorcycle appears in several windows at different offsets: only
    # offset-identical duplicates merge, and tile-edge-cut boxes are dropped.
    assert all(p.box.width < 0.5 for p in predictions)
    assert max(p.confidence for p in predictions) == 0.5


class _LeftOnlyModel:
    """Sees a motorcycle only in the left half of whatever image it is given."""

    names = {0: "motorcycle"}

    def __call__(self, image, **_kwargs):
        return [type("Result", (), {"boxes": [_FakeBox(0, 0.6, (10, 20, 30, 60))]})()]


def test_flip_view_maps_detections_back_to_original_coordinates() -> None:
    from app.services.audit.detector import detect_reference

    image = Image.new("RGB", (100, 80))
    predictions = detect_reference(_LeftOnlyModel(), image, views=("original", "flip"))
    lefts = sorted(round(p.box.left * 100) for p in predictions)
    rights = sorted(round(p.box.right * 100) for p in predictions)
    assert lefts == [10, 70]
    assert rights == [30, 90]


def test_default_views_layer_a_flip_pass_and_unknown_views_are_rejected() -> None:
    from app.services.audit.detector import DEFAULT_VIEWS, detect_reference

    assert DEFAULT_VIEWS == ("original", "flip")
    with pytest.raises(ValueError):
        detect_reference(_LeftOnlyModel(), Image.new("RGB", (100, 80)), views=("sepia",))
    with pytest.raises(ValueError):
        detect_reference(_LeftOnlyModel(), Image.new("RGB", (100, 80)), views=())


def test_photometric_views_run_the_model_on_an_enhanced_copy() -> None:
    from app.services.audit.detector import detect_reference

    seen: list[Image.Image] = []

    class Recorder(_LeftOnlyModel):
        def __call__(self, image, **kwargs):
            seen.append(image)
            return super().__call__(image, **kwargs)

    image = Image.effect_noise((64, 64), 40).convert("RGB")
    detect_reference(Recorder(), image, views=("original", "sharpen"))
    assert len(seen) == 2 and seen[0] is image and seen[1] is not image
