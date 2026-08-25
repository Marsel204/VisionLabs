from app.services.annotation.domain import Annotation, BoundingBox
from app.services.fusion import remove_overlapping_annotations
from app.services.inference.grounding import grounding_class, prompt_variants, tile_positions


def test_grounding_phrase_mapping() -> None:
    assert grounding_class("scooter") == "motorcycle"
    assert grounding_class("motorcycle") == "motorcycle"
    assert grounding_class("car") == "car"
    assert grounding_class("unknown object") is None


def test_prompt_variants_split_class_list() -> None:
    assert prompt_variants("motorcycle, car. bus") == [
        "motorcycle.",
        "car.",
        "bus.",
    ]


def test_tile_positions_cover_far_edge() -> None:
    assert tile_positions(1300, 640, 512) == [0, 512, 660]


def test_overlap_cleanup_preserves_distinct_classes() -> None:
    annotations = (
        Annotation("motorcycle", BoundingBox(0.2, 0.2, 0.7, 0.8), 0.9),
        Annotation("car", BoundingBox(0.25, 0.1, 0.65, 0.7), 0.8),
    )
    kept, removed = remove_overlapping_annotations(annotations)
    assert removed == 0
    assert len(kept) == 2

