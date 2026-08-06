from pathlib import Path

from app.models.contracts import Detection
from app.services.active_learning.hard_examples import DifficultySignals, HardExampleManager
from app.services.annotation import AnnotationSource, BoundingBox
from app.services.dataset.index import DatasetIndex
from app.services.fusion.fusion import fuse_detections


def test_fusion_flags_model_disagreement() -> None:
    box = BoundingBox(0.1, 0.1, 0.5, 0.5)
    detections = [
        Detection("motorcycle", box, 0.95, AnnotationSource.YOLO),
        Detection("motorcycle", BoundingBox(0.11, 0.11, 0.51, 0.51), 0.4, AnnotationSource.SAM2),
    ]
    result = fuse_detections(detections)
    assert len(result) == 1
    assert result[0].disagreement


def test_dataset_index_prioritizes_hard_examples(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    easy = root / "easy.jpg"
    hard = root / "hard.jpg"
    easy.touch()
    hard.touch()
    with DatasetIndex(tmp_path / "index.sqlite") as index:
        assert index.scan(root) == 2
        manager = HardExampleManager(index)
        manager.record(easy, DifficultySignals(low_confidence=0.1))
        manager.record(hard, DifficultySignals(occlusion=1.0, missed_motorcycles=1.0))
        assert manager.prioritize(1) == [hard]
