"""Locate Anything's structured output and auto-label integration contracts."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from PIL import Image

from app.models.contracts import ModelRuntimeError
from app.services.annotation.domain import AnnotationDocument, AnnotationSource, BoundingBox
from app.services.auto_label.engine import AutoLabelEngine
from app.services.auto_label.models import AutoLabelClass, AutoLabelConfig, AutoLabelPipelineMode
from app.services.inference.cloud import InferenceOptions
from app.services.inference.locate_anything import LocateAnythingDetector
from pipeline_bridge import BoxPixel


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "image.jpg"
    Image.new("RGB", (640, 480)).save(path)
    return path


def locate_config(**overrides):
    return replace(
        AutoLabelConfig(
            mode=AutoLabelPipelineMode.LOCATE_ANYTHING_BOXES,
            classes=[AutoLabelClass("car", "red sedan", "#112233")],
            enable_grounding_dino=False,
            enable_yolo=False,
            enable_sam2_masks=False,
            enable_locate_anything=True,
        ),
        **overrides,
    )


def test_parse_structured_boxes_and_clip_to_image():
    answer = (
        "<ref>car</ref><box><100><200><700><800></box>"
        "<box><-20><0><1200><1000></box>"
        "<box><700><200><100><800></box>"
        "<box><300><300><300><900></box>"
        "<box><0><1001><500><1100></box>"
        "<box><100><200></box><box>broken</box>"
    )
    assert LocateAnythingDetector.parse_boxes(answer) == [
        BoundingBox(.1, .2, .7, .8), BoundingBox(0, 0, 1, 1)
    ]
    assert LocateAnythingDetector.parse_boxes("None") == []


def test_detect_uses_custom_processor_and_preserves_class_association(monkeypatch):
    import torch

    detector = LocateAnythingDetector(device="cpu")
    model = Mock()
    model.generate.side_effect = [
        "<box><100><200><700><800></box>",
        ("<box><200><300><600><900></box>", [], {}),
    ]
    processor = Mock()
    processor.process_vision_info.return_value = ([Image.new("RGB", (64, 48))], None)
    inputs = {"pixel_values": torch.zeros(1), "input_ids": torch.ones((1, 2), dtype=torch.long),
              "attention_mask": torch.ones((1, 2)), "image_grid_hws": torch.ones((1, 2))}
    batch = Mock()
    batch.to.return_value = inputs
    processor.return_value = batch
    detector._model = model
    detector._processor = processor
    detector._tokenizer = Mock()
    detector._dtype = torch.float32
    boxes = detector.detect(Image.new("RGB", (64, 48)), ["red sedan", "delivery truck"])
    assert boxes == [(0, BoundingBox(.1, .2, .7, .8)), (1, BoundingBox(.2, .3, .6, .9))]
    questions = [call.args[0][0]["content"][1]["text"]
                 for call in processor.py_apply_chat_template.call_args_list]
    assert "red sedan" in questions[0] and "delivery truck" in questions[1]
    kwargs = model.generate.call_args.kwargs
    assert kwargs["tokenizer"] is detector._tokenizer
    assert kwargs["generation_mode"] == "hybrid" and kwargs["use_cache"] is True
    assert kwargs["max_new_tokens"] == 8192
    assert kwargs["pixel_values"].dtype == torch.float32


def test_loading_is_lazy_pinned_and_reused(monkeypatch):
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    from app.services.inference.locate_anything import MODEL_ID, MODEL_REVISION

    loaders = [Mock() for _ in range(3)]
    for cls, loader in zip((AutoModel, AutoProcessor, AutoTokenizer), loaders, strict=True):
        monkeypatch.setattr(cls, "from_pretrained", loader)
    detector = LocateAnythingDetector(device="cpu")
    assert all(not loader.called for loader in loaders)
    detector._ensure_loaded()
    detector._ensure_loaded()
    for loader in loaders:
        loader.assert_called_once()
        assert loader.call_args.args == (MODEL_ID,)
        assert loader.call_args.kwargs["revision"] == MODEL_REVISION
        assert loader.call_args.kwargs["trust_remote_code"] is True


def test_single_detector_skips_other_models_and_disabled_classes(image_path):
    detector = Mock()
    detector.detect.return_value = [(0, BoundingBox(.1, .2, .7, .8))]
    engine = AutoLabelEngine(locate_anything_detector=detector, allow_cloud=False)
    engine._get_grounding_detector = Mock(side_effect=AssertionError("unselected DINO"))
    engine._get_yolo_detector = Mock(side_effect=AssertionError("unselected YOLO"))
    config = locate_config(classes=[AutoLabelClass("car", "red sedan", "#112233"),
                                    AutoLabelClass("truck", enabled=False)])
    result = engine.run_preview(image_path, config)
    assert detector.detect.call_args.kwargs["prompts"] == ["red sedan"]
    assert result.count == 1 and result.detections[0].class_name == "car"
    assert result.detections[0].box == BoundingBox(.1, .2, .7, .8)
    assert result.detections[0].color == "#112233"
    document = AnnotationDocument(image_path, 640, 480)
    annotations = engine.run_batch([document], config)[image_path].annotations
    assert annotations[0].source == AnnotationSource.LOCATE_ANYTHING


def test_ensemble_fuses_same_class_proposals(image_path):
    locate = Mock()
    locate.detect.return_value = [(0, BoundingBox(.1, .2, .7, .8))]
    dino = Mock()
    dino.detect.return_value = [("red sedan", 0, .9, BoxPixel(64, 96, 448, 384))]
    engine = AutoLabelEngine(grounding_detector=dino, locate_anything_detector=locate,
                             allow_cloud=False)
    result = engine.run_preview(image_path, locate_config(
        mode=AutoLabelPipelineMode.ENSEMBLE_FUSION_BOXES, enable_grounding_dino=True))
    assert result.count == 1
    dino.detect.assert_called_once()
    locate.detect.assert_called_once()


def test_locate_boxes_are_passed_to_sam(image_path):
    import numpy as np

    locate, sam, polygons = Mock(), Mock(), Mock()
    locate.detect.return_value = [(0, BoundingBox(.1, .2, .7, .8))]
    sam.segment_boxes.return_value = [np.ones((480, 640), dtype=np.uint8)]
    polygons.mask_to_polygons.return_value = ([], [])
    engine = AutoLabelEngine(locate_anything_detector=locate, sam_segmenter=sam,
                             polygon_processor=polygons, allow_cloud=False)
    result = engine.run_preview(image_path, locate_config(
        mode=AutoLabelPipelineMode.LOCATE_ANYTHING_SAM2_MASKS, enable_sam2_masks=True))
    assert result.count == 1
    assert sam.segment_boxes.call_args.args[1] == [BoxPixel(64, 96, 448, 384)]


def test_selected_locate_failure_is_reported(image_path):
    detector = Mock()
    detector.detect.side_effect = ModelRuntimeError("Locate Anything could not load")
    engine = AutoLabelEngine(locate_anything_detector=detector, allow_cloud=False)
    with pytest.raises(ModelRuntimeError, match="Locate Anything"):
        engine.run_preview(image_path, locate_config())


def test_cloud_and_colab_preserve_selection():
    from app.services.colab_batch import make_config

    config = locate_config()
    assert InferenceOptions.from_config(config).to_config().enable_locate_anything is True
    assert InferenceOptions.from_config(config).to_config().mode == config.mode
    for pipeline, masks in [("locate-boxes", False), ("locate-masks", True)]:
        colab = make_config(["car"], pipeline=pipeline)
        assert colab.enable_locate_anything is True
        assert colab.enable_grounding_dino is False and colab.enable_yolo is False
        assert colab.mode.produces_masks is masks


def test_prompt_tuning_keeps_locate_selected():
    from app.services.ai_tuner.tuner_engine import AITunerEngine

    config = locate_config()
    tuner = AITunerEngine(auto_label_engine=Mock())
    assert tuner._update_config_thresholds(config, .4, .5).enable_locate_anything
    assert tuner._update_config_classes(config, [AutoLabelClass("truck")]).enable_locate_anything


def test_qt_selector_synchronizes_locate_mode(image_path, qapp):
    from app.ui.dialogs.auto_label_dialog import AutoLabelDialog

    dialog = AutoLabelDialog(image_paths=[image_path], engine=Mock(spec=AutoLabelEngine))
    index = dialog.model_combo.findData(AutoLabelPipelineMode.LOCATE_ANYTHING_BOXES.value)
    assert index >= 0
    dialog.model_combo.setCurrentIndex(index)
    config = dialog._get_current_config()
    assert config.enable_locate_anything and not config.enable_grounding_dino
    assert not config.enable_yolo and not config.enable_sam2_masks
    dialog.sam2_chk.setChecked(True)
    assert dialog._get_current_config().mode == AutoLabelPipelineMode.LOCATE_ANYTHING_SAM2_MASKS
    dialog.dino_chk.setChecked(True)
    assert dialog._get_current_config().mode.is_ensemble
    dialog.close()
