"""Tests for FastAPI backend endpoints."""

from fastapi.testclient import TestClient
from app.api.server import app
from app.api import server
import pytest
from PIL import Image
from pathlib import Path
from types import SimpleNamespace
from app.services.dataset.index import DatasetIndex
from app.services.auto_label.models import AutoLabelResult, AutoLabelDetection
from app.services.annotation.domain import BoundingBox

client = TestClient(app, headers={"X-VisionLab-Token": server.SESSION_TOKEN})


@pytest.fixture(autouse=True)
def isolated_api_dataset(tmp_path, monkeypatch):
    """Exercise HTTP contracts with synthetic inputs and deterministic inference."""
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (100, 80)).save(image)
    index = DatasetIndex(tmp_path / ".dataset_index.sqlite")
    index.scan(tmp_path)
    monkeypatch.setattr(server, "ACTIVE_DATASET_DIR", tmp_path)
    monkeypatch.setattr(server, "DATASET_INDEX", index)
    class Detector:
        names = {0: "car"}
        def __call__(self, *args, **kwargs):
            pixel_box = SimpleNamespace(cls=[0], conf=[.9],
                xyxy=[SimpleNamespace(tolist=lambda: [10, 8, 30, 24])])
            return [SimpleNamespace(boxes=[pixel_box])]
    monkeypatch.setattr(server.AUTOLABEL_ENGINE, "_get_yolo_detector", lambda *args: Detector())
    monkeypatch.setattr(server, "get_yolo_model", lambda: None)
    monkeypatch.setattr(server.AUTOLABEL_ENGINE, "run_preview", lambda path, config:
        AutoLabelResult(Path(path), 100, 80, [AutoLabelDetection("car", .9, BoundingBox(.1, .1, .3, .3))]))
    monkeypatch.setattr("shutil.which", lambda name: None)  # No native dialogs in contract tests.
    yield
    index.close()


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert "classes" in data
    assert "motorcycle" in data["classes"]
    assert "database_connected" in data
    assert data["database_connected"] is True


def test_locate_anything_preview_and_batch_forward_selection(monkeypatch):
    from unittest.mock import Mock

    thread_factory = Mock()
    monkeypatch.setattr(server, "threading", SimpleNamespace(Thread=thread_factory))
    monkeypatch.setattr(server, "get_yolo_model", lambda: pytest.fail("Unselected YOLO loaded"))
    configs = []
    def preview(path, config):
        configs.append(config)
        return AutoLabelResult(Path(path), 100, 80, [
            AutoLabelDetection("car", 1.0, BoundingBox(.1, .2, .7, .8))])
    monkeypatch.setattr(server.AUTOLABEL_ENGINE, "run_preview", preview)
    options = {"classes": ["car"], "pipeline_mode": "locate_anything_boxes",
               "enable_locate_anything": True, "enable_grounding_dino": False,
               "enable_yolo": False, "enable_sam2_masks": False}
    response = client.post("/api/autolabel/preview", json={"image_name": "sample.jpg", **options})
    assert response.status_code == 200, response.text
    assert configs[-1].enable_locate_anything is True
    assert configs[-1].mode.value == "locate_anything_boxes"
    response = client.post("/api/autolabel/batch", json={"only_unannotated": False, **options})
    assert response.status_code == 200, response.text
    work = thread_factory.call_args.kwargs
    work["target"](*work["args"])
    assert len(configs) == 2 and configs[-1].enable_locate_anything is True
    annotations = client.get("/api/annotations/sample.jpg").json()
    assert annotations["boxes"][0]["source"] == "locate_anything"


def test_locate_anything_preview_reports_failure_without_yolo_fallback(monkeypatch):
    from app.models.contracts import ModelRuntimeError

    def fail(*args):
        raise ModelRuntimeError("Locate Anything could not load; install its dependencies")
    monkeypatch.setattr(server.AUTOLABEL_ENGINE, "run_preview", fail)
    monkeypatch.setattr(server, "detect_yolo", lambda *args: pytest.fail("Unexpected YOLO fallback"))
    response = client.post("/api/autolabel/preview", json={
        "image_name": "sample.jpg", "classes": ["car"], "enable_locate_anything": True,
        "enable_grounding_dino": False, "enable_yolo": False,
    })
    assert response.status_code == 503
    assert "Locate Anything" in response.json()["detail"]


def test_locate_anything_empty_preview_does_not_run_yolo(monkeypatch):
    monkeypatch.setattr(server.AUTOLABEL_ENGINE, "run_preview", lambda path, config:
                        AutoLabelResult(Path(path), 100, 80, []))
    monkeypatch.setattr(server, "get_yolo_model", lambda: object())
    monkeypatch.setattr(server, "detect_yolo", lambda *args: pytest.fail("Unexpected YOLO fallback"))
    response = client.post("/api/autolabel/preview", json={
        "image_name": "sample.jpg", "classes": ["car"], "enable_locate_anything": True,
        "enable_grounding_dino": False, "enable_yolo": False,
    })
    assert response.status_code == 200 and response.json()["count"] == 0


def test_colab_endpoints_require_local_session():
    unauthenticated = TestClient(app)
    assert unauthenticated.get("/api/colab/status").status_code == 401
    assert unauthenticated.post("/api/colab/connect").status_code == 401
    assert unauthenticated.post("/api/colab/switch-account").status_code == 401
    assert unauthenticated.post("/api/colab/account").status_code == 401


def test_colab_account_controls_use_the_local_connection(monkeypatch):
    calls = []
    monkeypatch.setattr(server.COLAB_CONNECTION, "switch_account", lambda: calls.append("switch"))
    monkeypatch.setattr(server.COLAB_CONNECTION, "refresh_account", lambda: calls.append("refresh"))
    assert client.post("/api/colab/account").status_code == 200
    assert client.post("/api/colab/switch-account").status_code == 200
    assert calls == ["refresh", "switch"]


def test_colab_job_selects_three_images_and_skips_existing_labels(tmp_path, monkeypatch):
    for index in range(5):
        Image.new("RGB", (100, 80)).save(tmp_path / f"image-{index}.jpg")
    server.DATASET_INDEX.scan(tmp_path)
    repo = server.AnnotationRepository(tmp_path)
    repo.save(tmp_path / "image-0.jpg", [], status="reviewed", expected_revision=0)
    selected = []
    monkeypatch.setattr(server.COLAB_CONNECTION, "start_job",
                        lambda root, images, *args, **kwargs: selected.extend(images))
    response = client.post("/api/colab/jobs", json={"dataset_id": str(tmp_path), "classes": ["car"]})
    assert response.status_code == 200
    assert len(selected) == 3
    assert tmp_path / "image-0.jpg" not in selected
    assert client.post("/api/colab/jobs", json={
        "dataset_id": "/another-dataset", "classes": ["car"],
    }).status_code == 409
    assert client.post("/api/colab/jobs", json={
        "dataset_id": str(tmp_path), "classes": ["car"], "limit": 101,
    }).status_code == 422


def test_images_list():
    res = client.get("/api/images")
    assert res.status_code == 200
    data = res.json()
    assert "images" in data
    assert data["total"] >= 1
    first = data["images"][0]
    assert "filename" in first
    assert "width" in first
    assert "height" in first
    assert "status" in first


@pytest.mark.parametrize("gpu", ["T4", "L4", "A100", "H100", "G4"])
def test_colab_api_accepts_supported_gpu_names(tmp_path, monkeypatch, gpu):
    selected = []
    monkeypatch.setattr(server.COLAB_CONNECTION, "start_job",
                        lambda *args, **kwargs: selected.append(kwargs["gpu"]))
    response = client.post("/api/colab/jobs", json={
        "dataset_id": str(tmp_path), "classes": ["car"], "gpu": gpu,
    })
    assert response.status_code == 200, response.text
    assert selected == [gpu]


def test_prompt_auto_refine():
    res = client.post("/api/prompt/auto-refine", json={"class_name": "motorcycle"})
    assert res.status_code == 200
    data = res.json()
    assert "refined_prompt" in data
    assert "motorcycle" in data["refined_prompt"].lower()


def test_yolo_detection():
    res_imgs = client.get("/api/images")
    img_name = res_imgs.json()["images"][0]["filename"]

    res = client.post("/api/detect/yolo", json={"image_name": img_name, "conf_threshold": 0.25})
    assert res.status_code == 200
    data = res.json()
    assert data["image_name"] == img_name
    assert "boxes" in data
    assert data["count"] > 0


def test_dataset_stats():
    res = client.get("/api/dataset/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_images" in data
    assert "reviewed" in data
    assert "unreviewed" in data
    assert "class_counts" in data
    assert "database_file" in data


def test_save_annotations_updates_sqlite_and_difficulty():
    res_imgs = client.get("/api/images")
    img_name = res_imgs.json()["images"][0]["filename"]

    boxes_payload = [
        {
            "id": "box-1",
            "class_name": "car",
            "class_id": 1,
            "confidence": 0.95,
            "x": 100.0,
            "y": 100.0,
            "width": 200.0,
            "height": 150.0,
            "norm_left": 0.1,
            "norm_top": 0.1,
            "norm_right": 0.3,
            "norm_bottom": 0.25,
            "occluded": False,
            "truncated": False,
            "source": "human",
        }
    ]

    res = client.post(f"/api/annotations/{img_name}", json={"image_name": img_name, "boxes": boxes_payload})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "saved"
    assert data["review_status"] == "reviewed"
    assert "difficulty" in data

    # Verify retrieval
    res_get = client.get(f"/api/annotations/{img_name}")
    assert res_get.status_code == 200
    assert len(res_get.json()["boxes"]) == 1


def test_autolabel_preview():
    res = client.post("/api/autolabel/preview", json={
        "classes": [{"name": "car", "prompt": "passenger car", "color": "#06b6d4", "enabled": True}],
        "confidence_threshold": 0.3,
    })
    assert res.status_code == 200
    data = res.json()
    assert "image_name" in data
    assert "detections" in data
    assert "count" in data


def test_dataset_export():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = client.post("/api/dataset/export", json={
            "format": "yolo",
            "train_ratio": 0.70,
            "val_ratio": 0.20,
            "test_ratio": 0.10,
            "output_dir": tmp_dir,
        })
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["format"] == "yolo"
        assert "splits" in data


def test_active_learning_queue():
    res = client.get("/api/active-learning/queue?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert "queue" in data
    assert "count" in data


def test_model_presets():
    res = client.get("/api/models/presets")
    assert res.status_code == 200
    data = res.json()
    assert "presets" in data
    assert len(data["presets"]) >= 3
    preset_ids = [p["id"] for p in data["presets"]]
    assert "yolo11n.pt" in preset_ids


def test_model_browse_weights():
    res = client.post("/api/models/browse-weights")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data


def test_model_validate_nonexistent():
    res = client.post("/api/models/validate", json={"path": "/nonexistent/model.pt"})
    assert res.status_code == 400


def test_yolo_ensemble_detection():
    res_imgs = client.get("/api/images")
    img_name = res_imgs.json()["images"][0]["filename"]

    res = client.post("/api/detect/yolo", json={
        "image_name": img_name,
        "conf_threshold": 0.25,
        "models": ["yolo11n.pt"],
    })
    assert res.status_code == 200
    data = res.json()
    assert data["image_name"] == img_name
    assert "boxes" in data
    assert data["count"] > 0


def test_autolabel_preview_with_florence_and_yolo_ensemble():
    res = client.post("/api/autolabel/preview", json={
        "classes": [{"name": "car", "prompt": "passenger car", "color": "#06b6d4", "enabled": True}],
        "confidence_threshold": 0.3,
        "enable_grounding_dino": True,
        "enable_yolo": True,
        "yolo_models": ["yolo11n.pt"],
        "enable_florence2": True,
        "enable_florence2_verifier": True,
        "enable_sam2_masks": False,
    })
    assert res.status_code == 200
    data = res.json()
    assert "image_name" in data
    assert "detections" in data
