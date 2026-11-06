"""Tests for FastAPI backend endpoints."""

from fastapi.testclient import TestClient
from app.api.server import app
from app.api import server
import pytest
from PIL import Image
from pathlib import Path
from types import SimpleNamespace
import errno
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


@pytest.mark.parametrize("code", [errno.EPERM, errno.ENOTSUP, errno.EXDEV, errno.ENOSYS, errno.EMLINK, "windows"])
def test_upload_without_hard_links(tmp_path, monkeypatch, code):
    payload = (tmp_path / "sample.jpg").read_bytes()

    def unsupported(*args):
        error = OSError(code if code != "windows" else errno.EINVAL, "hard links unavailable")
        if code == "windows":
            error.winerror = 50
        raise error

    monkeypatch.setattr(server.os, "link", unsupported)
    response = client.post("/api/dataset/upload", files={"files": ("new.jpg", payload, "image/jpeg")})
    assert response.status_code == 200, response.text
    assert response.json() == {"saved": ["new.jpg"], "count": 1, "total_indexed": 2}
    assert (tmp_path / "new.jpg").read_bytes() == payload


@pytest.mark.parametrize("fallback", [False, True])
def test_upload_link_collision_rolls_back_batch(tmp_path, monkeypatch, fallback):
    payload = (tmp_path / "sample.jpg").read_bytes()
    link = server.os.link

    def collide(source, target):
        if target.name == "second.jpg":
            target.write_bytes(b"other writer")
            if fallback:
                raise OSError(errno.EPERM, "hard links unavailable")
            raise FileExistsError(target)
        link(source, target)

    monkeypatch.setattr(server.os, "link", collide)
    response = client.post("/api/dataset/upload", files=[
        ("files", (name, payload, "image/jpeg")) for name in ("first.jpg", "second.jpg")
    ])
    assert response.status_code == 409
    assert not (tmp_path / "first.jpg").exists()
    assert (tmp_path / "second.jpg").read_bytes() == b"other writer"
    assert server.DATASET_INDEX.count() == 1


def test_upload_fallback_partial_write_rolls_back_batch(tmp_path, monkeypatch):
    payload = (tmp_path / "sample.jpg").read_bytes()
    link = server.os.link
    open_path = Path.open
    written = []

    def unsupported_second(source, target):
        if target.name == "second.jpg":
            raise OSError(errno.EPERM, "hard links unavailable")
        link(source, target)

    class FailingWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, chunk):
            self.stream.write(chunk[:10])
            written.append(chunk[:10])
            raise OSError(errno.ENOSPC, "disk full")

    def fail_write(path, mode="r", *args, **kwargs):
        stream = open_path(path, mode, *args, **kwargs)
        return FailingWriter(stream) if path == tmp_path / "second.jpg" and mode == "xb" else stream

    monkeypatch.setattr(server.os, "link", unsupported_second)
    monkeypatch.setattr(Path, "open", fail_write)
    response = client.post("/api/dataset/upload", files=[
        ("files", (name, payload, "image/jpeg")) for name in ("first.jpg", "second.jpg")
    ])
    assert response.status_code == 409
    assert written == [payload[:10]]
    assert not (tmp_path / "first.jpg").exists()
    assert not (tmp_path / "second.jpg").exists()
    assert server.DATASET_INDEX.count() == 1


@pytest.mark.parametrize("limit", ["file", "batch"])
def test_upload_size_limits_leave_no_files(tmp_path, monkeypatch, limit):
    payload = (tmp_path / "sample.jpg").read_bytes()
    # For the batch limit, neither individual file exceeds its own cap.
    monkeypatch.setattr(server, "MAX_IMAGE_BYTES", len(payload) - 1 if limit == "file" else len(payload))
    monkeypatch.setattr(server, "MAX_TRANSFER_BYTES", len(payload) if limit == "batch" else 10 * len(payload))
    response = client.post("/api/dataset/upload", files=[
        ("files", ("first.jpg", payload, "image/jpeg")),
        ("files", ("second.jpg", payload, "image/jpeg")),
    ])
    assert response.status_code == 413, response.text
    assert {p.name for p in tmp_path.iterdir() if not p.name.startswith(".dataset_index.sqlite")} == {"sample.jpg"}
    assert server.DATASET_INDEX.count() == 1


def test_upload_at_size_limit_indexes_only_new_paths(tmp_path, monkeypatch):
    payload = (tmp_path / "sample.jpg").read_bytes()
    monkeypatch.setattr(server, "MAX_IMAGE_BYTES", len(payload))
    monkeypatch.setattr(server, "MAX_TRANSFER_BYTES", len(payload))
    monkeypatch.setattr(DatasetIndex, "scan", lambda *args: pytest.fail("Upload rescanned the dataset"))
    response = client.post("/api/dataset/upload", files={
        "files": ("../../new.JPG", payload, "image/jpeg"),
    })
    assert response.status_code == 200, response.text
    assert response.json() == {"saved": ["new.JPG"], "count": 1, "total_indexed": 2}
    record = server.DATASET_INDEX.get_image(tmp_path / "new.JPG")
    assert (record["width"], record["height"]) == (100, 80)
    assert (tmp_path / "new.JPG").read_bytes() == payload
    assert not list(tmp_path.glob(".visionlab-upload-*"))


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


def test_colab_job_uses_cached_annotation_state_without_repository_reads(tmp_path, monkeypatch):
    states = [("reviewed", 0), ("ai_labeled", 0), ("unreviewed", 1),
              ("unreviewed", 0), ("unreviewed", 0)]
    for index in range(len(states)):
        Image.new("RGB", (100, 80)).save(tmp_path / f"image-{index}.jpg")
    server.DATASET_INDEX.scan(tmp_path)
    for index, (status, count) in enumerate(states):
        server.DATASET_INDEX.set_status_and_count(tmp_path / f"image-{index}.jpg", status, count)
    server.DATASET_INDEX.set_status_and_count(tmp_path / "sample.jpg", "unreviewed", 0)
    monkeypatch.setattr(server.AnnotationRepository, "read",
                        lambda *args: pytest.fail("Cached admission must not read annotations"))
    selected = []
    monkeypatch.setattr(server.COLAB_CONNECTION, "start_job",
                        lambda root, images, *args, **kwargs: selected.extend(images))

    response = client.post("/api/colab/jobs", json={"dataset_id": str(tmp_path), "classes": ["car"]})
    assert response.status_code == 200, response.text
    assert selected == [tmp_path / "image-3.jpg", tmp_path / "image-4.jpg", tmp_path / "sample.jpg"]
    assert client.post("/api/colab/jobs", json={
        "dataset_id": "/another-dataset", "classes": ["car"],
    }).status_code == 409


@pytest.mark.parametrize("unknown_count", [-1, None])
def test_colab_job_unknown_counts_read_annotations_and_preserve_selection(tmp_path, monkeypatch, unknown_count):
    for index in range(5):
        Image.new("RGB", (100, 80)).save(tmp_path / f"image-{index}.jpg")
    server.DATASET_INDEX.scan(tmp_path)
    repo = server.AnnotationRepository(tmp_path)
    repo.save(tmp_path / "image-0.jpg", [], status="reviewed", expected_revision=0)
    repo.save(tmp_path / "image-1.jpg", [], status="ai_labeled", expected_revision=0)
    (tmp_path / "image-2.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    # Repository saves now prime the cache; explicitly create the unknown state
    # this test exercises (as with an old or rebuilt index).
    for record in server.DATASET_INDEX.list_records():
        server.DATASET_INDEX.set_status_and_count(Path(record["path"]), "unreviewed", -1)
    records = server.DATASET_INDEX.list_records()
    expected = []
    for record in records:
        image = Path(record["path"])
        data = repo.read(image)
        if not data["boxes"] and data["status"] not in {"reviewed", "ai_labeled"}:
            expected.append(image)
    assert expected == [tmp_path / "image-3.jpg", tmp_path / "image-4.jpg", tmp_path / "sample.jpg"]
    if unknown_count is None:
        # SQLite uses -1; also exercise the helper's nullable/absent-count compatibility.
        list_records = server.DATASET_INDEX.list_records
        monkeypatch.setattr(server.DATASET_INDEX, "list_records", lambda *args, **kwargs: [
            {**record, "annotation_count": None} for record in list_records(*args, **kwargs)
        ])
    reads = []
    read = server.AnnotationRepository.read
    def counted_read(repository, image):
        reads.append(image)
        return read(repository, image)
    monkeypatch.setattr(server.AnnotationRepository, "read", counted_read)
    selected = []
    monkeypatch.setattr(server.COLAB_CONNECTION, "start_job",
                        lambda root, images, *args, **kwargs: selected.extend(images))

    response = client.post("/api/colab/jobs", json={"dataset_id": str(tmp_path), "classes": ["car"]})
    assert response.status_code == 200, response.text
    assert selected == expected
    assert reads == [Path(record["path"]) for record in records]


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


def _api_box(box_id, class_name, left, top, right, bottom, width=100, height=80):
    return {
        "id": box_id, "class_name": class_name, "class_id": 0, "confidence": 1.0,
        "x": left * width, "y": top * height,
        "width": (right - left) * width, "height": (bottom - top) * height,
        "norm_left": left, "norm_top": top, "norm_right": right, "norm_bottom": bottom,
        "source": "human",
    }


def test_audit_reports_unlabelled_detection_with_pixel_box():
    res = client.post(
        "/api/audit", json={"image_name": "sample.jpg", "boxes": [], "views": ["original"]}
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["summary"] == {"possibly_missed": 1}
    finding = data["findings"][0]
    assert finding["class_name"] == "car"
    assert finding["box"]["x"] == 10.0 and finding["box"]["width"] == 20.0


def test_audit_accepts_editor_boxes_and_reports_occlusion():
    boxes = [_api_box("a", "car", 0.1, 0.1, 0.3, 0.3)]
    res = client.post(
        "/api/audit", json={"image_name": "sample.jpg", "boxes": boxes, "views": ["original"]}
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["findings"] == []
    assert data["occlusion"] == [
        {"annotation_id": "a", "fraction": 0.0, "occluded": False, "occluder_ids": []}
    ]


def test_audit_rejects_unknown_image():
    res = client.post("/api/audit", json={"image_name": "missing.jpg", "boxes": []})
    assert res.status_code >= 400


def test_image_enhance_preset_serves_filtered_image_of_same_size():
    plain = client.get("/api/image/sample.jpg")
    sharp = client.get("/api/image/sample.jpg?enhance=sharpen")
    assert plain.status_code == sharp.status_code == 200
    assert sharp.headers["content-type"] == "image/jpeg"
    import io
    assert Image.open(io.BytesIO(sharp.content)).size == (100, 80)
    assert client.get("/api/image/sample.jpg?enhance=none").content == plain.content


def test_image_enhance_rejects_unknown_preset():
    assert client.get("/api/image/sample.jpg?enhance=cartoon").status_code == 422


def test_audit_layers_a_flipped_pass_by_default_and_rejects_unknown_views():
    # The stub detector returns the same pixel box for every image, so the flipped pass
    # reports it mirrored: one finding per view, at mirrored positions.
    res = client.post("/api/audit", json={"image_name": "sample.jpg", "boxes": []})
    assert res.status_code == 200, res.text
    xs = sorted(f["box"]["x"] for f in res.json()["findings"])
    assert xs == [10.0, 70.0]
    bad = client.post("/api/audit", json={"image_name": "sample.jpg", "boxes": [], "views": ["sepia"]})
    assert bad.status_code >= 400


def test_detect_yolo_layers_flipped_pass_only_when_requested():
    plain = client.post("/api/detect/yolo", json={"image_name": "sample.jpg"}).json()
    assert [b["x"] for b in plain["boxes"]] == [10.0]
    layered = client.post(
        "/api/detect/yolo", json={"image_name": "sample.jpg", "views": ["original", "flip"]}
    ).json()
    assert sorted(b["x"] for b in layered["boxes"]) == [10.0, 70.0]


def test_detect_yolo_accepts_a_single_flipped_view():
    response = TestClient(
        app, headers={"X-VisionLab-Token": server.SESSION_TOKEN},
        raise_server_exceptions=False,
    ).post("/api/detect/yolo", json={"image_name": "sample.jpg", "views": ["flip"]})
    assert response.status_code == 200, response.text
    assert [box["x"] for box in response.json()["boxes"]] == [70.0]


def test_layered_views_are_validated_and_refused_on_cloud(monkeypatch):
    bad = client.post("/api/detect/yolo", json={"image_name": "sample.jpg", "views": ["sepia"]})
    assert bad.status_code == 422
    monkeypatch.setattr(type(server.AUTOLABEL_ENGINE), "uses_cloud", property(lambda self: True))
    cloud = client.post(
        "/api/detect/yolo", json={"image_name": "sample.jpg", "views": ["original", "flip"]}
    )
    assert cloud.status_code == 400
    batch = client.post("/api/autolabel/batch", json={"classes": ["car"], "yolo_views": ["original", "flip"]})
    assert batch.status_code == 400


def _export_fixture(tmp_path):
    Image.new("RGB", (100, 80)).save(tmp_path / "second.jpg")
    server.DATASET_INDEX.scan(tmp_path)
    box = _api_box("a", "car", 0.1, 0.1, 0.3, 0.3)
    box["class_id"] = client.get("/api/health").json()["classes"].index("car")
    saved = client.post("/api/annotations/sample.jpg", json={"image_name": "sample.jpg", "boxes": [box]})
    assert saved.status_code == 200, saved.text


@pytest.mark.parametrize("fmt, artifact", [("voc", "sample.xml"), ("cvat", "annotations.xml")])
def test_export_exposes_voc_and_cvat_and_filters_to_approved_images(tmp_path, fmt, artifact):
    _export_fixture(tmp_path)
    out = tmp_path / "exports" / fmt
    res = client.post("/api/dataset/export", json={
        "format": fmt, "only_reviewed": True, "output_dir": str(out),
        "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1,
    })
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["format"] == fmt
    assert data["total_documents"] == 1
    assert data["skipped_unreviewed"] == 1
    assert (out / artifact).is_file()


def test_export_rejects_unknown_format_and_explains_empty_approved_set(tmp_path):
    assert client.post("/api/dataset/export", json={"format": "tfrecord"}).status_code == 422
    res = client.post("/api/dataset/export", json={"format": "yolo", "only_reviewed": True,
                                                   "output_dir": str(tmp_path / "out")})
    assert res.status_code == 400
    assert "approved" in res.json()["detail"]
