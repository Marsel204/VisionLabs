"""Cloud annotation contracts, including operation without local model runtimes."""

from __future__ import annotations

import io
import json
import threading
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.cloud_worker import create_app
from app.services.annotation.domain import BoundingBox
from app.services.auto_label.engine import AutoLabelEngine
from app.services.auto_label.models import (
    AutoLabelClass,
    AutoLabelConfig,
    AutoLabelDetection,
    AutoLabelResult,
)
from app.services.inference.cloud import (
    CloudInferenceClient,
    CloudInferenceError,
    CloudInferenceSettings,
)


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "private-image.jpg"
    Image.new("RGB", (100, 80), "white").save(path)
    return path


@pytest.fixture
def settings():
    return CloudInferenceSettings("https://worker.example/v1/infer", "test-only-token", 30)


@pytest.fixture
def config():
    return AutoLabelConfig(
        classes=[AutoLabelClass("car", "red car", "#112233")],
        enable_florence2_verifier=True,
    )


def response_payload():
    return {
        "version": 1,
        "width": 100,
        "height": 80,
        "detections": [
            {
                "class_name": "car",
                "confidence": 0.9,
                "box": [0.1, 0.2, 0.7, 0.8],
                "polygon": [[0.1, 0.2], [0.7, 0.2], [0.7, 0.8]],
            }
        ],
    }


def install_response(monkeypatch, payload):
    opener = Mock()
    opener.open.side_effect = lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr("app.services.inference.cloud.build_opener", lambda *args: opener)
    return opener


def test_cloud_settings_are_explicit_and_do_not_disclose_token(monkeypatch, settings):
    monkeypatch.delenv("VISIONLAB_INFERENCE_BACKEND", raising=False)
    assert CloudInferenceSettings.from_env() is None
    monkeypatch.setenv("VISIONLAB_INFERENCE_BACKEND", "cloud")
    monkeypatch.delenv("VISIONLAB_CLOUD_URL", raising=False)
    monkeypatch.delenv("VISIONLAB_CLOUD_TOKEN", raising=False)
    with pytest.raises(ValueError, match="VISIONLAB_CLOUD_URL"):
        CloudInferenceSettings.from_env()
    monkeypatch.setenv("VISIONLAB_CLOUD_URL", settings.url)
    monkeypatch.setenv("VISIONLAB_CLOUD_TOKEN", settings.token)
    assert CloudInferenceSettings.from_env().url == settings.url
    assert settings.token not in repr(settings)


@pytest.mark.parametrize(
    "url",
    [
        "http://worker.example/v1/infer",
        "https://user:password@worker.example/v1/infer",
        "https://worker.example/v1/infer?token=secret",
        "file:///etc/passwd",
    ],
)
def test_cloud_requires_https_without_credentials_in_url(url):
    with pytest.raises(ValueError):
        CloudInferenceSettings(url, "test-token")


def test_response_preserves_geometry_prompts_and_local_identity(
    monkeypatch,
    image_path,
    settings,
    config,
):
    opener = install_response(monkeypatch, response_payload())
    result = CloudInferenceClient(settings).predict(image_path, config)
    assert result.image_path == image_path.resolve()
    assert result.image_width == 100 and result.image_height == 80
    assert result.detections[0].box == BoundingBox(0.1, 0.2, 0.7, 0.8)
    assert result.detections[0].polygon_pixels == [[10, 16], [70, 16], [70, 64]]
    assert result.detections[0].color == "#112233"
    request = opener.open.call_args.args[0]
    assert request.get_header("Authorization") == "Bearer test-only-token"
    assert opener.open.call_args.kwargs["timeout"] == 30
    payload = json.loads(request.data)
    assert payload["options"]["classes"][0]["prompt"] == "red car"
    assert payload["options"]["enable_florence2_verifier"] is True
    assert str(image_path) not in request.data.decode()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(width=99),
        lambda p: p.update(version=2),
        lambda p: p["detections"][0].update(confidence=float("nan")),
        lambda p: p["detections"][0].update(confidence=True),
        lambda p: p["detections"][0].update(box=[0.7, 0.2, 0.1, 0.8]),
        lambda p: p["detections"][0].update(polygon=[[0.1, 0.2], [2, 0.5], [0.5, 0.8]]),
        lambda p: p["detections"][0].update(class_name="unrequested class"),
    ],
)
def test_bad_cloud_results_are_rejected_before_annotation(
    monkeypatch,
    image_path,
    settings,
    config,
    mutate,
):
    payload = response_payload()
    mutate(payload)
    install_response(monkeypatch, payload)
    with pytest.raises(CloudInferenceError):
        CloudInferenceClient(settings).predict(image_path, config)


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("timeout"),
        URLError("network unavailable"),
        HTTPError("https://worker.example", 401, "Unauthorized", {}, None),
        HTTPError("https://worker.example", 503, "Unavailable", {}, None),
    ],
)
def test_cloud_failures_are_reported_without_credentials_or_silent_empty_results(
    monkeypatch,
    image_path,
    settings,
    config,
    failure,
):
    opener = Mock()
    opener.open.side_effect = failure
    monkeypatch.setattr("app.services.inference.cloud.build_opener", lambda *args: opener)
    with pytest.raises(CloudInferenceError) as error:
        CloudInferenceClient(settings).predict(image_path, config)
    assert settings.token not in str(error.value)
    assert opener.open.call_count == 1  # Do not silently repeat billable inference.


def test_cloud_engine_never_initializes_local_models(monkeypatch, image_path, settings, config):
    install_response(monkeypatch, response_payload())
    engine = AutoLabelEngine(cloud_settings=settings)
    assert engine.uses_cloud
    for name in (
        "_get_grounding_detector",
        "_get_sam_segmenter",
        "_get_vlm_helper",
        "_get_yolo_detector",
    ):
        monkeypatch.setattr(engine, name, Mock(side_effect=AssertionError("local model loaded")))
    assert engine.run_preview(image_path, config).count == 1
    payload = response_payload()
    payload["detections"] = []
    install_response(monkeypatch, payload)
    assert engine.run_preview(image_path, config).count == 0


def test_worker_authentication_and_full_client_round_trip(
    monkeypatch, image_path, settings, config
):
    worker_engine = Mock()
    worker_engine.run_preview.side_effect = lambda path, options: AutoLabelResult(
        path,
        100,
        80,
        [
            AutoLabelDetection(
                "car",
                0.9,
                BoundingBox(0.1, 0.2, 0.7, 0.8),
                polygon_normalized=[[0.1, 0.2], [0.7, 0.2], [0.7, 0.8]],
            )
        ],
    )
    worker = TestClient(create_app(token=settings.token, engine=worker_engine))
    assert worker.post("/v1/infer", json={}).status_code == 401
    assert (
        worker.post("/v1/infer", headers={"Authorization": "Bearer wrong"}, json={}).status_code
        == 401
    )
    worker_engine.run_preview.assert_not_called()
    captured = []

    def send(request, **kwargs):
        reply = worker.post("/v1/infer", content=request.data, headers=dict(request.header_items()))
        assert reply.status_code == 200, reply.text
        captured.append(reply.json())
        return io.BytesIO(reply.content)

    opener = Mock()
    opener.open.side_effect = send
    monkeypatch.setattr("app.services.inference.cloud.build_opener", lambda *args: opener)
    assert CloudInferenceClient(settings).predict(image_path, config).count == 1
    assert captured[0]["detections"][0]["polygon"] == response_payload()["detections"][0]["polygon"]
    remote_path, remote_options = worker_engine.run_preview.call_args.args
    assert not remote_path.exists()  # Worker removes temporary image after inference.
    assert remote_options.classes[0].prompt == "red car"


def test_worker_rejects_bad_image_custom_paths_and_large_body(settings):
    worker = TestClient(create_app(token=settings.token, engine=Mock()))
    headers = {"Authorization": f"Bearer {settings.token}"}
    bad_image = {"image_base64": "not-base64", "options": {"classes": [{"name": "car"}]}}
    assert worker.post("/v1/infer", headers=headers, json=bad_image).status_code == 422
    bad_image["options"]["yolo_models"] = ["/private/unsafe.pt"]
    assert worker.post("/v1/infer", headers=headers, json=bad_image).status_code == 422
    headers["Content-Length"] = str(100 * 1024 * 1024)
    assert worker.post("/v1/infer", headers=headers, content=b"{}").status_code == 413


def test_local_worker_ignores_cloud_client_environment(monkeypatch):
    monkeypatch.setenv("VISIONLAB_INFERENCE_BACKEND", "cloud")
    monkeypatch.delenv("VISIONLAB_CLOUD_URL", raising=False)
    assert not AutoLabelEngine(allow_cloud=False).uses_cloud


def test_worker_reports_model_failure_instead_of_empty_success(monkeypatch, image_path, config):
    import base64

    from app.services.inference.cloud import InferenceOptions

    detector = Mock()
    detector.detect.side_effect = RuntimeError("model unavailable")
    engine = AutoLabelEngine(
        grounding_detector=detector,
        allow_cloud=False,
        fail_on_model_error=True,
    )
    client = TestClient(create_app(token="test-token", engine=engine))
    reply = client.post(
        "/v1/infer",
        headers={"Authorization": "Bearer test-token"},
        json={
            "image_base64": base64.b64encode(image_path.read_bytes()).decode(),
            "options": InferenceOptions.from_config(config).model_dump(mode="json"),
        },
    )
    assert reply.status_code == 503
    assert "model unavailable" not in reply.text


def test_cloud_quick_detect_preview_batch_and_startup_skip_local_models(
    monkeypatch,
    image_path,
    settings,
    config,
):
    from app.api import server
    from app.services.dataset.index import DatasetIndex
    from app.services.dataset.repository import AnnotationRepository

    install_response(monkeypatch, response_payload())
    engine = AutoLabelEngine(cloud_settings=settings)
    monkeypatch.setattr(server, "AUTOLABEL_ENGINE", engine)
    monkeypatch.setattr(server, "ACTIVE_DATASET_DIR", image_path.parent)
    index = DatasetIndex(image_path.parent / ".dataset_index.sqlite")
    index.scan(image_path.parent)
    monkeypatch.setattr(server, "DATASET_INDEX", index)
    monkeypatch.setattr(
        engine, "_get_yolo_detector", Mock(side_effect=AssertionError("local model loaded"))
    )
    finished = threading.Event()
    run_batch = server._run_batch_autolabel_worker

    def tracked_batch(*args):
        try:
            run_batch(*args)
        finally:
            finished.set()

    monkeypatch.setattr(server, "_run_batch_autolabel_worker", tracked_batch)
    with TestClient(server.app, headers={"X-VisionLab-Token": server.SESSION_TOKEN}) as client:
        assert server.get_yolo_model() is None
        health = client.get("/api/health").json()
        assert health["inference"]["backend"] == "cloud"
        detected = client.post(
            "/api/detect/yolo", json={"image_name": image_path.name, "classes": ["car"]}
        )
        assert detected.status_code == 200, detected.text
        assert detected.json()["count"] == 1
        preview = client.post(
            "/api/autolabel/preview", json={"image_name": image_path.name, "classes": ["car"]}
        )
        assert preview.status_code == 200, preview.text
        assert preview.json()["detections"][0]["polygon_pixels"] == [[10, 16], [70, 16], [70, 64]]
        batch = client.post("/api/autolabel/batch", json={"classes": ["car"]})
        assert batch.status_code == 200, batch.text
        assert finished.wait(3), "Batch worker did not complete"
        assert server.AUTOLABEL_BATCH_STATUS["processed_count"] == 1, server.AUTOLABEL_BATCH_STATUS
        stored = AnnotationRepository(image_path.parent).read(image_path)
        assert stored["boxes"][0]["polygon_normalized"]
        opener = Mock()
        opener.open.side_effect = TimeoutError("timeout")
        monkeypatch.setattr("app.services.inference.cloud.build_opener", lambda *args: opener)
        failed = client.post(
            "/api/autolabel/preview", json={"image_name": image_path.name, "classes": ["car"]}
        )
        assert failed.status_code == 503
        assert AnnotationRepository(image_path.parent).read(image_path)["boxes"] == stored["boxes"]
    index.close()
