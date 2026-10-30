"""Model download/readiness contracts without fetching weights or allocating a GPU."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.services.inference import preparation
from app.services.inference.preparation import ModelPreparation


@pytest.fixture
def workers(monkeypatch):
    threads = []

    def create_thread(**kwargs):
        thread = threading.Thread(**kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(
        preparation, "threading", SimpleNamespace(Lock=threading.Lock, Thread=create_thread)
    )
    yield threads
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive(), "Model preparation worker did not finish"


def join_workers(workers):
    for thread in workers:
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_status_stays_responsive_until_real_weights_finish_loading(workers):
    entered, release = threading.Event(), threading.Event()

    def load():
        entered.set()
        assert release.wait(timeout=5)

    # Constructing a lazy model alone must never signal readiness.
    lazy_model = SimpleNamespace(_ensure_loaded=Mock(side_effect=load))
    engine = SimpleNamespace(
        uses_cloud=False, _get_grounding_detector=Mock(return_value=lazy_model)
    )
    service = ModelPreparation(engine, threading.RLock())
    try:
        service.start(["grounding_dino"])
        assert entered.wait(timeout=5)
        status = service.snapshot()
        assert status["models"]["grounding_dino"]["state"] == "preparing"
        service.start(["grounding_dino"])
        assert len(workers) == 1
        status["models"]["grounding_dino"]["state"] = "ready"
        assert service.snapshot()["models"]["grounding_dino"]["state"] == "preparing"
    finally:
        release.set()
        join_workers(workers)
    assert service.snapshot()["models"]["grounding_dino"]["state"] == "ready"
    lazy_model._ensure_loaded.assert_called_once()


def test_failed_download_can_retry_without_reloading_ready_models(workers):
    sam = SimpleNamespace(
        _ensure_loaded=Mock(side_effect=[OSError("Connection interrupted"), None])
    )
    engine = SimpleNamespace(
        uses_cloud=False,
        _get_yolo_detector=Mock(),
        _get_sam_segmenter=Mock(return_value=sam),
    )
    service = ModelPreparation(engine, threading.RLock())
    service.start(["yolo:custom.pt", "sam2"])
    join_workers(workers)
    status = service.snapshot()["models"]
    assert status["sam2"]["state"] == "error"
    assert "Connection interrupted" in status["sam2"]["error"]
    assert status["yolo:custom.pt"]["state"] == "ready"

    service.start(["yolo:custom.pt", "sam2"])
    join_workers(workers)
    assert all(model["state"] == "ready" for model in service.snapshot()["models"].values())
    engine._get_yolo_detector.assert_called_once_with("custom.pt")
    assert sam._ensure_loaded.call_count == 2


def test_new_selection_queues_behind_an_active_download(workers):
    entered, release = threading.Event(), threading.Event()

    def load():
        entered.set()
        assert release.wait(timeout=5)

    engine = SimpleNamespace(
        uses_cloud=False,
        _get_grounding_detector=lambda: SimpleNamespace(_ensure_loaded=load),
        _get_vlm_helper=Mock(return_value=SimpleNamespace(ensure_loaded=Mock())),
        _get_locate_anything_detector=Mock(return_value=SimpleNamespace(_ensure_loaded=Mock())),
    )
    service = ModelPreparation(engine, threading.RLock())
    try:
        service.start(["grounding_dino"])
        assert entered.wait(timeout=5)
        service.start(["florence2", "locate_anything", "florence2"])
        assert service.snapshot()["models"]["florence2"]["state"] == "queued"
        assert len(workers) == 1
    finally:
        release.set()
        join_workers(workers)
    assert all(model["state"] == "ready" for model in service.snapshot()["models"].values())
    engine._get_vlm_helper.assert_called_once()
    engine._get_locate_anything_detector.assert_called_once()


def test_cloud_preparation_never_starts_local_model_downloads(workers):
    service = ModelPreparation(SimpleNamespace(uses_cloud=True), threading.RLock())
    assert service.start(["grounding_dino", "yolo:yolo11n.pt", "sam2"]) == {
        "backend": "cloud", "models": {},
    }
    assert workers == []


def test_api_prepares_only_selected_models_and_exposes_status(monkeypatch):
    from app.api import server

    snapshot = {"backend": "local", "models": {"locate_anything": {
        "name": "Locate Anything 3B", "state": "preparing", "error": None,
    }}}
    service = Mock()
    service.start.return_value = snapshot
    service.snapshot.return_value = snapshot
    monkeypatch.setattr(server, "MODEL_PREPARATION", service)
    # Avoid the lifespan context: startup pre-warms real YOLO weights.
    client = TestClient(server.app, headers={"X-VisionLab-Token": server.SESSION_TOKEN})
    response = client.post("/api/models/prepare", json={
        "enable_grounding_dino": False, "enable_yolo": False, "enable_sam2_masks": False,
        "enable_locate_anything": True, "enable_florence2_verifier": True,
    })
    assert response.status_code == 200
    assert response.json() == snapshot
    service.start.assert_called_once_with(["florence2", "locate_anything"])
    response = client.get("/api/models/status")
    assert response.status_code == 200
    assert response.json() == snapshot
    assert TestClient(server.app).get("/api/models/status").status_code == 401

    response = client.post("/api/models/prepare", json={"yolo_models": ["a", "b", "c", "d"]})
    assert response.status_code == 422
    assert service.start.call_count == 1
