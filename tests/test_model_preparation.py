"""Model download/readiness contracts without fetching weights or allocating a GPU."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.services.auto_label.engine import AutoLabelEngine
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
    engine = AutoLabelEngine(
        yolo_detector=Mock(return_value=[]), yolo_model_name="custom.pt",
        sam_segmenter=sam, device="cpu", allow_cloud=False,
    )
    engine._get_yolo_detector = Mock(wraps=engine._get_yolo_detector)
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
        "loaded": False, "inference_ready": None, "warning": None,
    }}}
    service = Mock()
    service.start.return_value = snapshot
    service.snapshot.return_value = snapshot
    monkeypatch.setattr(server, "MODEL_PREPARATION", service)
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


def test_selected_yolo_warms_once_after_construction_and_reuses_on_prepare(workers, monkeypatch):
    import sys

    events = []

    def predict(image, **kwargs):
        events.append("predict")
        assert image.mode == "RGB" and image.size == (64, 64)
        assert image.getbbox() is None
        assert kwargs == {"device": "cpu", "verbose": False}
        return []

    detector = Mock(side_effect=predict)

    def construct(name):
        events.append("construct")
        assert name == "selected.pt"
        return detector

    factory = Mock(side_effect=construct)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))
    engine = AutoLabelEngine(device="cpu", allow_cloud=False)
    service = ModelPreparation(engine, threading.RLock())
    service.start(["yolo:selected.pt", "yolo:selected.pt"])
    join_workers(workers)
    for _ in range(3):
        service.start(["yolo:selected.pt"])
    join_workers(workers)
    assert events == ["construct", "predict"]
    factory.assert_called_once_with("selected.pt")
    assert set(engine._yolo_detectors) == {"selected.pt"}
    assert service.snapshot()["models"]["yolo:selected.pt"] == {
        "name": "selected.pt", "state": "ready", "error": None,
        "loaded": True, "inference_ready": True, "warning": None,
    }


def test_yolo_warmup_is_once_per_device_and_uses_inference_device(workers, monkeypatch):
    from app.core import runtime

    detector = Mock(return_value=[])
    engine = AutoLabelEngine(yolo_detector=detector, device="auto", allow_cloud=False)
    monkeypatch.setattr(runtime, "detect_gpu", lambda: SimpleNamespace(device="cpu"))
    service = ModelPreparation(engine, threading.RLock())
    for device in ("auto", "cuda", "cpu", "gpu"):
        engine._device = device
        service.start(["yolo:yolo11n.pt"])
        join_workers(workers)
        assert service.snapshot()["models"]["yolo:yolo11n.pt"]["inference_ready"] is True
    assert [call.kwargs["device"] for call in detector.call_args_list] == ["cpu", 0]


@pytest.mark.parametrize("message", ["CUDA out of memory", "runtime initialization failed"])
def test_warmup_failure_keeps_loaded_model_usable_without_retrying(workers, message):
    detector = Mock(side_effect=RuntimeError(message))
    sam = SimpleNamespace(_ensure_loaded=Mock())
    engine = AutoLabelEngine(
        yolo_detector=detector, sam_segmenter=sam, device="cpu", allow_cloud=False,
    )
    service = ModelPreparation(engine, threading.RLock())
    service.start(["yolo:yolo11n.pt", "sam2"])
    join_workers(workers)
    status = service.snapshot()["models"]
    assert status["yolo:yolo11n.pt"]["state"] == "ready"
    assert status["yolo:yolo11n.pt"]["loaded"] is True
    assert status["yolo:yolo11n.pt"]["inference_ready"] is False
    assert status["yolo:yolo11n.pt"]["error"] is None
    assert message in status["yolo:yolo11n.pt"]["warning"]
    assert status["sam2"]["state"] == "ready"
    assert status["sam2"]["inference_ready"] is None
    service.start(["yolo:yolo11n.pt"])
    join_workers(workers)
    detector.assert_called_once()


def test_status_endpoint_reports_loaded_but_not_ready_during_background_warmup(
    workers, monkeypatch,
):
    from app.api import server

    entered, release = threading.Event(), threading.Event()
    inference_lock = threading.RLock()

    def predict(*args, **kwargs):
        assert inference_lock._is_owned()
        entered.set()
        assert release.wait(timeout=5)
        return []

    engine = AutoLabelEngine(
        yolo_detector=Mock(side_effect=predict), device="cpu", allow_cloud=False,
    )
    service = ModelPreparation(engine, inference_lock)
    monkeypatch.setattr(server, "MODEL_PREPARATION", service)
    client = TestClient(server.app, headers={"X-VisionLab-Token": server.SESSION_TOKEN})
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            response = client.post("/api/models/prepare", json={
                "enable_grounding_dino": False, "enable_yolo": True,
                "enable_sam2_masks": False, "enable_florence2_verifier": False,
            })
            assert response.status_code == 200
            assert entered.wait(timeout=5)
            response = pool.submit(client.get, "/api/models/status").result(timeout=1)
            status = response.json()["models"]["yolo:yolo11n.pt"]
            assert status["state"] == "preparing"
            assert status["loaded"] is True
            assert status["inference_ready"] is False
            assert status["warning"] is None
            # Re-prepare is also responsive and does not queue a second forward pass.
            pool.submit(service.start, ["yolo:yolo11n.pt"]).result(timeout=1)
            assert len(workers) == 1
        finally:
            release.set()
            join_workers(workers)
    assert client.get("/api/models/status").json()["models"]["yolo:yolo11n.pt"]["inference_ready"]


def test_eviction_invalidates_readiness_and_replacement_warms_again(workers, monkeypatch):
    import sys

    detectors = [Mock(return_value=[]), Mock(return_value=[])]
    factory = Mock(side_effect=detectors)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))
    engine = AutoLabelEngine(device="cpu", allow_cloud=False)
    service = ModelPreparation(engine, threading.RLock())
    service.start(["yolo:selected.pt"])
    join_workers(workers)
    assert service.snapshot()["models"]["yolo:selected.pt"]["inference_ready"] is True
    with service._inference_lock:
        engine._yolo_detectors.clear()
    assert "yolo:selected.pt" not in service.snapshot()["models"]
    service.start(["yolo:selected.pt"])
    join_workers(workers)
    assert service.snapshot()["models"]["yolo:selected.pt"]["inference_ready"] is True
    assert factory.call_count == 2
    for detector in detectors:
        detector.assert_called_once()


def test_deselected_queued_yolo_is_not_constructed_or_warmed(workers):
    grounding = SimpleNamespace(_ensure_loaded=Mock())
    engine = AutoLabelEngine(grounding_detector=grounding, device="cpu", allow_cloud=False)
    engine._get_yolo_detector = Mock(side_effect=AssertionError("Unselected model loaded"))
    inference_lock = threading.RLock()
    service = ModelPreparation(engine, inference_lock)
    with inference_lock:
        service.start(["yolo:unselected.pt"])
        service.start(["grounding_dino"])
    join_workers(workers)
    assert set(service.snapshot()["models"]) == {"grounding_dino"}
    engine._get_yolo_detector.assert_not_called()
    grounding._ensure_loaded.assert_called_once()


@pytest.mark.parametrize("cloud", [False, True])
def test_startup_waits_for_model_selection_instead_of_loading_default(monkeypatch, cloud):
    from app.api import server

    engine = SimpleNamespace(uses_cloud=cloud, _get_yolo_detector=Mock())
    service = Mock()
    monkeypatch.setattr(server, "AUTOLABEL_ENGINE", engine)
    monkeypatch.setattr(server, "MODEL_PREPARATION", service)
    monkeypatch.setattr(server, "ACTIVE_DATASET_DIR", None)
    with TestClient(server.app, headers={"X-VisionLab-Token": server.SESSION_TOKEN}):
        pass
    engine._get_yolo_detector.assert_not_called()
    service.start.assert_not_called()
