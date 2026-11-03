"""Removing an app project closes its selection without deleting dataset content."""

import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import server
from app.services.dataset.index import DatasetIndex


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    image = root / "sample.jpg"
    Image.new("RGB", (100, 80)).save(image)
    label = root / "sample.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n")
    index = DatasetIndex(root / ".dataset_index.sqlite")
    index.scan(root)
    config = tmp_path / "active.json"
    monkeypatch.setattr(server, "ACTIVE_DATASET_DIR", root)
    monkeypatch.setattr(server, "DATASET_INDEX", index)
    monkeypatch.setattr(server, "DATASET_CONFIG_FILE", config)
    monkeypatch.setattr(server, "AUTOLABEL_BATCH_STATUS", {"running": False})
    monkeypatch.setattr(server, "IMPORT_STATUS", {"running": False})
    monkeypatch.setattr(server.COLAB_CONNECTION, "status", lambda: {"busy": False})
    client = TestClient(server.app, headers={"X-VisionLab-Token": server.SESSION_TOKEN})
    yield root, image, label, config, client
    index.close()
    if server.DATASET_INDEX is not None and server.DATASET_INDEX is not index:
        server.DATASET_INDEX.close()


def test_removing_active_project_preserves_files_and_persists_closed_state(project):
    root, image, label, config, client = project
    original = {path: path.read_bytes() for path in (image, label)}
    response = client.post("/api/dataset/close", json={"dataset_id": str(root)})
    assert response.status_code == 200, response.text
    assert server.ACTIVE_DATASET_DIR is None
    assert server.DATASET_INDEX is None
    assert json.loads(config.read_text()) == {"active_directory": None}
    assert server.get_initial_dataset_dir() is None
    assert {path: path.read_bytes() for path in (image, label)} == original
    assert (root / ".dataset_index.sqlite").is_file()
    health = client.get("/api/health").json()
    assert health["dataset_dir"] == "" and health["classes"] == []
    assert client.get("/api/images").json() == {"images": [], "total": 0, "directory": ""}
    assert client.get("/api/dataset/stats").json()["total_images"] == 0
    assert client.get("/api/active-learning/queue").json()["queue"] == []
    assert client.get("/api/dataset/annotation-sources").json()["sources"] == []
    assert client.get("/api/dataset/search-directories").status_code == 200
    assert client.get("/api/image/sample.jpg").status_code == 409
    assert client.post("/api/dataset/classes", json={"name": "new"}).status_code == 409
    upload = client.post(
        "/api/dataset/upload", files={"files": ("new.jpg", image.read_bytes(), "image/jpeg")}
    )
    assert upload.status_code == 409
    reopened = client.post("/api/dataset/select-folder", params={"path": str(root)})
    assert reopened.status_code == 200, reopened.text
    assert server.get_initial_dataset_dir() == root
    assert client.get("/api/images").json()["total"] == 1
    assert {path: path.read_bytes() for path in (image, label)} == original


def test_stale_project_cannot_close_a_different_selection(project):
    root, _, _, config, client = project
    response = client.post("/api/dataset/close", json={"dataset_id": str(root / "other")})
    assert response.status_code == 409
    assert server.ACTIVE_DATASET_DIR == root
    assert not config.exists()


@pytest.mark.parametrize("job", ["auto-label", "import", "colab"])
def test_running_work_blocks_project_removal(project, monkeypatch, job):
    root, _, _, config, client = project
    if job == "auto-label":
        monkeypatch.setattr(server, "AUTOLABEL_BATCH_STATUS", {"running": True})
    elif job == "import":
        monkeypatch.setattr(server, "IMPORT_STATUS", {"running": True})
    else:
        monkeypatch.setattr(server.COLAB_CONNECTION, "status", lambda: {"busy": True})
    assert client.post("/api/dataset/close", json={"dataset_id": str(root)}).status_code == 409
    assert server.ACTIVE_DATASET_DIR == root
    assert not config.exists()


def test_storage_failure_keeps_project_open(project, monkeypatch):
    root, _, _, _, client = project
    monkeypatch.setattr(server, "persist_active_dataset_dir", lambda _: False)
    assert client.post("/api/dataset/close", json={"dataset_id": str(root)}).status_code == 500
    assert server.ACTIVE_DATASET_DIR == root
    assert client.get("/api/images").status_code == 200
