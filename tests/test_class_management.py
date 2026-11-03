import json

import pytest
import yaml
from PIL import Image

from app.services.dataset.class_management import change_class
from app.services.dataset.repository import AnnotationRepository, RepositoryError


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "data.yaml").write_text("names: [motorcycle, car, bus]\nnc: 3\n")
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (100, 80)).save(image)
    (tmp_path / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n2 0.2 0.2 0.1 0.1\n")
    return AnnotationRepository(tmp_path), image


def test_rename_preserves_geometry_status_and_updates_every_projection(dataset):
    repo, image = dataset
    before = repo.read(image)
    saved = repo.save(image, before["boxes"], status="ai_labeled")
    result = change_class(repo, "rename", "motorcycle", "motorbike", repo.classes())
    after = repo.read(image)
    assert result["classes"] == ["motorbike", "car", "bus"]
    assert after["status"] == "ai_labeled"
    assert after["revision"] == saved["revision"] + 1
    assert after["boxes"][0]["class_name"] == "motorbike"
    assert after["boxes"][0]["norm_left"] == before["boxes"][0]["norm_left"]
    assert (repo.root / "sample.txt").read_text().startswith("0 ")
    assert (repo.root / result["backup"] / "data.yaml").read_text().startswith("names:")


def test_delete_unused_class_reindexes_yolo_and_updates_metadata(dataset):
    repo, image = dataset
    change_class(repo, "delete", "car", None, repo.classes())
    assert [box["class_id"] for box in repo.read(image)["boxes"]] == [0, 1]
    assert (repo.root / "sample.txt").read_text().splitlines()[1].startswith("1 ")
    assert yaml.safe_load((repo.root / "data.yaml").read_text())["nc"] == 2


def test_used_class_requires_explicit_mapping_or_annotation_deletion(dataset):
    repo, _ = dataset
    original = (repo.root / "sample.txt").read_bytes()
    with pytest.raises(RepositoryError, match="masih digunakan"):
        change_class(repo, "delete", "motorcycle", None, repo.classes())
    assert repo.classes() == ["motorcycle", "car", "bus"]
    assert (repo.root / "sample.txt").read_bytes() == original


def test_merge_class_keeps_annotations_and_reindexes_canonical_and_legacy(dataset):
    repo, image = dataset
    repo.save(image, repo.read(image)["boxes"])
    change_class(repo, "delete", "motorcycle", "car", repo.classes())
    after = repo.read(image)
    assert [(box["class_name"], box["class_id"]) for box in after["boxes"]] == [
        ("car", 0),
        ("bus", 1),
    ]
    assert (repo.root / "sample.txt").read_text().startswith("0 ")


def test_explicit_annotation_deletion_allows_empty_registry(dataset):
    repo, image = dataset
    repo.save(image, repo.read(image)["boxes"])
    for name in ["motorcycle", "car", "bus"]:
        change_class(repo, "delete", name, None, repo.classes(), delete_annotations=True)
    assert repo.classes() == []
    assert repo.read(image)["boxes"] == []
    assert (repo.root / "sample.txt").read_text() == ""


@pytest.mark.parametrize("target", ["car", "CAR", "", "  "])
def test_rename_rejects_duplicate_or_empty_names(dataset, target):
    repo, _ = dataset
    with pytest.raises(RepositoryError):
        change_class(repo, "rename", "motorcycle", target, repo.classes())


def test_stale_registry_and_invalid_labels_fail_without_changing_files(dataset):
    repo, _ = dataset
    with pytest.raises(RepositoryError):
        change_class(repo, "rename", "car", "taxi", ["car"])
    label = repo.root / "sample.txt"
    label.write_text("8 0.5 0.5 0.2 0.2\n")
    with pytest.raises(RepositoryError):
        change_class(repo, "delete", "car", None, repo.classes())
    assert repo.classes() == ["motorcycle", "car", "bus"]
    assert label.read_text().startswith("8 ")


def test_write_failure_restores_all_files(dataset, monkeypatch):
    from app.services.dataset import class_management

    repo, image = dataset
    repo.save(image, repo.read(image)["boxes"])
    paths = [repo.root / "data.yaml", repo.root / "sample.txt", repo.canonical_path(image)]
    original = {path: path.read_bytes() for path in paths}
    write = class_management.atomic_text
    failed = False

    def fail_once(path, value):
        nonlocal failed
        if path == repo.root / "data.yaml" and not failed:
            failed = True
            raise OSError("disk full")
        write(path, value)

    monkeypatch.setattr(class_management, "atomic_text", fail_once)
    with pytest.raises(OSError, match="disk full"):
        change_class(repo, "rename", "car", "taxi", repo.classes())
    assert {path: path.read_bytes() for path in paths} == original
    assert json.loads(repo.canonical_path(image).read_text())["revision"] == 1


def test_polygons_and_orphan_labels_keep_their_coordinates(dataset):
    repo, _ = dataset
    labels = repo.root / "labels"
    labels.mkdir()
    path = labels / "orphan.txt"
    path.write_text("2 0.1 0.1 0.9 0.1 0.5 0.9\n")
    change_class(repo, "delete", "car", None, repo.classes())
    assert path.read_text() == "1 0.1 0.1 0.9 0.1 0.5 0.9\n"


def test_symlink_label_cannot_modify_a_file_outside_the_dataset(dataset, tmp_path):
    repo, _ = dataset
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("2 0.5 0.5 0.2 0.2\n")
    label = repo.root / "sample.txt"
    label.unlink()
    label.symlink_to(outside)
    try:
        with pytest.raises(RepositoryError, match="escapes dataset"):
            change_class(repo, "delete", "car", None, repo.classes())
        assert outside.read_text().startswith("2 ")
    finally:
        outside.unlink()


@pytest.fixture
def api(dataset, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import server

    repo, _ = dataset
    monkeypatch.setattr(server, "ACTIVE_DATASET_DIR", repo.root)
    from app.services.dataset.index import DatasetIndex

    index = DatasetIndex(repo.root / ".dataset_index.sqlite")
    index.scan(repo.root)
    monkeypatch.setattr(server, "DATASET_INDEX", index)
    monkeypatch.setattr(server, "AUTOLABEL_BATCH_STATUS", {"running": False})
    monkeypatch.setattr(server, "IMPORT_STATUS", {"running": False})
    monkeypatch.setattr(server.COLAB_CONNECTION, "status", lambda: {"busy": False})
    yield (
        TestClient(server.app, headers={"X-VisionLab-Token": server.SESSION_TOKEN}),
        {
            "dataset_id": str(repo.root),
            "expected_classes": repo.classes(),
            "action": "rename",
            "name": "car",
            "target": "taxi",
        },
    )
    index.close()


def test_api_refreshes_cached_counts_after_annotation_deletion(api):
    client, payload = api
    assert client.get("/api/images").json()["images"][0]["annotation_count"] == 2
    response = client.patch(
        "/api/dataset/classes",
        json={
            **payload,
            "action": "delete",
            "name": "motorcycle",
            "target": None,
            "delete_annotations": True,
        },
    )
    assert response.status_code == 200, response.text
    assert client.get("/api/images").json()["images"][0]["annotation_count"] == 1


def test_api_requires_session_and_correct_dataset(api):
    client, payload = api
    assert (
        client.patch(
            "/api/dataset/classes", json=payload, headers={"X-VisionLab-Token": "bad"}
        ).status_code
        == 401
    )
    assert (
        client.patch("/api/dataset/classes", json={**payload, "dataset_id": "/other"}).status_code
        == 409
    )
    response = client.patch("/api/dataset/classes", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["classes"] == ["motorcycle", "taxi", "bus"]


@pytest.mark.parametrize("job", ["auto-label", "import", "colab"])
def test_api_blocks_class_changes_during_background_jobs(api, monkeypatch, job):
    from app.api import server

    client, payload = api
    if job == "auto-label":
        monkeypatch.setattr(server, "AUTOLABEL_BATCH_STATUS", {"running": True})
    elif job == "import":
        monkeypatch.setattr(server, "IMPORT_STATUS", {"running": True})
    else:
        monkeypatch.setattr(server.COLAB_CONNECTION, "status", lambda: {"busy": True})
    assert client.patch("/api/dataset/classes", json=payload).status_code == 409
