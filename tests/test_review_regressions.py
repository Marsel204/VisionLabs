"""Regression coverage for annotation ownership, persistence and review findings."""

from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from app.api import server as s
from app.export.exporters import CocoExporter, YoloExporter
from app.services.annotation.domain import Annotation, AnnotationDocument, BoundingBox
from app.services.auto_label.models import AutoLabelClass, AutoLabelDetection, AutoLabelResult
from app.services.dataset.coco_importer import CocoImporter, CocoImportError
from app.services.dataset.index import DatasetIndex
from app.services.dataset.repository import AnnotationRepository


def image(path: Path, size=(100, 80), color="white"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def box(**changes):
    values = dict(
        id="one",
        class_name="car",
        class_id=1,
        confidence=0.7,
        x=10.0,
        y=8.0,
        width=20.0,
        height=16.0,
        norm_left=0.1,
        norm_top=0.1,
        norm_right=0.3,
        norm_bottom=0.3,
        source="human",
        occluded=True,
        truncated=True,
    )
    values.update(changes)
    return values


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    path = image(root / "sample.jpg")
    index = DatasetIndex(root / ".dataset_index.sqlite")
    index.scan(root)
    monkeypatch.setattr(s, "ACTIVE_DATASET_DIR", root)
    monkeypatch.setattr(s, "DATASET_INDEX", index)
    monkeypatch.setattr(s, "persist_active_dataset_dir", lambda _: None)
    monkeypatch.setattr(s, "get_yolo_model", lambda: None)
    monkeypatch.setattr(s, "AUTOLABEL_BATCH_STATUS", {"running": False, "error": None})
    client = TestClient(s.app, headers={"X-VisionLab-Token": s.SESSION_TOKEN})
    yield root, path, index, client
    # Selection may have replaced the global cache; each handle is safe to close twice.
    index.close()
    if s.DATASET_INDEX is not index and s.DATASET_INDEX is not None:
        s.DATASET_INDEX.close()


def save(client, name, boxes, **extra):
    return client.post(
        f"/api/annotations/{name}", json={"image_name": name, "boxes": boxes, **extra}
    )


def test_existing_yolo_round_trip_and_metadata_export(dataset, tmp_path):
    root, path, _, client = dataset
    (root / "labels").mkdir()
    (root / "labels/sample.txt").write_text("1 .2 .2 .2 .2\n")
    assert save(client, path.name, [box()]).status_code == 200
    restored = client.get("/api/annotations/sample.jpg").json()["boxes"][0]
    assert restored["confidence"] == 0.7 and restored["occluded"] and restored["truncated"]
    result = client.post(
        "/api/dataset/export",
        json={
            "output_dir": str(tmp_path / "export"),
            "train_ratio": 1,
            "val_ratio": 0,
            "test_ratio": 0,
        },
    )
    assert result.status_code == 200, result.text
    metadata = json.loads((tmp_path / "export/annotation_metadata.json").read_text())
    assert metadata["train/sample.jpg"][0]["occluded"] is True
    # Legacy source remains available, canonical data is stored independently.
    assert (root / "labels/sample.txt").read_text().startswith("1 ")


@pytest.mark.parametrize("extension", ["png", "jpg"])
def test_independent_image_keys(dataset, extension):
    root, path, index, client = dataset
    second = image(root / "nested" / ("sample." + extension))
    index.scan(root)
    assert save(client, path.name, [box()]).status_code == 200
    assert (
        save(client, "nested/" + second.name, [box(class_name="truck", class_id=3)]).status_code
        == 200
    )
    assert client.get("/api/annotations/sample.jpg").json()["boxes"][0]["class_name"] == "car"
    assert (
        client.get("/api/annotations/nested/" + second.name).json()["boxes"][0]["class_name"]
        == "truck"
    )


def test_same_directory_different_extensions(dataset):
    root, path, index, client = dataset
    image(root / "sample.png")
    index.scan(root)
    assert save(client, "sample.jpg", [box()]).status_code == 200
    assert save(client, "sample.png", [box(class_name="truck", class_id=3)]).status_code == 200
    assert client.get("/api/annotations/sample.jpg").json()["boxes"][0]["class_name"] == "car"


def test_nested_yolo_export_uses_actual_dimensions(dataset, tmp_path):
    root, _, index, client = dataset
    image(root / "images/train/nested.jpg")
    label = root / "labels/train/nested.txt"
    label.parent.mkdir(parents=True)
    label.write_text("1 .2 .2 .2 .2\n")
    index.scan(root)
    output = tmp_path / "export"
    res = client.post(
        "/api/dataset/export",
        json={
            "format": "coco",
            "output_dir": str(output),
            "train_ratio": 1,
            "val_ratio": 0,
            "test_ratio": 0,
        },
    )
    assert res.status_code == 200, res.text
    data = json.loads((output / "train/annotations.json").read_text())
    record = next(i for i in data["images"] if i["file_name"] == "nested.jpg")
    assert (record["width"], record["height"]) == (100, 80)
    assert len(data["annotations"]) == 1


@pytest.mark.parametrize(
    "changes",
    [
        dict(confidence=1.5),
        dict(norm_right=0.05),
        dict(class_id=0),
        dict(polygon_normalized=[[0.1, 0.1], [0.2, 0.2]]),
        dict(source="unknown"),
    ],
)
def test_invalid_annotations_rejected_before_write(dataset, changes):
    root, path, _, client = dataset
    assert save(client, path.name, [box(**changes)]).status_code == 422
    assert not AnnotationRepository(root).canonical_path(path).exists()


def test_revision_and_dataset_guards(dataset):
    root, path, _, client = dataset
    assert (
        save(client, path.name, [box()], expected_revision=0, dataset_id=str(root)).status_code
        == 200
    )
    assert save(client, path.name, [], expected_revision=0).status_code == 409
    assert save(client, path.name, [], dataset_id="/other").status_code == 409
    assert client.get("/api/annotations/sample.jpg?dataset_id=/other").status_code == 409
    assert len(client.get("/api/annotations/sample.jpg").json()["boxes"]) == 1


def test_empty_review_and_list_cache_are_consistent(dataset):
    root, path, index, client = dataset
    path.with_suffix(".txt").write_text("1 .2 .2 .2 .2\n")
    for _ in range(2):
        assert client.get("/api/images").json()["images"][0]["status"] == "reviewed"
    assert index.get_image(path)["status"] == "reviewed"
    assert save(client, path.name, []).status_code == 200
    assert client.get("/api/images").json()["images"][0]["status"] == "reviewed"
    assert client.get("/api/dataset/stats").json()["reviewed"] == 1


def test_stats_count_legacy_nested_yolo(dataset):
    root, _, index, client = dataset
    image(root / "images/train/nested.jpg")
    path = root / "labels/train/nested.txt"
    path.parent.mkdir(parents=True)
    path.write_text("1 .2 .2 .2 .2\n")
    index.scan(root)
    assert client.get("/api/dataset/stats").json()["class_counts"]["car"] == 1


def test_changed_image_dimensions(dataset):
    root, path, index, client = dataset
    assert client.get("/api/images").json()["images"][0]["width"] == 100
    image(path, (300, 200))
    index.scan(root)
    assert index.get_image(path)["annotation_count"] == -1
    record = client.get("/api/images").json()["images"][0]
    assert (record["width"], record["height"]) == (300, 200)


def test_session_and_file_confinement(dataset, tmp_path):
    root, _, _, client = dataset
    untrusted = TestClient(s.app)
    assert untrusted.get("/api/images").status_code == 401
    assert (
        untrusted.get("/api/session", headers={"Origin": "https://foreign.example"}).status_code
        == 403
    )
    assert (
        client.get("/api/images", headers={"Origin": "https://foreign.example"}).status_code == 403
    )
    assert (
        untrusted.get("/api/session", headers={"Origin": "http://localhost:1420"}).status_code
        == 200
    )
    external = image(tmp_path / "external.jpg")
    sentinel = tmp_path / "secret.txt"
    sentinel.write_text("sentinel")
    for path in (external, sentinel):
        assert client.get("/api/image/" + str(path)).status_code == 422
    (root / "escape.jpg").symlink_to(external)
    assert client.get("/api/image/escape.jpg").status_code == 422
    assert client.get("/api/image/sample.jpg").status_code == 200


def test_upload_collision_and_invalid_content(dataset):
    root, path, _, client = dataset
    original = path.read_bytes()
    assert (
        client.post(
            "/api/dataset/upload", files={"files": ("sample.jpg", original, "image/jpeg")}
        ).status_code
        == 409
    )
    assert path.read_bytes() == original
    assert (
        client.post(
            "/api/dataset/upload", files={"files": ("new.jpg", b"bad", "image/jpeg")}
        ).status_code
        == 422
    )
    assert not (root / "new.jpg").exists()


def test_class_registry_is_persistent_and_consistent(dataset):
    root, path, _, client = dataset
    classes = client.post("/api/dataset/classes", json={"name": "cat"}).json()["classes"]
    assert (
        save(client, path.name, [box(class_name="cat", class_id=classes.index("cat"))]).status_code
        == 200
    )
    assert AnnotationRepository(root).classes() == classes
    assert client.get("/api/health").json()["classes"] == classes


def test_batch_pins_root_preserves_polygons_and_updates_counts(dataset, tmp_path, monkeypatch):
    root, path, index, _ = dataset
    other = tmp_path / "other"
    other.mkdir()
    image(other / path.name)

    def infer(p, config):
        s.select_folder(str(other))
        return AutoLabelResult(
            p,
            100,
            80,
            [
                AutoLabelDetection(
                    "car",
                    0.8,
                    BoundingBox(0.1, 0.1, 0.3, 0.3),
                    polygon_normalized=[[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]],
                )
            ],
        )

    monkeypatch.setattr(s.AUTOLABEL_ENGINE, "run_preview", infer)
    s._run_batch_autolabel_worker(
        s.AutoLabelBatchRequest(enable_sam2_masks=True), [path], [AutoLabelClass("car")], root
    )
    data = AnnotationRepository(root).read(path)
    assert data["boxes"][0]["polygon_normalized"]
    assert not (other / ".visionlab/annotations/sample.jpg.json").exists()
    with DatasetIndex(root / ".dataset_index.sqlite") as check:
        assert check.get_image(path)["annotation_count"] == 1
        assert check.get_image(path)["status"] == "ai_labeled"
    output = CocoExporter().export([AnnotationRepository(root).document(path)], tmp_path / "coco")
    assert json.loads(output.read_text())["annotations"][0]["segmentation"]
    assert s.AUTOLABEL_BATCH_STATUS["failed_count"] == 0


def test_batch_never_overwrites_newer_human_edits(dataset, monkeypatch):
    root, path, _, _ = dataset
    repo = AnnotationRepository(root)

    def infer(p, config):
        repo.save(p, [box(class_name="truck", class_id=3)])
        return AutoLabelResult(
            p, 100, 80, [AutoLabelDetection("car", 0.8, BoundingBox(0.1, 0.1, 0.3, 0.3))]
        )

    monkeypatch.setattr(s.AUTOLABEL_ENGINE, "run_preview", infer)
    s._run_batch_autolabel_worker(s.AutoLabelBatchRequest(), [path], [AutoLabelClass("car")], root)
    assert repo.read(path)["boxes"][0]["class_name"] == "truck"
    assert s.AUTOLABEL_BATCH_STATUS["failed_count"] == 1
    assert s.AUTOLABEL_BATCH_STATUS["error"]


def test_batch_empty_filter_is_no_work_and_admission_is_exclusive(dataset, monkeypatch):
    root, path, index, client = dataset
    AnnotationRepository(root).save(path, [])
    launched = []

    class FakeThread:
        def __init__(self, **kwargs):
            launched.append(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(s.threading, "Thread", FakeThread)
    response = s.start_autolabel_batch(s.AutoLabelBatchRequest(), None)
    assert response["total"] == 0 and not launched
    req = s.AutoLabelBatchRequest(only_unannotated=False)
    assert s.start_autolabel_batch(req, None)["total"] == 1
    with pytest.raises(s.HTTPException) as error:
        s.start_autolabel_batch(req, None)
    assert error.value.status_code == 409 and len(launched) == 1


def test_batch_setup_failure_finalizes_status(dataset, monkeypatch):
    root, path, _, _ = dataset
    monkeypatch.setattr(
        s,
        "_resolve_pipeline_mode",
        lambda **_: (_ for _ in ()).throw(ValueError("bad configuration")),
    )
    s.AUTOLABEL_BATCH_STATUS["running"] = True
    s._run_batch_autolabel_worker(s.AutoLabelBatchRequest(), [path], [AutoLabelClass("car")], root)
    assert not s.AUTOLABEL_BATCH_STATUS["running"] and s.AUTOLABEL_BATCH_STATUS["completed"]
    assert s.AUTOLABEL_BATCH_STATUS["error"] == "bad configuration"


def test_export_admission_pages_past_ten_thousand(dataset, monkeypatch, tmp_path):
    root, path, index, client = dataset
    calls = []

    def pages(**kwargs):
        offset = kwargs.get("offset", 0)
        limit = kwargs["limit"]
        calls.append(offset)
        return [{"path": str(path)} for _ in range(min(limit, max(0, 10001 - offset)))]

    monkeypatch.setattr(index, "list_records", pages)
    monkeypatch.setattr(s.YoloExporter, "export", lambda self, docs, out: out)
    result = s.export_dataset(s.ExportRequest(output_dir=str(tmp_path / "out")))
    assert result["total_documents"] == 10001
    assert max(calls) > 10000


@pytest.mark.parametrize("exporter", [YoloExporter, CocoExporter])
def test_export_allocates_unique_image_and_label_names(tmp_path, exporter):
    paths = [
        image(tmp_path / "a/same.jpg"),
        image(tmp_path / "b/same.jpg", color="black"),
        image(tmp_path / "c/same.png"),
    ]
    docs = [
        AnnotationDocument(p, 100, 80, (Annotation("car", BoundingBox(0.1, 0.1, 0.3, 0.3)),))
        for p in paths
    ]
    exporter().export(docs, tmp_path / "out")
    assert len(list((tmp_path / "out/images").iterdir())) == 3
    if exporter is YoloExporter:
        assert len(list((tmp_path / "out/labels").iterdir())) == 3
    else:
        payload = json.loads((tmp_path / "out/annotations.json").read_text())
        assert len({i["file_name"] for i in payload["images"]}) == 3


def test_coco_rejects_escape_and_imports_equal_basenames(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    external = image(tmp_path / "outside.jpg")
    for name in ("../outside.jpg", str(external)):
        with pytest.raises(CocoImportError):
            CocoImporter._safe_source_path(root, name)
    (root / "escape.jpg").symlink_to(external)
    with pytest.raises(CocoImportError):
        CocoImporter._safe_source_path(root, "escape.jpg")
    paths = [image(root / "a/same.jpg"), image(root / "b/same.jpg", color="black")]
    manifest = tmp_path / "coco.json"
    manifest.write_text(
        json.dumps(
            {
                "images": [
                    {"id": i, "file_name": str(p.relative_to(root)), "width": 100, "height": 80}
                    for i, p in enumerate(paths)
                ],
                "categories": [{"id": 1, "name": "car"}],
                "annotations": [],
            }
        )
    )
    result = CocoImporter().import_dataset(manifest, root, tmp_path / "project")
    assert len({d.image_path for d in result.documents}) == 2
    assert (
        result.documents[0].image_path.read_bytes() != result.documents[1].image_path.read_bytes()
    )


def test_qt_arbitrary_class_save_and_folder_reopen(tmp_path, qapp):
    from app.ui.main_window import MainWindow
    from app.configs.settings import AppSettings

    path = image(tmp_path / "cat.jpg")
    doc = AnnotationDocument(
        path, 100, 80, (Annotation("cat", BoundingBox(0.1, 0.1, 0.3, 0.3), occluded=True),)
    )
    first = MainWindow(AppSettings())
    first._document = doc
    first._save_annotations()
    first.close()
    second = MainWindow(AppSettings())
    with patch("app.ui.main_window.QFileDialog.getExistingDirectory", return_value=str(tmp_path)):
        second._import_folder()
    assert second._project_documents[path].annotations[0].class_name == "cat"
    assert second._project_documents[path].annotations[0].occluded is True
    second.close()


def test_projection_failure_acknowledges_canonical_commit(dataset, monkeypatch):
    from app.services.dataset import repository

    root, path, _, client = dataset
    real_write = repository.atomic_text

    def fail_projection(target, content):
        if target.suffix == ".txt":
            raise OSError("disk projection unavailable")
        return real_write(target, content)

    monkeypatch.setattr(repository, "atomic_text", fail_projection)
    response = save(client, path.name, [box()], expected_revision=0)
    assert response.status_code == 200
    assert response.json()["revision"] == 1
    assert "projection unavailable" in response.json()["projection_warning"]
    data = AnnotationRepository(root).read(path)
    assert data["revision"] == 1 and data["boxes"][0]["occluded"] is True


def test_writers_in_separate_processes_reject_one_stale_edit(tmp_path):
    import subprocess
    import sys

    root = tmp_path / "dataset"
    path = image(root / "image.jpg")
    gate = tmp_path / "start"
    code = """
import json, sys, time
from pathlib import Path
from app.services.dataset.repository import AnnotationRepository, RevisionConflict
root, name, gate = map(Path, sys.argv[1:])
while not gate.exists(): time.sleep(.01)
try:
    AnnotationRepository(root).save(name, [], expected_revision=0)
    print('saved')
except RevisionConflict:
    print('conflict')
"""
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(root), str(path), str(gate)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    gate.touch()
    try:
        results = [worker.communicate(timeout=20) for worker in workers]
        assert all(worker.returncode == 0 for worker in workers), results
        assert sorted(out.strip() for out, _ in results) == ["conflict", "saved"]
        assert AnnotationRepository(root).read(path)["revision"] == 1
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()


def test_qt_rejects_changes_made_since_its_document_loaded(dataset, qapp):
    from app.ui.main_window import MainWindow
    from app.configs.settings import AppSettings

    root, path, _, _ = dataset
    window = MainWindow(AppSettings())
    with patch("app.ui.main_window.QFileDialog.getExistingDirectory", return_value=str(root)):
        window._import_folder()
    repo = AnnotationRepository(root)
    repo.save(path, [box()], expected_revision=0)
    with patch("app.ui.main_window.QMessageBox.critical") as error:
        window._save_annotations()
    assert error.called
    assert repo.read(path)["boxes"][0]["id"] == "one"
    window.close()


def test_changed_image_dimensions_and_review_status_are_revalidated(dataset):
    import os

    root, path, index, _ = dataset
    repo = AnnotationRepository(root)
    repo.save(path, [box()])
    index.set_metadata(path, 100, 80)
    index.set_status_and_count(path, "reviewed", 1)
    old_stat = path.stat()
    image(path, (211, 109))
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert path.stat().st_size != old_stat.st_size
    index.scan(root)
    assert index.get_image(path)["width"] is None
    data = repo.read(path)
    assert (data["width"], data["height"], data["status"]) == (211, 109, "unreviewed")


@pytest.mark.parametrize("exporter", [YoloExporter, CocoExporter])
def test_export_reports_image_copy_failures(tmp_path, monkeypatch, exporter):
    from app.export import exporters

    path = image(tmp_path / "image.jpg")
    doc = AnnotationDocument(path, 100, 80, ())

    def fail_copy(*_args):
        raise OSError("copy failed")

    monkeypatch.setattr(exporters.shutil, "copyfile", fail_copy)
    with pytest.raises(exporters.ExportError, match="copy failed"):
        exporter().export([doc], tmp_path / "out")


def test_completed_job_persists_its_own_state_when_a_new_job_starts(dataset, monkeypatch):
    root, path, _, _ = dataset
    old = {"running": True, "job_id": "original", "dataset_id": str(root)}
    new = {"running": True, "job_id": "next", "dataset_id": str(root)}

    def finish(_req, _paths, _classes, _root, status):
        status.update(running=False, completed=True)
        monkeypatch.setattr(s, "AUTOLABEL_BATCH_STATUS", new)

    monkeypatch.setattr(s, "_execute_batch_autolabel_worker", finish)
    s._run_batch_autolabel_worker(s.AutoLabelBatchRequest(), [path], [], root, old)
    persisted = json.loads((root / ".visionlab/jobs/original.json").read_text())
    assert persisted["job_id"] == "original" and persisted["completed"] is True
    assert s.AUTOLABEL_BATCH_STATUS["running"] is True
    assert not (root / ".visionlab/jobs/next.json").exists()


def test_stale_dataset_is_rejected_before_batch_preview_or_class_registration(dataset):
    root, path, _, client = dataset
    for endpoint, payload in (
        ("/api/autolabel/batch", {}),
        ("/api/autolabel/preview", {"image_name": path.name}),
        ("/api/dataset/classes", {"name": "new class"}),
    ):
        response = client.post(endpoint, json={**payload, "dataset_id": str(root / "other")})
        assert response.status_code == 409
    assert s.AUTOLABEL_BATCH_STATUS["running"] is False
    assert "new class" not in AnnotationRepository(root).classes()


def test_legacy_manual_provenance_exports_and_stable_annotation_ids(dataset):
    root, path, _, _ = dataset
    path.with_suffix(".json").write_text(json.dumps({"boxes": [box(source="manual")]}))
    repo = AnnotationRepository(root)
    first = repo.document(path)
    second = repo.document(path)
    assert str(first.annotations[0].source) == "human"
    assert first.annotations[0].annotation_id == second.annotations[0].annotation_id


def test_loading_annotations_preserves_active_learning_priority(dataset):
    _, path, index, client = dataset
    index.set_status_and_count(path, "ai_labeled", 0, 0.85)
    assert client.get(f"/api/annotations/{path.name}").status_code == 200
    assert index.get_image(path)["difficulty"] == 0.85
