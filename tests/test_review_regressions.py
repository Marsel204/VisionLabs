"""Regression coverage for annotation ownership, persistence and review findings."""

from __future__ import annotations
import asyncio
import io
import json
import tempfile
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import UploadFile
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


@pytest.mark.parametrize("canonical", [False, True])
def test_stats_response_parity(dataset, canonical):
    """Expected response captured against the pre-M5 endpoint for both storage formats."""
    root, path, index, client = dataset
    path.with_suffix(".txt").write_text("1 .2 .2 .2 .2\n1 .6 .6 .2 .2\n0 .3 .3 .2 .2\n")
    empty = image(root / "empty.jpg")
    index.scan(root)
    if canonical:
        repo = AnnotationRepository(root)
        repo.save(path, repo.read(path)["boxes"], status="reviewed")
    # Prime only the empty row: known zero-box images must never be opened by stats.
    index.set_status_and_count(empty, "unreviewed", 0)
    expected = dict(directory=str(root), total_images=2, total_annotations=3,
                    reviewed=1, unreviewed=1, ai_labeled=0,
                    class_counts={"motorcycle": 1, "car": 2, "bus": 0, "truck": 0, "minivan": 0, "person": 0},
                    database_file=str(index._database_path))
    assert client.get("/api/dataset/stats").json() == expected
    assert client.get("/api/dataset/stats").json() == expected


@pytest.mark.parametrize("canonical", [False, True])
def test_warm_stats_read_no_annotations(dataset, monkeypatch, canonical):
    root, path, _, client = dataset
    path.with_suffix(".txt").write_text("1 .2 .2 .2 .2\n")
    if canonical:
        repo = AnnotationRepository(root)
        repo.save(path, repo.read(path)["boxes"])
    first = client.get("/api/dataset/stats").json()
    monkeypatch.setattr(AnnotationRepository, "read", lambda *a, **k: pytest.fail("Warm stats read annotations"))
    original = Path.read_text

    def no_labels(path, *args, **kwargs):
        assert path.suffix not in {".txt", ".json"}, f"Warm stats read {path}"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_labels)
    assert client.get("/api/dataset/stats").json() == first


def test_stats_resolves_classes_once_and_fills_all_pages(dataset, monkeypatch):
    root, path, index, _ = dataset
    other = image(root / "other.jpg")
    for item in (path, other):
        item.with_suffix(".txt").write_text("1 .2 .2 .2 .2\n")
    index.scan(root)
    classes = AnnotationRepository.classes
    calls = []

    def counted(repo):
        calls.append(repo.root)
        return classes(repo)

    unknown = index.unknown_histograms
    monkeypatch.setattr(index, "unknown_histograms", lambda: unknown(page_size=1))
    monkeypatch.setattr(AnnotationRepository, "classes", counted)
    assert s.get_dataset_stats()["total_annotations"] == 2
    assert calls == [root]
    assert list(unknown()) == []


def test_known_empty_images_are_never_read_by_stats(dataset, monkeypatch):
    _, path, index, _ = dataset
    index.set_status_and_count(path, "reviewed", 0)
    # Include the pre-histogram schema's known-empty representation.
    index._connection.execute("UPDATE images SET class_histogram=NULL")
    index._connection.commit()
    monkeypatch.setattr(AnnotationRepository, "read", lambda *a, **k: pytest.fail("Read a known empty image"))
    result = s.get_dataset_stats()
    assert result["total_annotations"] == 0 and result["reviewed"] == 1


def test_count_only_updates_invalidate_positive_histograms(dataset):
    root, path, index, _ = dataset
    AnnotationRepository(root).save(path, [box()])
    assert index.class_counts() == {"car": 1}
    index.set_annotation_count(path, 1)
    assert index.class_counts() == {}
    assert s.get_dataset_stats()["class_counts"]["car"] == 1
    index.set_annotation_counts_batch([(path, 1)])
    assert index.class_counts() == {}
    assert s.get_dataset_stats()["class_counts"]["car"] == 1


def test_saved_document_survives_histogram_refresh_failure(dataset, monkeypatch):
    import sqlite3

    root, path, index, _ = dataset
    repo = AnnotationRepository(root)
    repo.save(path, [box()])

    def fail_refresh(*args, **kwargs):
        raise sqlite3.OperationalError("cache refresh failed")

    with monkeypatch.context() as patch:
        patch.setattr(DatasetIndex, "set_status_and_count", fail_refresh)
        saved = repo.save(path, [box(class_name="bus", class_id=2)])
    assert saved["revision"] == 2
    assert "cache refresh failed" in saved["projection_warning"]
    assert len(list(index.unknown_histograms())) == 1
    result = s.get_dataset_stats()
    assert result["class_counts"]["car"] == 0
    assert result["class_counts"]["bus"] == 1


@pytest.mark.parametrize("via_api", [False, True])
def test_annotation_overwrite_refreshes_histogram(dataset, monkeypatch, via_api):
    root, path, _, client = dataset
    repo = AnnotationRepository(root)
    repo.save(path, [box()])
    assert s.get_dataset_stats()["class_counts"]["car"] == 1
    replacement = [box(class_name="bus", class_id=2)]
    if via_api:
        assert save(client, path.name, replacement).status_code == 200
    else:
        repo.save(path, replacement)
    monkeypatch.setattr(AnnotationRepository, "read", lambda *a, **k: pytest.fail("Saved histogram not refreshed"))
    result = s.get_dataset_stats()
    assert result["class_counts"]["car"] == 0
    assert result["class_counts"]["bus"] == result["total_annotations"] == 1


def test_listing_and_annotation_reads_populate_histograms(dataset, monkeypatch):
    _, path, index, client = dataset
    path.with_suffix(".txt").write_text("1 .2 .2 .2 .2\n")
    for endpoint in ("/api/images", "/api/annotations/" + path.name):
        index.invalidate_annotations()
        assert client.get(endpoint).status_code == 200
        with monkeypatch.context() as patch:
            patch.setattr(AnnotationRepository, "read", lambda *a, **k: pytest.fail("Read did not cache histogram"))
            assert s.get_dataset_stats()["total_annotations"] == 1


def test_rescan_refreshes_external_labels_including_empty_images(dataset):
    root, path, index, client = dataset
    assert s.get_dataset_stats()["total_annotations"] == 0
    path.with_suffix(".txt").write_text("1 .2 .2 .2 .2\n")
    assert client.post("/api/dataset/rescan").status_code == 200
    assert s.get_dataset_stats()["class_counts"]["car"] == 1
    path.with_suffix(".txt").write_text("2 .2 .2 .2 .2\n")
    index.scan(root)
    result = s.get_dataset_stats()
    assert result["class_counts"]["car"] == 0
    assert result["class_counts"]["bus"] == 1


def test_external_image_change_invalidates_histogram_and_review(dataset):
    import os

    root, path, index, _ = dataset
    AnnotationRepository(root).save(path, [box()])
    assert s.get_dataset_stats()["reviewed"] == 1
    original = path.stat()
    image(path, size=(217, 113))
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    index.add_paths([(path, 217, 113)])
    assert [r["path"] for r in index.unknown_histograms()] == [str(path)]
    result = s.get_dataset_stats()
    assert result["reviewed"] == 0 and result["unreviewed"] == 1
    assert result["total_annotations"] == 1


@pytest.mark.parametrize("canonical", [False, True])
def test_stats_cache_file_is_fully_rebuildable(dataset, monkeypatch, canonical):
    root, path, index, _ = dataset
    path.with_suffix(".txt").write_text("1 .2 .2 .2 .2\n")
    repo = AnnotationRepository(root)
    if canonical:
        repo.save(path, repo.read(path)["boxes"], status="ai_labeled")
    expected = s.get_dataset_stats()
    index.close()
    index._database_path.unlink()
    monkeypatch.setattr(s, "DATASET_INDEX", None)
    assert s.get_dataset_stats() == expected


@pytest.mark.parametrize("old_count_column", [False, True])
def test_old_index_schema_upgrades_in_place(tmp_path, monkeypatch, old_count_column):
    import sqlite3

    path = image(tmp_path / "old.jpg")
    path.with_suffix(".txt").write_text("1 .2 .2 .2 .2\n")
    database = tmp_path / ".dataset_index.sqlite"
    with sqlite3.connect(database) as connection:
        count_column = ", annotation_count INTEGER NOT NULL DEFAULT -1" if old_count_column else ""
        connection.execute("CREATE TABLE images (path TEXT PRIMARY KEY, modified_ns INTEGER NOT NULL, "
                           "width INTEGER, height INTEGER, difficulty REAL NOT NULL DEFAULT 0, "
                           "status TEXT NOT NULL DEFAULT 'unreviewed'" + count_column + ")")
        connection.execute("INSERT INTO images(path, modified_ns, width, height, difficulty, status) "
                           "VALUES(?, ?, 100, 80, .75, 'reviewed')", (str(path), path.stat().st_mtime_ns))
        if old_count_column:
            connection.execute("UPDATE images SET annotation_count=1")
    with DatasetIndex(database) as index:
        before = index.get_image(path)
        assert (before["width"], before["height"], before["difficulty"], before["status"]) == (100, 80, .75, "reviewed")
        monkeypatch.setattr(s, "ACTIVE_DATASET_DIR", tmp_path)
        monkeypatch.setattr(s, "DATASET_INDEX", index)
        assert s.get_dataset_stats()["class_counts"]["car"] == 1
        assert index.get_image(path)["difficulty"] == .75
        assert list(index.unknown_histograms()) == []


def test_unreadable_histogram_remains_unknown_until_repaired(dataset):
    _, path, index, _ = dataset
    label = path.with_suffix(".txt")
    label.write_text("99 .2 .2 .2 .2\n")
    assert s.get_dataset_stats()["total_annotations"] == 0
    assert len(list(index.unknown_histograms())) == 1
    label.write_text("1 .2 .2 .2 .2\n")
    assert s.get_dataset_stats()["total_annotations"] == 1


def test_stats_scan_releases_global_dataset_lock(dataset, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    _, _, _, client = dataset
    entered, release = threading.Event(), threading.Event()
    read = AnnotationRepository.read

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return read(*args, **kwargs)

    monkeypatch.setattr(AnnotationRepository, "read", blocked)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(s.get_dataset_stats)
        try:
            assert entered.wait(5)
            assert s.DATASET_STATE_LOCK.acquire(timeout=1)
            s.DATASET_STATE_LOCK.release()
            assert client.get("/api/image/sample.jpg").status_code == 200
        finally:
            release.set()
        assert future.result(timeout=5)["total_images"] == 1


def test_stats_rejects_concurrent_rescan(dataset, monkeypatch):
    from fastapi import HTTPException

    root, _, index, _ = dataset
    read = AnnotationRepository.read

    def rescanning(*args, **kwargs):
        data = read(*args, **kwargs)
        # Change labels after they were read, before their histogram is published.
        (root / "sample.txt").write_text("2 .2 .2 .2 .2\n")
        # A separate handle also covers invalidation from another worker/process.
        with DatasetIndex(index._database_path) as other:
            other.scan(root)
        return data

    with monkeypatch.context() as patch:
        patch.setattr(AnnotationRepository, "read", rescanning)
        with pytest.raises(HTTPException) as error:
            s.get_dataset_stats()
        assert error.value.status_code == 409
    result = s.get_dataset_stats()
    assert result["total_images"] == 1
    assert result["class_counts"]["bus"] == 1


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


def assert_no_upload_staging(root):
    assert not list(root.glob(".visionlab-upload-*"))


def test_upload_invalid_mid_batch_rolls_back(dataset):
    root, path, index, client = dataset
    response = client.post("/api/dataset/upload", files=[
        ("files", ("first.jpg", path.read_bytes(), "image/jpeg")),
        ("files", ("invalid.jpg", b"invalid image", "image/jpeg")),
    ])
    assert response.status_code == 422
    assert list(root.glob("*.jpg")) == [path]
    assert index.count() == 1
    assert_no_upload_staging(root)


@pytest.mark.parametrize("names", [("new.jpg", "NEW.JPG"), ("SAMPLE.JPG",)])
def test_upload_case_insensitive_collisions(dataset, names):
    root, path, index, client = dataset
    response = client.post("/api/dataset/upload", files=[
        ("files", (name, path.read_bytes(), "image/jpeg")) for name in names
    ])
    assert response.status_code == 409
    assert list(root.glob("*.jpg")) == [path]
    assert index.count() == 1
    assert_no_upload_staging(root)


def test_upload_unsupported_suffix_cleans_staging(dataset):
    root, path, _, client = dataset
    response = client.post("/api/dataset/upload", files=[
        ("files", ("valid.jpg", path.read_bytes(), "image/jpeg")),
        ("files", ("invalid.txt", path.read_bytes(), "image/jpeg")),
    ])
    assert response.status_code == 422
    assert not (root / "valid.jpg").exists()
    assert_no_upload_staging(root)


@pytest.mark.parametrize("second_name", ["same.jpg", "SAME.JPG"])
def test_upload_concurrent_same_name_has_one_winner(dataset, monkeypatch, second_name):
    root, path, index, _ = dataset
    commit = s._commit_uploads
    barrier = threading.Barrier(2)

    def concurrent_commit(*args):
        barrier.wait(timeout=5)
        return commit(*args)

    monkeypatch.setattr(s, "_commit_uploads", concurrent_commit)

    async def run():
        async def attempt(name):
            try:
                return await s.upload_images([UploadFile(filename=name, file=io.BytesIO(path.read_bytes()))])
            except s.HTTPException as error:
                return error.status_code
        return await asyncio.gather(attempt("same.jpg"), attempt(second_name))

    results = asyncio.run(run())
    assert sum(isinstance(result, dict) for result in results) == 1
    assert 409 in results
    saved = next(result for result in results if isinstance(result, dict))
    assert saved["total_indexed"] == 2
    assert index.count() == 2
    assert len([p for p in root.iterdir() if p.name.casefold() == "same.jpg"]) == 1
    assert_no_upload_staging(root)


def test_upload_commit_keeps_event_loop_responsive(dataset, monkeypatch):
    root, path, _, _ = dataset
    release = threading.Event()
    commit = s._commit_uploads

    async def run():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()

        def blocked_commit(*args):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(timeout=5), "event loop did not release commit worker"
            return commit(*args)

        monkeypatch.setattr(s, "_commit_uploads", blocked_commit)
        task = asyncio.create_task(s.upload_images([
            UploadFile(filename="new.jpg", file=io.BytesIO(path.read_bytes()))
        ]))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            assert not task.done()
            # This separate task must tick while the synchronous commit is blocked.
            ticked = asyncio.Event()
            async def tick():
                await asyncio.sleep(0)
                ticked.set()
            await asyncio.wait_for(asyncio.create_task(tick()), timeout=1)
            assert ticked.is_set() and not release.is_set()
        finally:
            release.set()
            await task

    asyncio.run(run())
    assert_no_upload_staging(root)


def test_upload_reads_bounded_chunks_on_event_loop(dataset):
    root, path, _, _ = dataset
    payload = path.read_bytes() + b"\0" * (2 * 1024 * 1024)

    class ChunkedUpload:
        filename = "chunked.jpg"

        def __init__(self):
            self.stream = io.BytesIO(payload)
            self.read_sizes = []

        async def read(self, size=-1):
            asyncio.get_running_loop()  # The async reader must stay on the event loop.
            assert 0 < size <= 1024 * 1024, "Upload attempted whole-file buffering"
            self.read_sizes.append(size)
            return self.stream.read(size)

    upload = ChunkedUpload()
    result = asyncio.run(s.upload_images([upload]))
    assert result["count"] == 1 and len(upload.read_sizes) >= 4
    assert (root / upload.filename).read_bytes() == payload
    assert_no_upload_staging(root)


def test_upload_read_failure_cleans_partial_staging(dataset):
    root, path, index, _ = dataset

    class BrokenUpload:
        filename = "broken.jpg"
        reads = 0

        async def read(self, size=-1):
            self.reads += 1
            if self.reads == 1:
                return path.read_bytes()
            raise OSError("disconnected")

    with pytest.raises(OSError, match="disconnected"):
        asyncio.run(s.upload_images([BrokenUpload()]))
    assert not (root / "broken.jpg").exists()
    assert index.count() == 1
    assert_no_upload_staging(root)


def test_upload_spooled_reads_do_not_exhaust_worker_pool(dataset):
    from anyio import to_thread

    root, path, _, _ = dataset

    async def run():
        limiter = to_thread.current_default_thread_limiter()
        previous_tokens = limiter.total_tokens
        limiter.total_tokens = 1
        try:
            with tempfile.SpooledTemporaryFile(max_size=1) as stream:
                stream.write(path.read_bytes())
                stream.seek(0)
                upload = UploadFile(filename="spooled.jpg", file=stream)
                return await asyncio.wait_for(s.upload_images([upload]), timeout=5)
        finally:
            limiter.total_tokens = previous_tokens

    assert asyncio.run(run())["total_indexed"] == 2
    assert_no_upload_staging(root)


def test_upload_partial_write_failure_cleans_staging(dataset, monkeypatch):
    root, path, index, _ = dataset
    open_path = Path.open

    class BrokenWriter:
        def __enter__(self):
            self.stream = open_path(self.path, "xb")
            return self

        def write(self, chunk):
            self.stream.write(chunk[:10])
            raise OSError("disk full")

        def __exit__(self, *args):
            self.stream.close()

    def fail_staging_write(target, mode="r", *args, **kwargs):
        if target.parent.name.startswith(".visionlab-upload-") and mode == "xb":
            writer = BrokenWriter()
            writer.path = target
            return writer
        return open_path(target, mode, *args, **kwargs)

    payload = path.read_bytes()
    monkeypatch.setattr(Path, "open", fail_staging_write)
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(s.upload_images([UploadFile(filename="new.jpg", file=io.BytesIO(payload))]))
    assert not (root / "new.jpg").exists()
    assert index.count() == 1
    assert_no_upload_staging(root)


def test_upload_dataset_switch_rejects_commit(dataset, tmp_path, monkeypatch):
    root, path, index, _ = dataset
    other = tmp_path / "other"
    other.mkdir()
    commit = s._commit_uploads

    def switch_then_commit(*args):
        s.select_folder(str(other))
        return commit(*args)

    monkeypatch.setattr(s, "_commit_uploads", switch_then_commit)
    with pytest.raises(s.HTTPException) as error:
        asyncio.run(s.upload_images([UploadFile(filename="new.jpg", file=io.BytesIO(path.read_bytes()))]))
    assert error.value.status_code == 409
    assert not (root / "new.jpg").exists() and not (other / "new.jpg").exists()
    assert_no_upload_staging(root)


def test_upload_partial_publication_rolls_back_without_overwriting(dataset, monkeypatch):
    root, path, index, client = dataset
    link = s.os.link

    def race(source, target):
        if target.name == "second.jpg":
            target.write_bytes(b"concurrent writer")
        return link(source, target)

    monkeypatch.setattr(s.os, "link", race)
    response = client.post("/api/dataset/upload", files=[
        ("files", (name, path.read_bytes(), "image/jpeg")) for name in ["first.jpg", "second.jpg"]
    ])
    assert response.status_code == 409
    assert not (root / "first.jpg").exists()
    assert (root / "second.jpg").read_bytes() == b"concurrent writer"
    assert index.count() == 1
    assert_no_upload_staging(root)


def test_upload_index_failure_rolls_back_files_and_rows(dataset):
    root, path, index, client = dataset
    # Fail the second insert, after the first row has been added to the transaction.
    index._connection.execute("""
        CREATE TRIGGER fail_upload BEFORE INSERT ON images
        WHEN NEW.path LIKE '%/second.jpg'
        BEGIN SELECT RAISE(ABORT, 'injected index failure'); END
    """)
    response = client.post("/api/dataset/upload", files=[
        ("files", (name, path.read_bytes(), "image/jpeg")) for name in ["first.jpg", "second.jpg"]
    ])
    assert response.status_code == 409
    assert list(root.glob("*.jpg")) == [path]
    assert index.count() == 1
    assert_no_upload_staging(root)


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


def test_arbitrary_class_save_and_folder_reopen(tmp_path):
    path = image(tmp_path / "cat.jpg")
    doc = AnnotationDocument(
        path, 100, 80, (Annotation("cat", BoundingBox(0.1, 0.1, 0.3, 0.3), occluded=True),)
    )
    AnnotationRepository(tmp_path).save_document(doc)
    restored = AnnotationRepository(tmp_path).document(path)
    assert restored.annotations[0].class_name == "cat"
    assert restored.annotations[0].occluded is True


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


def test_repository_rejects_changes_made_since_its_document_loaded(dataset):
    from app.services.dataset.repository import RevisionConflict
    root, path, _, _ = dataset
    repo = AnnotationRepository(root)
    loaded = repo.read(path)
    repo.save(path, [box()], expected_revision=0)
    with pytest.raises(RevisionConflict):
        repo.save(path, loaded["boxes"], expected_revision=loaded["revision"])
    assert repo.read(path)["boxes"][0]["id"] == "one"


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
