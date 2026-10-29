"""Colab batches stay bounded, resumable, and portable back to the local studio."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from app.services.annotation.domain import BoundingBox
from app.services.auto_label.models import AutoLabelDetection, AutoLabelResult
from app.services.colab_batch import (
    SAMPLE_IMAGES,
    download_example_sample,
    extract_image_sample,
    make_config,
    run_batch,
)
from app.services.dataset.coco_importer import CocoImporter
from app.services.dataset.repository import AnnotationRepository
from app.services.dataset.yolo_importer import YoloImporter
from scripts.package_colab import build_source_bundle


class FakeEngine:
    def __init__(self, failures=()):
        self.calls = []
        self.failures = set(failures)

    def run_preview(self, path, config):
        self.calls.append(path.name)
        if path.name in self.failures:
            raise RuntimeError("inference unavailable")
        with Image.open(path) as image:
            width, height = image.size
        return AutoLabelResult(
            path,
            width,
            height,
            [
                AutoLabelDetection(
                    config.classes[0].name,
                    0.9,
                    BoundingBox(0.1, 0.2, 0.7, 0.8),
                    polygon_normalized=[[0.1, 0.2], [0.7, 0.2], [0.7, 0.8]],
                )
            ],
        )


@pytest.fixture
def images(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    for i in range(5):
        Image.new("RGB", (100, 80), "white").save(root / f"{i}.jpg")
    return root


def test_only_three_images_are_processed_and_all_exports_round_trip(images, tmp_path):
    output = tmp_path / "output"
    engine = FakeEngine()
    report = run_batch(images, output, make_config(["car", "helmet"]), engine=engine)
    assert report["selected"] == report["completed"] == 3
    assert report["failed"] == 0
    assert len(engine.calls) == 3
    assert len(list((output / "images").rglob("*.jpg"))) == 3
    repository = AnnotationRepository(output)
    assert repository.classes() == ["car", "helmet"]
    saved = repository.read(output / "images/0.jpg")
    assert saved["status"] == "ai_labeled"
    assert saved["boxes"][0]["confidence"] == 0.9
    assert saved["boxes"][0]["polygon_normalized"]
    assert saved["boxes"][0]["review_status"] == "pending"
    assert saved["boxes"][0]["class_id"] == 0
    coco = json.loads((output / "exports/coco/annotations.json").read_text())
    assert len(coco["images"]) == 3
    assert coco["annotations"][0]["segmentation"]
    imported_coco = CocoImporter().import_dataset(
        output / "exports/coco/annotations.json",
        output / "exports/coco/images",
        tmp_path / "coco-import",
        remove_overlaps=False,
    )
    assert imported_coco.documents[0].annotations[0].polygon_normalized
    imported_yolo = YoloImporter().import_dataset(
        output / "exports/yolo/data.yaml",
        tmp_path / "yolo-import",
        remove_overlaps=False,
    )
    assert len(imported_yolo.documents) == 3
    native = tmp_path / "extracted"
    with zipfile.ZipFile(report["archives"]["visionlab"]) as archive:
        archive.extractall(native)
        assert not any("exports/" in name for name in archive.namelist())
    restored = AnnotationRepository(native).read(native / "images/0.jpg")
    assert restored["boxes"][0]["polygon_normalized"] == saved["boxes"][0]["polygon_normalized"]
    assert restored["boxes"][0]["confidence"] == saved["boxes"][0]["confidence"]


def test_resume_retries_failures_without_repeating_completed_inference(images, tmp_path):
    output = tmp_path / "output"
    config = make_config(["car"])
    first = run_batch(images, output, config, engine=FakeEngine({"1.jpg"}))
    assert first["completed"] == 2 and first["failed"] == 1
    retry_engine = FakeEngine()
    retry = run_batch(images, output, config, engine=retry_engine)
    assert retry["completed"] == 1 and retry["skipped"] == 2
    assert retry_engine.calls == ["1.jpg"]
    image = output / "images/0.jpg"
    repo = AnnotationRepository(output)
    data = repo.read(image)
    data["boxes"][0]["confidence"] = 1.0
    repo.save(image, data["boxes"], status="reviewed")
    repeat = FakeEngine()
    run_batch(images, output, config, engine=repeat)
    assert repeat.calls == []
    assert repo.read(image)["status"] == "reviewed"
    assert repo.read(image)["boxes"][0]["confidence"] == 1.0


def test_config_change_requires_another_output_directory(images, tmp_path):
    output = tmp_path / "output"
    run_batch(images, output, make_config(["car"]), engine=FakeEngine())
    engine = FakeEngine()
    with pytest.raises(ValueError, match="configuration"):
        run_batch(images, output, make_config(["helmet"]), engine=engine)
    assert engine.calls == []
    assert AnnotationRepository(output).classes() == ["car"]


def test_changed_input_is_not_allowed_to_replace_a_saved_image(images, tmp_path):
    output = tmp_path / "output"
    config = make_config(["car"])
    run_batch(images, output, config, engine=FakeEngine())
    original = (output / "images/0.jpg").read_bytes()
    Image.new("RGB", (100, 80), "red").save(images / "0.jpg")
    report = run_batch(images, output, config, engine=FakeEngine())
    assert report["failed"] == 1
    assert (output / "images/0.jpg").read_bytes() == original


def test_interruption_keeps_completed_checkpoint_for_resume(images, tmp_path):
    output = tmp_path / "output"
    config = make_config(["car"])
    engine = FakeEngine()
    infer = engine.run_preview

    def interrupt(path, config):
        if path.name == "1.jpg":
            raise KeyboardInterrupt()
        return infer(path, config)

    engine.run_preview = interrupt
    with pytest.raises(KeyboardInterrupt):
        run_batch(images, output, config, engine=engine)
    checkpoint = json.loads((output / ".visionlab/colab-batch.json").read_text())
    assert checkpoint["entries"]["0.jpg"]["state"] == "completed"
    resume = FakeEngine()
    run_batch(images, output, config, engine=resume)
    assert resume.calls == ["1.jpg", "2.jpg"]


@pytest.mark.parametrize("classes", [[], ["car", "Car"], [" "]])
def test_invalid_class_lists_are_rejected(classes):
    with pytest.raises(ValueError):
        make_config(classes)


def test_archive_extraction_takes_only_three_pictures(tmp_path):
    image = io.BytesIO()
    Image.new("RGB", (12, 8)).save(image, format="PNG")
    source = tmp_path / "input.zip"
    with zipfile.ZipFile(source, "w") as archive:
        for i in range(8):
            archive.writestr(f"pictures/{i}.png", image.getvalue())
        archive.writestr("annotations.json", "unused")
    selected = extract_image_sample(source, tmp_path / "selected")
    assert len(selected) == 3
    assert len(list((tmp_path / "selected").rglob("*.png"))) == 3
    assert not (tmp_path / "selected/annotations.json").exists()


def test_archive_traversal_is_rejected_without_writing_outside(tmp_path):
    source = tmp_path / "input.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("../outside.png", b"bad")
    with pytest.raises(ValueError):
        extract_image_sample(source, tmp_path / "selected")
    assert not (tmp_path / "outside.png").exists()


def test_builtin_sample_downloads_only_the_three_pinned_images(tmp_path, monkeypatch):
    import hashlib

    requested = []
    image = io.BytesIO()
    Image.new("RGB", (12, 8)).save(image, format="JPEG")
    payload = image.getvalue()
    oid = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    monkeypatch.setattr(
        "app.services.colab_batch.SAMPLE_IMAGES", tuple((name, oid) for name, _ in SAMPLE_IMAGES)
    )

    def get(url, **kwargs):
        requested.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr("app.services.colab_batch.urlopen", get)
    assert len(download_example_sample(tmp_path)) == 3
    assert len(requested) == 3
    assert all("/images/" in url and "annotations.json" not in url for url in requested)
    download_example_sample(tmp_path)
    assert len(requested) == 3  # Reusing the input sample does not download it again.


def test_source_bundle_excludes_credentials_data_and_binaries(tmp_path):
    root = tmp_path / "source"
    (root / "app").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "app/__init__.py").write_text("")
    (root / "src/__init__.py").write_text("")
    (root / "pipeline_bridge.py").write_text("# inference")
    (root / "requirements-colab.txt").write_text("numpy")
    (root / ".env").write_text("PRIVATE_TOKEN=do-not-package")
    (root / "app/.private.py").write_text("private")
    (root / "app/data.jpg").write_bytes(b"image")
    path = build_source_bundle(tmp_path / "source.zip", root=root)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        assert (
            ".env" not in names and "app/.private.py" not in names and "app/data.jpg" not in names
        )
        manifest = json.loads(archive.read("bundle-manifest.json"))
        assert names == set(manifest["files"]) | {"bundle-manifest.json"}


def test_frozen_source_download_uses_the_shipped_bundle(tmp_path, monkeypatch):
    import sys

    bundled = tmp_path / "frozen"
    bundled.mkdir()
    source = bundled / "visionlab-colab-source.zip"
    source.write_bytes(b"prebuilt-source-archive")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundled), raising=False)
    target = build_source_bundle(tmp_path / "download/source.zip")
    assert target.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("frozen", [False, True])
def test_unsupported_backend_cannot_start_google_login(tmp_path, monkeypatch, frozen):
    import sys
    from unittest.mock import Mock

    from app.services.colab_connection import ColabConnection, ColabConnectionError

    monkeypatch.setattr(sys, "platform", "linux" if frozen else "win32")
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    connection = ColabConnection(tmp_path / "private")
    setup = Mock()
    monkeypatch.setattr(connection, "_ensure_cli", setup)
    with pytest.raises(ColabConnectionError, match="use the notebook"):
        connection.connect()
    setup.assert_not_called()
    assert not connection.profile.exists()


def test_notebook_cells_compile_and_default_to_the_three_image_sample():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/visionlab_colab_batch.ipynb").read_text())
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), cell["id"], "exec")
    settings = next(c for c in notebook["cells"] if c["id"] == "settings")
    namespace = {"json": json}
    exec(compile("".join(settings["source"]), "settings", "exec"), namespace)
    assert namespace["LIMIT"] == 3
    assert namespace["DATA_SOURCE"] == "GitHub sample (3 images)"
    assert namespace["CONFIG"].enable_grounding_dino
    assert not namespace["CONFIG"].enable_yolo


def test_notebook_bootstrap_loads_the_local_bundle(tmp_path, monkeypatch):
    import sys
    import tempfile
    import types

    root = Path(__file__).resolve().parents[1]
    bundle = build_source_bundle(tmp_path / "source.zip", root=root)
    notebook = json.loads((root / "notebooks/visionlab_colab_batch.ipynb").read_text())
    cell = next(c for c in notebook["cells"] if c["id"] == "source")
    extracted = tmp_path / "unpacked"
    extracted.mkdir()
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kwargs: str(extracted))
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {"source.zip": bundle.read_bytes()})
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    original_path = list(sys.path)
    try:
        exec(compile("".join(cell["source"]), "source", "exec"), {})
        assert (extracted / "app/services/colab_batch.py").read_bytes() == (
            root / "app/services/colab_batch.py"
        ).read_bytes()
    finally:
        sys.path[:] = original_path
