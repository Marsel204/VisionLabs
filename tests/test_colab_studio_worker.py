"""Remote job and bootstrap contracts using synthetic pictures and no GPU."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from app.services.annotation.domain import BoundingBox
from app.services.auto_label.models import AutoLabelDetection, AutoLabelResult
from app.services.colab_batch import make_config
from app.services.colab_bundle import build_source_bundle
from app.services.colab_studio_worker import bootstrap_source, run
from app.services.inference.cloud import InferenceOptions


def request(directory):
    rows = []
    for i in range(3):
        path = directory / f"image-{i}.jpg"
        Image.new("RGB", (100, 80)).save(path)
        rows.append({"key": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    payload = {
        "version": 1,
        "job_id": "test-job",
        "images": rows,
        "options": InferenceOptions.from_config(make_config(["car"])).model_dump(),
    }
    (directory / "request.json").write_text(json.dumps(payload))
    return payload


def test_remote_worker_requires_gpu_and_does_not_fall_back_to_cpu(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    engine = Mock()
    monkeypatch.setattr("app.services.colab_studio_worker.AutoLabelEngine", engine)
    with pytest.raises(RuntimeError, match="did not allocate a GPU"):
        run(tmp_path)
    engine.assert_not_called()


def test_worker_returns_each_selected_image_and_retains_masks_on_partial_failure(
    tmp_path, monkeypatch, capsys
):
    payload = request(tmp_path)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    )
    calls = []

    def infer(path, config):
        calls.append(path.name)
        assert config.enable_sam2_masks and not config.enable_yolo
        if len(calls) == 2:
            raise RuntimeError("Internal diagnostic must not be returned")
        return AutoLabelResult(
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

    engine = Mock(return_value=SimpleNamespace(run_preview=infer))
    monkeypatch.setattr("app.services.colab_studio_worker.AutoLabelEngine", engine)
    run(tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    assert calls == [row["key"] for row in payload["images"]]
    assert len(result["images"]) == 3
    assert result["images"][0]["result"]["detections"][0]["polygon"]
    assert "error" in result["images"][1]
    assert "Internal diagnostic" not in json.dumps(result)
    assert len(capsys.readouterr().out.splitlines()) == 3
    engine.assert_called_once_with(device="cuda", allow_cloud=False, fail_on_model_error=True)


@pytest.mark.parametrize("unsafe", [False, True])
def test_remote_bootstrap_checks_source_and_safe_input_paths(tmp_path, monkeypatch, unsafe):
    # Use a minimal source fixture and replace only the remote /content prefix.
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pipeline_bridge.py").write_text("# inference source\n")
    (root / "requirements-colab.txt").write_text("# test only\n")
    build_source_bundle(tmp_path / "visionlab-source.zip", root=root)
    with zipfile.ZipFile(tmp_path / "visionlab-input.zip", "w") as archive:
        archive.writestr("../escape.txt" if unsafe else "request.json", "{}")
    install = Mock()
    worker = Mock()
    monkeypatch.setattr("subprocess.check_call", install)
    monkeypatch.setattr("app.services.colab_studio_worker.run", worker)
    monkeypatch.setattr(sys, "path", list(sys.path))
    code = compile(
        bootstrap_source().replace("/content", str(tmp_path)), "remote-bootstrap", "exec"
    )
    if unsafe:
        with pytest.raises(ValueError, match="Invalid input path"):
            exec(code, {})
        worker.assert_not_called()
    else:
        exec(code, {})
        worker.assert_called_once_with(tmp_path / "visionlab-job")
    install.assert_called_once()
    assert not (tmp_path / "escape.txt").exists()
