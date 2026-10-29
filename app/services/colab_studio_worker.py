"""Remote entry point for an automatic VisionLab job on a Colab GPU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from app.services.auto_label.engine import AutoLabelEngine
from app.services.inference.cloud import InferenceOptions, InferenceResponse


def run(directory: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Colab did not allocate a GPU")
    request = json.loads((directory / "request.json").read_text())
    options = InferenceOptions.model_validate(request["options"])
    pipeline = AutoLabelEngine(device="cuda", allow_cloud=False, fail_on_model_error=True)
    results = {"version": 1, "job_id": request["job_id"], "images": []}
    for index, row in enumerate(request["images"], 1):
        path = directory / row["key"]
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Image path escapes job directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Input checksum mismatch")
        try:
            with Image.open(path) as image:
                image.verify()
            result = pipeline.run_preview(path, options.to_config())
            results["images"].append(
                {
                    "key": row["key"],
                    "sha256": row["sha256"],
                    "result": InferenceResponse.from_result(result).model_dump(),
                }
            )
        except Exception:
            results["images"].append(
                {
                    "key": row["key"],
                    "sha256": row["sha256"],
                    "error": "Model inference failed for this image",
                }
            )
        (directory / "result.json").write_text(json.dumps(results, allow_nan=False))
        print(
            "VISIONLAB_PROGRESS "
            + json.dumps({"completed": index, "total": len(request["images"])}),
            flush=True,
        )


def bootstrap_source() -> str:
    """The CLI executes this script; users do not upload or run notebook cells."""
    return """import hashlib, json, sys, subprocess, zipfile
from pathlib import Path, PurePosixPath
source = Path('/content/visionlab-source')
source.mkdir(exist_ok=True)
with zipfile.ZipFile('/content/visionlab-source.zip') as archive:
    manifest = json.loads(archive.read('bundle-manifest.json'))
    for name, checksum in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or chr(92) in name:
            raise ValueError('Invalid source path')
        payload = archive.read(name)
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError('Source checksum mismatch')
        target = source.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
sys.path.insert(0, str(source))
requirements = source / 'requirements-colab.txt'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(requirements)])
job = Path('/content/visionlab-job')
job.mkdir(exist_ok=True)
with zipfile.ZipFile('/content/visionlab-input.zip') as archive:
    for entry in archive.infolist():
        relative = PurePosixPath(entry.filename)
        if relative.is_absolute() or '..' in relative.parts or chr(92) in entry.filename:
            raise ValueError('Invalid input path')
        target = job.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(entry))
from app.services.colab_studio_worker import run
run(job)
"""
