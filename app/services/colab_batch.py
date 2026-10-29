"""Bounded notebook/CLI annotation batches with per-image resume and portable exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from uuid import uuid4

import yaml
from PIL import Image

from app.export.exporters import CocoExporter, YoloExporter
from app.services.auto_label.engine import AutoLabelEngine
from app.services.auto_label.models import AutoLabelClass, AutoLabelConfig, AutoLabelPipelineMode
from app.services.dataset.index import IMAGE_SUFFIXES
from app.services.dataset.repository import AnnotationRepository, atomic_text

SAMPLE_REVISION = "ed87d7c66396c24cab31023b4c76a4e7f964bd62"
SAMPLE_IMAGES = (
    (
        "000001_jpg.rf.b073bea01084e55425513802c3daf67c.jpg",
        "2fa0a76e23a46b34f13ab80c05f7746e01a2a764",
    ),
    (
        "000002_jpg.rf.f5219fa5580c366fef1dd12ca07bdb18.jpg",
        "3ab6c8402ec86ef7118a44bc47b632214c8a216f",
    ),
    (
        "000003_jpg.rf.3203c0bcf39e472da251bf3e40d38620.jpg",
        "5a5c54c319ee9cfed842771d8fba1cc317a8a6d8",
    ),
)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
PIPELINES = {
    "dino-masks": AutoLabelPipelineMode.DINO_SAM2_MASKS,
    "dino-boxes": AutoLabelPipelineMode.DINO_BOXES,
    "yolo-masks": AutoLabelPipelineMode.YOLO_SAM2_MASKS,
    "yolo-boxes": AutoLabelPipelineMode.YOLO_BOXES,
    "locate-masks": AutoLabelPipelineMode.LOCATE_ANYTHING_SAM2_MASKS,
    "locate-boxes": AutoLabelPipelineMode.LOCATE_ANYTHING_BOXES,
}


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_config(
    classes: Sequence[str],
    *,
    pipeline: str = "dino-masks",
    confidence: float = 0.35,
    prompts: dict[str, str] | None = None,
    device: str = "cuda",
    yolo_weights: str = "yolo11n.pt",
) -> AutoLabelConfig:
    names = [name.strip() for name in classes]
    if (
        not names
        or any(not name for name in names)
        or len({n.casefold() for n in names}) != len(names)
    ):
        raise ValueError("Classes must be unique nonempty names")
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between zero and one")
    mode = PIPELINES[pipeline]
    return AutoLabelConfig(
        mode=mode,
        classes=[AutoLabelClass(n, (prompts or {}).get(n, "")) for n in names],
        confidence_threshold=confidence,
        device=device,
        yolo_model_name=yolo_weights,
        yolo_models=[yolo_weights],
        enable_grounding_dino=mode.uses_grounding,
        enable_yolo=mode.uses_yolo,
        enable_sam2_masks=mode.produces_masks,
        enable_florence2=False,
        enable_locate_anything=mode.uses_locate_anything,
        enable_florence2_verifier=False,
    )


def download_example_sample(destination: Path, *, limit: int = 3) -> list[Path]:
    """Download only the explicitly pinned sample JPEGs; never clone the dataset."""
    if not 1 <= limit <= len(SAMPLE_IMAGES):
        raise ValueError("The built-in example is limited to three images")
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, oid in SAMPLE_IMAGES[:limit]:
        path = destination / name
        raw = path.read_bytes() if path.is_file() else b""
        if hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != oid:
            url = f"https://raw.githubusercontent.com/Marsel204/Dataset/{SAMPLE_REVISION}/images/{name}"
            with urlopen(url, timeout=60) as response:
                raw = response.read(MAX_IMAGE_BYTES + 1)
            if len(raw) > MAX_IMAGE_BYTES:
                raise ValueError("Sample image exceeds 20 MiB")
            if hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != oid:
                raise ValueError(f"Sample image integrity check failed: {name}")
            path.write_bytes(raw)
        with Image.open(path) as image:
            image.verify()
        paths.append(path)
    return paths


def extract_image_sample(archive_path: Path, destination: Path, *, limit: int = 3) -> list[Path]:
    """Extract only selected images, rejecting traversal, symlinks, and oversized members."""
    if limit < 1:
        raise ValueError("Image limit must be positive")
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            (
                m
                for m in archive.infolist()
                if not m.is_dir() and PurePosixPath(m.filename).suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda m: m.filename,
        )[:limit]
        if not members:
            raise ValueError("No supported images in ZIP")
        selected = []
        for member in members:
            relative = PurePosixPath(member.filename)
            target = destination.joinpath(*relative.parts)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.filename
                or not target.resolve().is_relative_to(destination)
                or stat.S_ISLNK(member.external_attr >> 16)
                or member.file_size > MAX_IMAGE_BYTES
            ):
                raise ValueError("Unsafe or oversized image entry in ZIP")
            if target.exists():
                raise ValueError("Extract into a fresh directory to preserve existing files")
            selected.append((member, target))
        for member, target in selected:
            with archive.open(member) as source:
                raw = source.read(MAX_IMAGE_BYTES + 1)
            if len(raw) > MAX_IMAGE_BYTES:
                raise ValueError("Image exceeds 20 MiB")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            with Image.open(target) as image:
                image.verify()
    return [target for _, target in selected]


def _copy_image(source: Path, target: Path, digest: str) -> None:
    if target.exists():
        if _digest(target) != digest:
            raise ValueError("Input image changed; use a new output directory")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".image-", dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _config_digest(config: AutoLabelConfig) -> str:
    root = Path(__file__).resolve().parents[2]
    sources = [
        "app/services/colab_batch.py",
        "app/services/auto_label/engine.py",
        "pipeline_bridge.py",
        "src/vlm_helper.py",
    ]
    payload = {
        "version": 1,
        "config": asdict(config),
        "code": {name: _digest(root / name) for name in sources},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _archive(source: Path, target: Path, *, exclude_exports: bool = False) -> None:
    temporary = target.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(source.rglob("*")):
                relative = path.relative_to(source)
                if path.is_file() and not (exclude_exports and relative.parts[0] == "exports"):
                    if path.name != "repository.lock":
                        archive.write(path, relative.as_posix())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def export_results(output: Path) -> dict[str, str]:
    """Produce separate portable archives; native project retains review and confidence."""
    repo = AnnotationRepository(output)
    documents = [
        repo.document(p)
        for p in sorted((output / "images").rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and repo.canonical_path(p).is_file()
    ]
    classes = repo.classes()
    exports = output / "exports"
    YoloExporter(class_order=classes).export(documents, exports / "yolo")
    CocoExporter(class_order=classes).export(documents, exports / "coco")
    archives = {}
    for kind, source in (
        ("visionlab", output),
        ("yolo", exports / "yolo"),
        ("coco", exports / "coco"),
    ):
        target = output.parent / f"{output.name}-{kind}.zip"
        _archive(source, target, exclude_exports=(kind == "visionlab"))
        archives[kind] = str(target)
    return archives


def run_batch(
    images: Path,
    output: Path,
    config: AutoLabelConfig,
    *,
    limit: int = 3,
    engine: AutoLabelEngine | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Commit each image before continuing; retries never replace saved annotations."""
    images, output = images.resolve(), output.resolve()
    if not images.is_dir() or output.is_relative_to(images) or images.is_relative_to(output):
        raise ValueError("Input and output must be separate directories")
    if limit < 1:
        raise ValueError("Image limit must be positive")
    selected = sorted(
        p
        for p in images.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and not any(part.startswith(".") for part in p.relative_to(images).parts)
    )[:limit]
    if not selected:
        raise ValueError("No supported images found")
    names = [c.name for c in config.classes if c.enabled]
    if not names or len({n.casefold() for n in names}) != len(names):
        raise ValueError("Classes must be unique nonempty names")
    marker = output / ".visionlab/colab-batch.json"
    fingerprint = _config_digest(config)
    if marker.is_file():
        state = json.loads(marker.read_text())
        if state.get("version") != 1 or state.get("config_digest") != fingerprint:
            raise ValueError("Batch configuration changed; choose a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Choose an empty output directory for a new batch")
        output.mkdir(parents=True, exist_ok=True)
        atomic_text(
            output / "data.yaml",
            yaml.safe_dump(
                {
                    "path": ".",
                    "train": "images",
                    "val": "images",
                    "names": names,
                },
                sort_keys=False,
            ),
        )
        state = {"version": 1, "config_digest": fingerprint, "entries": {}}
        atomic_text(marker, json.dumps(state, indent=2))
    repo = AnnotationRepository(output)
    if repo.classes() != names:
        raise ValueError("Dataset class registry differs from the batch configuration")
    pipeline = engine  # Lazy: a fully resumed batch does not even initialize models.
    report = {"selected": len(selected), "completed": 0, "skipped": 0, "failed": 0, "errors": {}}
    for source in selected:
        relative = source.relative_to(images)
        key = relative.as_posix()
        target = output / "images" / relative
        try:
            if not source.resolve().is_relative_to(images):
                raise ValueError("Input symlink escapes the image directory")
            if not target.resolve().is_relative_to(output):
                raise ValueError("Output symlink escapes the batch directory")
            checksum = _digest(source)
            _copy_image(source, target, checksum)
            if repo.canonical_path(target).is_file():
                report["skipped"] += 1
                event = {"image": key, "sha256": checksum, "state": "skipped"}
            else:
                if pipeline is None:
                    pipeline = AutoLabelEngine(
                        device=config.device,
                        allow_cloud=False,
                        fail_on_model_error=True,
                    )
                result = pipeline.run_preview(target, config)
                with Image.open(target) as image:
                    if (result.image_width, result.image_height) != image.size:
                        raise ValueError("Inference dimensions do not match the image")
                source_kind = str(config.mode.annotation_source)
                boxes = [
                    {
                        "id": str(uuid4()),
                        "class_name": d.class_name,
                        "class_id": names.index(d.class_name),
                        "confidence": d.confidence,
                        "source": source_kind,
                        "review_status": "pending",
                        "norm_left": d.box.left,
                        "norm_top": d.box.top,
                        "norm_right": d.box.right,
                        "norm_bottom": d.box.bottom,
                        "polygon_normalized": d.polygon_normalized or None,
                    }
                    for d in result.detections
                ]
                repo.save(target, boxes, status="ai_labeled", expected_revision=0)
                report["completed"] += 1
                event = {
                    "image": key,
                    "sha256": checksum,
                    "state": "completed",
                    "detections": len(boxes),
                }
        except Exception as error:
            report["failed"] += 1
            report["errors"][key] = str(error)
            event = {"image": key, "state": "failed", "error": str(error)}
        state["entries"][key] = event
        atomic_text(marker, json.dumps(state, indent=2))
        if progress is not None:
            progress(event)
    report["archives"] = {
        kind: str(output.parent / f"{output.name}-{kind}.zip")
        for kind in ("visionlab", "yolo", "coco")
    }
    atomic_text(output / "batch-report.json", json.dumps(report, indent=2))
    export_results(output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--pipeline", choices=PIPELINES, default="dino-masks")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args(argv)
    config = make_config(
        args.classes, pipeline=args.pipeline, device=args.device, confidence=args.confidence
    )
    try:
        result = run_batch(
            args.images,
            args.output,
            config,
            limit=args.limit,
            progress=lambda event: print(json.dumps(event), flush=True),
        )
    except KeyboardInterrupt:
        print("Interrupted; completed annotations are saved. Rerun with the same output to resume.")
        return 130
    print(json.dumps(result, indent=2))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
