"""Package this checkout's inference source for Colab, excluding credentials and datasets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from uuid import uuid4


def build_source_bundle(destination: Path, *, root: Path | None = None) -> Path:
    if root is None and getattr(sys, "frozen", False):
        bundled = Path(sys._MEIPASS) / "visionlab-colab-source.zip"
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            shutil.copyfile(bundled, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
    root = (root or Path(__file__).resolve().parents[2]).resolve()
    sources = [root / "pipeline_bridge.py", root / "requirements-colab.txt"]
    for directory in (root / "app", root / "src"):
        sources.extend(
            p
            for p in sorted(directory.rglob("*.py"))
            if not any(part.startswith(".") for part in p.relative_to(root).parts)
        )
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    manifest = {"version": 1, "files": {}}
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sources:
                if not path.resolve().is_relative_to(root) or path.is_symlink():
                    raise ValueError("Source bundle cannot contain external symlinks")
                relative = path.relative_to(root).as_posix()
                payload = path.read_bytes()
                archive.writestr(relative, payload)
                manifest["files"][relative] = hashlib.sha256(payload).hexdigest()
            archive.writestr("bundle-manifest.json", json.dumps(manifest, indent=2))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
