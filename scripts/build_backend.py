#!/usr/bin/env python3
"""Freeze the Python engine with PyInstaller into ``dist/visionlab-backend``.

The Electron app (``desktop/electron-builder.yml``) ships that directory as ``resources/backend``.
Run this before ``npm run package`` in ``desktop/``.

Usage:
    python scripts/build_backend.py [--no-clean] [--console]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT_DIR / "dist"
SPEC_FILE = ROOT_DIR / "traffic_annotator.spec"
BACKEND_NAME = "visionlab-backend"


def backend_executable(directory: Path, platform: str = sys.platform) -> Path:
    return directory / (f"{BACKEND_NAME}.exe" if platform == "win32" else BACKEND_NAME)


def run_build(clean: bool = True, console: bool = False) -> Path:
    """Run PyInstaller and return the directory holding the frozen engine."""
    env = os.environ.copy()
    if console:
        env["TRAFFIC_ANNOTATOR_CONSOLE"] = "1"
    command = [sys.executable, "-m", "PyInstaller", str(SPEC_FILE), "--noconfirm"]
    if clean:
        command.append("--clean")
    print(f"[*] Running PyInstaller: {' '.join(command)}")
    subprocess.check_call(command, cwd=str(ROOT_DIR), env=env)

    target = DIST_DIR / BACKEND_NAME
    executable = backend_executable(target)
    if not executable.is_file():
        raise RuntimeError(f"The engine was not produced: {executable}")
    print(f"[+] Engine ready: {executable}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen Anotara engine")
    parser.add_argument("--no-clean", action="store_true", help="reuse PyInstaller's build cache")
    parser.add_argument("--console", action="store_true", help="keep a console window for debugging")
    args = parser.parse_args()
    try:
        run_build(clean=not args.no_clean, console=args.console)
    except Exception as error:
        print(f"[ERROR] Engine build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
