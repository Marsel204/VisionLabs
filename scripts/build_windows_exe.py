#!/usr/bin/env python3
"""Build script to generate a standalone Windows executable distribution using PyInstaller.

Usage:
    python scripts/build_windows_exe.py [--clean] [--zip] [--console]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT_DIR / "dist"
SPEC_FILE = ROOT_DIR / "traffic_annotator.spec"


def check_prerequisites() -> None:
    """Ensure required build tools are available."""
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("[!] PyInstaller not found. Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller>=6.10"])


def run_build(clean: bool = True, console: bool = False) -> Path:
    """Execute PyInstaller build."""
    env = os.environ.copy()
    if console:
        env["TRAFFIC_ANNOTATOR_CONSOLE"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(SPEC_FILE),
        "--noconfirm",
    ]
    if clean:
        cmd.append("--clean")

    print(f"[*] Running PyInstaller build: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(ROOT_DIR), env=env)

    target_dir = DIST_DIR / "VisionLab"
    if not target_dir.is_dir():
        raise RuntimeError(f"Build failed: output directory {target_dir} was not created.")

    # Determine executable name
    exe_name = "VisionLab.exe" if sys.platform == "win32" else "VisionLab"
    exe_path = target_dir / exe_name
    if not exe_path.exists():
        found = list(target_dir.iterdir())[:5]
        print(f"[!] Warning: Expected executable {exe_path} not found (files: {found}...)")
    else:

        print(f"[+] Executable created successfully: {exe_path}")

    # Create a quick-launch batch file inside the distribution folder
    launch_bat = target_dir / "Launch_VisionLab.bat"
    launch_bat.write_text(
        "@echo off\r\n"
        "cd /d \"%~dp0\"\r\n"
        "start \"\" \"VisionLab.exe\" %*\r\n",
        encoding="utf-8",
    )

    # Create a portable README inside the distribution folder
    readme_txt = target_dir / "README.txt"
    readme_txt.write_text(
        "VisionLab - Windows Standalone Distribution\r\n"
        "===========================================\r\n\r\n"
        "To start the application:\r\n"
        "  - Double-click 'VisionLab.exe' or 'Launch_VisionLab.bat'.\r\n\r\n"
        "Features:\r\n"
        "  - Fully self-contained (no Python installation required).\r\n"
        "  - CUDA acceleration enabled automatically if NVIDIA GPU and drivers are present.\r\n"
        "  - Datasets are stored in '%USERPROFILE%\\VisionLab\\datasets'.\r\n"
        "  - Cache and logs are stored in '%LOCALAPPDATA%\\VisionLab'.\r\n",
        encoding="utf-8",
    )

    return target_dir


def create_zip(target_dir: Path) -> Path:
    """Package the standalone directory into a zip archive for release."""
    zip_path = DIST_DIR / "VisionLab-windows-x64.zip"
    print(f"[*] Packaging {target_dir} into {zip_path}...")
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in target_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(DIST_DIR)
                zf.write(file, arcname)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[+] Archive created successfully: {zip_path} ({size_mb:.1f} MB)")
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Traffic Annotator standalone executable")
    parser.add_argument(
        "--clean", action="store_true", default=True, help="Clean cache before building"
    )
    parser.add_argument(
        "--zip", action="store_true", default=True, help="Create a distribution zip archive"
    )
    parser.add_argument(
        "--console", action="store_true", help="Keep console window open for debugging"
    )
    args = parser.parse_args()


    try:
        check_prerequisites()
        output_dir = run_build(clean=args.clean, console=args.console)
        if args.zip:
            create_zip(output_dir)
        print("\n[SUCCESS] Windows standalone build complete!")
        return 0
    except Exception as error:
        print(f"\n[ERROR] Build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
