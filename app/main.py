"""Run the Anotara local Python API, or launch the Electron studio from a source checkout."""

from __future__ import annotations

import argparse
import io
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()
LOGGER = logging.getLogger(__name__)


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anotara: aplikasi anotasi AI")
    parser.add_argument("--config", type=Path, help="path to a JSON configuration file")
    parser.add_argument("--backend", action="store_true", help="run only the local API")
    return parser


def frontend_command(root: Path) -> tuple[list[str], Path]:
    """Command that opens the Electron desktop shell from a source checkout."""
    if getattr(sys, "frozen", False):
        raise RuntimeError(
            "The packaged engine has no window of its own; start the Anotara desktop app instead."
        )
    desktop = root / "desktop"
    npm = shutil.which("npm")
    if not (desktop / "package.json").is_file() or not npm:
        raise RuntimeError("Anotara requires its desktop sources and Node.js/npm.")
    command = [npm]
    if os.name == "nt":
        command = [os.environ.get("COMSPEC", "cmd.exe"), "/c", npm]
    if not (desktop / "node_modules").is_dir():
        subprocess.run([*command, "ci"], cwd=desktop, check=True)
    return [*command, "run", "electron:dev"], desktop


def session_ready() -> bool:
    try:
        with urlopen("http://127.0.0.1:8765/api/session", timeout=1) as response:
            payload = json.load(response)
            return (
                isinstance(payload, dict)
                and isinstance(payload.get("token"), str)
                and bool(payload["token"])
            )
    except (URLError, OSError, ValueError):
        return False


def wait_for_backend(process: subprocess.Popen, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Anotara API exited before becoming ready; check backend.log.")
        if session_ready():
            return
        time.sleep(0.25)
    raise RuntimeError("Anotara API did not become ready; check backend.log.")


def launch_studio(root: Path, log_root: Path, config: Path | None = None) -> int:
    command, directory = frontend_command(root)
    environment = os.environ.copy()
    backend = None
    log_root.mkdir(parents=True, exist_ok=True)
    with (log_root / "backend.log").open("a", encoding="utf-8") as log:
        try:
            if not session_ready():
                backend_command = [sys.executable]
                if not getattr(sys, "frozen", False):
                    backend_command.extend(["-m", "app.main"])
                backend_command.append("--backend")
                if config is not None:
                    backend_command.extend(["--config", str(config.resolve())])
                backend = subprocess.Popen(
                    backend_command,
                    cwd=root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                wait_for_backend(backend)
            return subprocess.run(command, cwd=directory, env=environment, check=False).returncode
        finally:
            if backend is not None and backend.poll() is None:
                backend.terminate()
                try:
                    backend.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    backend.kill()
                    backend.wait()


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    args = build_parser().parse_args(argv)
    from app.configs.settings import load_settings
    from app.core.logging import configure_logging

    try:
        settings = load_settings(args.config)
        settings.ensure_directories()
        configure_logging(settings.paths.log_root, settings.log_level)
        # A frozen build only ever ships the engine; the Electron app owns the window.
        if args.backend or getattr(sys, "frozen", False):
            import uvicorn

            uvicorn.run("app.api.server:app", host="127.0.0.1", port=8765, log_level="warning")
            return 0
        return launch_studio(get_base_dir(), settings.paths.log_root, args.config)
    except KeyboardInterrupt:
        return 130
    except Exception:
        LOGGER.exception("Anotara startup failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
