import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import main

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_interfaces_and_template_assets_are_absent():
    for relative in (
        "app/ui",
        "docs/images",
        "docs/screenshots",
        "scripts/visual_verification.py",
        "desktop/src/assets",
        "desktop/public/icons.svg",
    ):
        assert not (ROOT / relative).exists(), relative
    assert "PySide6" not in (ROOT / "pyproject.toml").read_text()


def test_source_launch_has_only_the_current_desktop_frontend(tmp_path, monkeypatch):
    desktop = tmp_path / "desktop"
    (desktop / "node_modules").mkdir(parents=True)
    (desktop / "package.json").write_text("{}")
    monkeypatch.setattr(main.shutil, "which", lambda name: f"/tools/{name}")
    command, directory = main.frontend_command(tmp_path)
    assert command[-2:] == ["run", "electron:dev"]
    assert directory == desktop


def test_missing_desktop_dependencies_never_fall_back_to_a_different_ui(tmp_path, monkeypatch):
    monkeypatch.setattr(main.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="Anotara"):
        main.frontend_command(tmp_path)


def test_frozen_engine_never_opens_a_window(tmp_path, monkeypatch):
    monkeypatch.setattr(main.sys, "frozen", True, raising=False)
    with pytest.raises(RuntimeError, match="Anotara"):
        main.frontend_command(tmp_path)


def test_frozen_engine_runs_only_the_api(tmp_path, monkeypatch):
    import uvicorn

    from app.configs import settings
    from app.core import logging as app_logging

    config = SimpleNamespace(
        ensure_directories=Mock(), paths=SimpleNamespace(log_root=tmp_path), log_level="INFO"
    )
    monkeypatch.setattr(settings, "load_settings", lambda _: config)
    monkeypatch.setattr(app_logging, "configure_logging", Mock())
    monkeypatch.setattr(main.sys, "frozen", True, raising=False)
    start_api = Mock()
    monkeypatch.setattr(uvicorn, "run", start_api)
    monkeypatch.setattr(main, "launch_studio", Mock(side_effect=AssertionError("no window")))
    assert main.main([]) == 0
    start_api.assert_called_once()


def test_launcher_stops_only_the_backend_it_started(tmp_path, monkeypatch):
    backend = Mock()
    backend.poll.return_value = None
    monkeypatch.setattr(main, "frontend_command", lambda _: (["frontend"], tmp_path))
    monkeypatch.setattr(main, "session_ready", lambda: False)
    monkeypatch.setattr(main, "wait_for_backend", lambda process: None)
    monkeypatch.setattr(main.subprocess, "Popen", Mock(return_value=backend))
    monkeypatch.setattr(main.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0)))
    assert main.launch_studio(tmp_path, tmp_path) == 0
    backend.terminate.assert_called_once()
    backend.wait.assert_called_once()


def test_launcher_reuses_an_existing_backend_without_terminating_it(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "frontend_command", lambda _: (["frontend"], tmp_path))
    monkeypatch.setattr(main, "session_ready", lambda: True)
    start = Mock()
    monkeypatch.setattr(main.subprocess, "Popen", start)
    monkeypatch.setattr(main.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0)))
    assert main.launch_studio(tmp_path, tmp_path) == 0
    start.assert_not_called()


def test_backend_startup_failure_is_reported_and_cleaned_up(tmp_path, monkeypatch):
    backend = Mock()
    backend.poll.return_value = None
    monkeypatch.setattr(main, "frontend_command", lambda _: (["frontend"], tmp_path))
    monkeypatch.setattr(main, "session_ready", lambda: False)
    monkeypatch.setattr(main, "wait_for_backend", Mock(side_effect=RuntimeError("not ready")))
    monkeypatch.setattr(main.subprocess, "Popen", Mock(return_value=backend))
    frontend = Mock()
    monkeypatch.setattr(main.subprocess, "run", frontend)
    with pytest.raises(RuntimeError, match="not ready"):
        main.launch_studio(tmp_path, tmp_path)
    backend.terminate.assert_called_once()
    frontend.assert_not_called()


def test_retired_launch_flag_is_not_accepted():
    with pytest.raises(SystemExit):
        main.build_parser().parse_args(["--qt"])


def test_desktop_is_electron_not_tauri():
    package = json.loads((ROOT / "desktop/package.json").read_text())
    assert package["main"] == "electron/main.mjs"
    assert "electron" in package["devDependencies"]
    assert "electron-updater" in package["dependencies"]
    assert not any(name.startswith("@tauri-apps") for name in package["devDependencies"])
    assert not (ROOT / "desktop/src-tauri").exists()


def test_native_window_uses_the_anotara_title():
    main_process = (ROOT / "desktop/electron/main.mjs").read_text()
    assert "Anotara - AI Annotation Studio" in main_process


def test_native_icon_uses_the_current_logo_not_the_starter_logo():
    from PIL import Image

    with Image.open(ROOT / "desktop/build/icon.png") as icon:
        colors = {pixel[:3] for pixel in icon.convert("RGBA").getdata() if pixel[3] == 255}
    assert colors == {(47, 111, 228)}


def test_api_trusts_the_electron_origin_and_no_retired_shell():
    from app.api import server

    assert "app://anotara" in server.ALLOWED_ORIGINS
    assert not any("tauri" in origin for origin in server.ALLOWED_ORIGINS)


def test_backend_mode_never_launches_a_desktop(tmp_path, monkeypatch):
    import uvicorn

    from app.configs import settings
    from app.core import logging as app_logging

    config = SimpleNamespace(
        ensure_directories=Mock(), paths=SimpleNamespace(log_root=tmp_path), log_level="INFO"
    )
    monkeypatch.setattr(settings, "load_settings", lambda _: config)
    monkeypatch.setattr(app_logging, "configure_logging", Mock())
    start_api = Mock()
    monkeypatch.setattr(uvicorn, "run", start_api)
    desktop = Mock(side_effect=AssertionError("must not open another UI"))
    monkeypatch.setattr(main, "launch_studio", desktop)
    assert main.main(["--backend"]) == 0
    start_api.assert_called_once()
    desktop.assert_not_called()
