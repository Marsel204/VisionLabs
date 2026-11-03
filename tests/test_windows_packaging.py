"""Exercise the spec with the globals PyInstaller supplies, outside the source cwd."""

import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def test_spec_uses_pyinstaller_directory_without_dunder_file(tmp_path, monkeypatch):
    import PyInstaller.utils.hooks as hooks

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(hooks, "collect_data_files", lambda _: [])
    monkeypatch.setattr(hooks, "collect_submodules", lambda _: [])
    monkeypatch.chdir(tmp_path)
    analysis = Mock(
        return_value=SimpleNamespace(
            pure=[], zipped_data=[], scripts=[], binaries=[], zipfiles=[], datas=[]
        )
    )
    collect = Mock()
    namespace = {
        "SPECPATH": str(root),
        "Analysis": analysis,
        "PYZ": Mock(),
        "EXE": Mock(),
        "COLLECT": collect,
    }
    spec = root / "traffic_annotator.spec"
    exec(compile(spec.read_bytes(), str(spec), "exec"), namespace)
    assert "__file__" not in namespace
    args, options = analysis.call_args
    assert args[0] == [str(root / "app/main.py")]
    assert options["pathex"] == [str(root)]
    assert all(Path(source).exists() for source, _ in options["datas"])
    bundle = next(
        Path(source)
        for source, _ in options["datas"]
        if Path(source).name == "visionlab-colab-source.zip"
    )
    with zipfile.ZipFile(bundle) as archive:
        assert "requirements-colab.txt" in archive.namelist()
        assert "pipeline_bridge.py" in archive.namelist()
        assert ".env" not in archive.namelist()
    assert any(Path(source).name == "visionlab_colab_batch.ipynb" for source, _ in options["datas"])
    assert "pipeline_bridge" in options["hiddenimports"]
    assert "src.vlm_helper" in options["hiddenimports"]
    assert not any(
        name.startswith("app.ui") or name.startswith("PySide6") for name in options["hiddenimports"]
    )
    assert "app.api.server" in options["hiddenimports"]
    assert collect.call_args.kwargs["name"] == "VisionLab"


def test_distribution_includes_the_current_desktop_binary(tmp_path, monkeypatch):
    from scripts import build_windows_exe as build

    frontend = tmp_path / "current-ui"
    frontend.write_bytes(b"anotara-native")
    distribution = tmp_path / "dist" / "VisionLab"
    distribution.mkdir(parents=True)
    (distribution / ("VisionLab.exe" if build.sys.platform == "win32" else "VisionLab")).touch()
    monkeypatch.setattr(build, "DIST_DIR", distribution.parent)
    monkeypatch.setattr(build, "build_frontend", lambda: frontend)
    monkeypatch.setattr(build.subprocess, "check_call", Mock())
    assert build.run_build() == distribution
    native = distribution / ("Anotara.exe" if build.os.name == "nt" else "Anotara")
    assert native.read_bytes() == b"anotara-native"


def test_packaged_api_keeps_dataset_state_outside_the_installation(tmp_path, monkeypatch):
    from app.api import server

    user = tmp_path / "user"
    monkeypatch.setattr(server.sys, "frozen", True, raising=False)
    monkeypatch.setattr(server, "PathSettings", lambda: SimpleNamespace(
        dataset_root=user / "datasets", cache_root=user / "cache"
    ))
    dataset, config = server.dataset_state_paths(tmp_path / "readonly-installation")
    assert dataset == user / "datasets"
    assert config == user / "cache" / "active_dataset_config.json"
