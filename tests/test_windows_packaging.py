"""Exercise the spec with the globals PyInstaller supplies, outside the source cwd."""

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
    assert "pipeline_bridge" in options["hiddenimports"]
    assert "src.vlm_helper" in options["hiddenimports"]
    assert collect.call_args.kwargs["name"] == "VisionLab"
