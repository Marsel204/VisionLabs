"""PyInstaller build specification for Anotara.

Produces the Anotara launcher and Python API runtime. The build script adds the
current Tauri desktop executable to the same distribution directory.
"""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Increase recursion depth for deep PyTorch / Transformers AST traversal
sys.setrecursionlimit(5000)

block_cipher = None
# PyInstaller executes specs without __file__; SPECPATH is its spec directory.
PROJECT_ROOT = Path(SPECPATH).resolve()

# Check if console window should be kept for debugging
console_mode = (
    os.environ.get("VISIONLAB_CONSOLE", os.environ.get("TRAFFIC_ANNOTATOR_CONSOLE", "0")).lower()
    in ("1", "true", "yes")
)

# Base datas: configuration files and UI assets
datas = [
    (str(PROJECT_ROOT / "configs" / "active_learning.yaml"), "configs"),
    (str(PROJECT_ROOT / "fusion.yaml"), "."),
]

# Frozen backends offer the notebook fallback without needing unpacked Python source.
sys.path.insert(0, str(PROJECT_ROOT))
from app.services.colab_bundle import build_source_bundle

colab_resources = TemporaryDirectory(prefix="visionlab-colab-build-")
colab_bundle = build_source_bundle(
    Path(colab_resources.name) / "visionlab-colab-source.zip", root=PROJECT_ROOT
)
datas.extend([
    (str(colab_bundle), "."),
    (str(PROJECT_ROOT / "notebooks" / "visionlab_colab_batch.ipynb"), "."),
])

if (PROJECT_ROOT / ".env.example").is_file():
    datas.append((str(PROJECT_ROOT / ".env.example"), "."))

# Collect data files from dynamic third-party libraries if present
for pkg in ("ultralytics", "transformers", "timm"):
    try:
        datas.extend(collect_data_files(pkg))
    except Exception:
        pass

# Collect submodules
hidden_imports = [
    # Application packages
    "app",
    "app.main",
    "app.configs",
    "app.configs.settings",
    "app.core",
    "app.core.exceptions",
    "app.core.logging",
    "app.core.runtime",
    "app.export",
    "app.export.exporters",
    "app.models",
    "app.models.contracts",
    "app.models.adapters",
    "app.services",
    "app.services.active_learning",
    "app.services.active_learning.active_learning_engine",
    "app.services.active_learning.active_learning_models",
    "app.services.active_learning.difficulty_score",
    "app.services.active_learning.disagreement",
    "app.services.active_learning.hard_examples",
    "app.services.active_learning.ranking",
    "app.services.active_learning.statistics",
    "app.services.active_learning.uncertainty",
    "app.services.ai_tuner",
    "app.services.ai_tuner.models",
    "app.services.ai_tuner.engine",
    "app.services.ai_tuner.evaluator",
    "app.services.ai_tuner.parametric",
    "app.services.annotation",
    "app.services.annotation.domain",
    "app.services.auto_label",
    "app.services.auto_label.models",
    "app.services.crop_assisted",
    "app.services.dataset",
    "app.services.dataset.index",
    "app.services.dataset.coco_importer",
    "app.services.dataset.yolo_importer",
    "app.services.fusion",
    "app.services.fusion.fusion_models",
    "app.services.inference",
    "app.services.inference.locate_anything",
    "app.services.integrations",
    "app.services.integrations.roboflow_client",
    "app.api.server",
    "app.api.colab_routes",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "src.vlm_helper",
    "pipeline_bridge",
    # Data & ML libraries
    "yaml",
    "sqlite3",
    "cv2",
    "PIL",
    "numpy",
    "torch",
    "torchvision",
    "ultralytics",
    "transformers",
    "timm",
    "peft",
    "decord",
    "einops",
    "lmdb",
    "bitsandbytes",
]

# Additional submodules for ultralytics and timm
for pkg in ("ultralytics", "timm"):
    try:
        hidden_imports.extend(collect_submodules(pkg))
    except Exception:
        pass

# Optional icon
icon_file = PROJECT_ROOT / "desktop" / "src-tauri" / "icons" / "icon.ico"
icon_path = str(icon_file) if icon_file.is_file() else None

a = Analysis(
    [str(PROJECT_ROOT / "app" / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6",
        "shiboken6",
        "tkinter",
        "unittest",
        "pytest",
        "ruff",
        "mypy",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VisionLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=console_mode,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="VisionLab",
)
