# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build specification for Traffic Annotator.

Produces a standalone, high-performance distribution directory containing
the executable, PySide6 GUI libraries, PyTorch runtime, model assets, and configs.
"""

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Increase recursion depth for deep PyTorch / Transformers AST traversal
sys.setrecursionlimit(5000)

block_cipher = None
PROJECT_ROOT = Path(__file__).resolve().parent

# Check if console window should be kept for debugging
console_mode = os.environ.get("TRAFFIC_ANNOTATOR_CONSOLE", "0").lower() in ("1", "true", "yes")

# Base datas: configuration files and UI assets
datas = [
    (str(PROJECT_ROOT / "configs" / "active_learning.yaml"), "configs"),
    (str(PROJECT_ROOT / "fusion.yaml"), "."),
    (str(PROJECT_ROOT / "app" / "ui" / "assets"), "app/ui/assets"),
]

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
    "app.services.integrations",
    "app.services.integrations.roboflow_client",
    "app.ui",
    "app.ui.main_window",
    "app.ui.theme",
    "app.ui.canvas",
    "app.ui.dialogs",
    "app.ui.views",
    "app.ui.widgets",
    "src.vlm_helper",
    "pipeline_bridge",
    # Qt / PySide6 modules
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
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
icon_file = PROJECT_ROOT / "resources" / "icon.ico"
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
    name="TrafficAnnotator",
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
    name="TrafficAnnotator",
)
