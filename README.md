# VisionLab

Universal AI-assisted visual annotation software and computer vision studio.

## Quick Start (One-Click)

Run the unified cross-platform launcher (automatically bootstraps `uv`, `.venv`, and dependencies on first run):
- **Windows**: Double-click `run.bat` (or run `run.bat` in CMD)
- **Linux**: Double-click `run.bat` / `run.sh` (or run `./run.sh` / `./run.bat` in terminal)

To force the standalone PySide6 Qt GUI instead of the modern Desktop Studio, pass `--qt`:
```bash
./run.sh --qt     # Linux
run.bat --qt      # Windows
```

## Development

```bash
uv sync --extra dev
uv run visionlab
uv run pytest
```

## Install on Ubuntu

Run the installer from this directory as the normal desktop user:

```bash
./install.sh
```

It installs the application under `~/.local/share/traffic-annotator`, adds a
`~/.local/bin/traffic-annotator` command, and creates an application-menu entry.
The installer supports Ubuntu 24.04 on x86_64 and ARM64. On Jetson systems with
`nvidia-l4t-core` installed, it requires an explicitly supplied CUDA-enabled
PyTorch wheel instead of downloading the incompatible generic PyPI wheel. The
wheel must target the installed Python version and include `sm_87` for Orin:

```bash
./install.sh --torch-wheel /path/to/torch-jetson.whl
```

An HTTPS wheel URL is also accepted. The installer verifies CUDA availability,
Orin compute capability `8.7`, and the presence of `sm_87` before completing.
Use `--cpu-only` to force the normal PyPI installation. Remove the application
with:

```bash
~/.local/share/traffic-annotator/uninstall.sh
```

The installer does not remove datasets, cache, or logs.

## Install on Windows

### Quick Install (Automated)

Run the PowerShell installer or double-click `install.bat`:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

Or for CPU-only systems:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 -CpuOnly
```

The installer:
1. Verifies or installs `uv` and Python 3.12+.
2. Copies files into `%LOCALAPPDATA%\VisionLab\app`.
3. Synchronizes locked dependencies.
4. Creates Desktop and Start Menu shortcuts (**VisionLab**).
5. Adds a `visionlab` CLI launcher to `%USERPROFILE%\.local\bin`.

To uninstall on Windows:
```powershell
powershell -ExecutionPolicy Bypass -File .\uninstall.ps1
```
Or double-click `uninstall.bat`.

### One-Click Run from Source

Double-click `run.bat` or run:

```cmd
run.bat
```

### Standalone Executable (.exe) & Packaging

To compile a standalone, self-contained Windows executable distribution (no Python installation required on the target machine):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_exe.ps1
```

Or double-click `scripts\build_windows_exe.bat`.

This produces:
- `dist\VisionLab\VisionLab.exe`: Standalone portable folder.
- `dist\VisionLab-windows-x64.zip`: Ready-to-distribute ZIP archive.

#### Creating a Windows Setup Wizard (.exe)
If you have [Inno Setup](https://jrsoftware.org/isinfo.php) installed, compile `installer.iss`:
```cmd
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
```
This generates `dist\VisionLab-Setup.exe` with a complete installation wizard.

The model adapters are intentionally isolated from the UI and will be implemented as separate features.


## Label Fusion

Label Fusion compares detections from Grounding DINO, YOLO11, and future model adapters. It
automatically accepts matching same-class detections, while sending single detections, class
conflicts, confidence disagreements, and very small boxes for review.

```python
from app.services.fusion import FusionEngine

result = FusionEngine().fuse([grounding_dino_detection, yolo_detection])
for item in result.detections:
    print(item.class_name, item.status, item.bbox)
print(result.statistics)
```

Defaults are stored in `fusion.yaml`. The public API is `FusionEngine`, `FusionConfig`,
`FusionResult`, `FusionStatistics`, `FusionStatus`, and fusion `Detection`.

## Active Learning

Active Learning ranks images by uncertainty, model disagreement, object density, occlusion,
small objects, conflicts, missing detections, duplicates, and motorcycle concentration.
Weights and thresholds are stored in `configs/active_learning.yaml`.

```python
from app.services.active_learning import (
    ActiveLearningConfig,
    ActiveLearningEngine,
    ImageAnalysis,
)

engine = ActiveLearningEngine(ActiveLearningConfig())
ranked = engine.score_many(analyses, max_workers=8)
for result in ranked[:10]:
    print(result.image_path, result.difficulty_score, result.recommended_action)
engine.close()
```

Results are cached in SQLite and automatically invalidated when detections, fusion results, or
active-learning configuration change. The desktop UI scores the active image in a background
worker after Label Fusion and displays its review priority and recommendation.

## Dataset Import (YOLO data.yaml, Roboflow & COCO)

Use `File > Import YOLO Dataset (data.yaml)` to import any Ultralytics or Roboflow YOLO dataset.
Select the `data.yaml` or `dataset.yaml` file and choose a new project destination. Supported
classes (`motorcycle`, `car`, `bus`, `truck`) and common aliases are automatically recognized,
overlapping duplicates are cleaned, and images/labels across splits are imported into the workspace.

Use `File > Import from Roboflow...` to download and import a dataset version directly from
Roboflow Universe or your private workspace using your Roboflow API key. You can paste full URLs
such as `https://universe.roboflow.com/workspace/project/dataset/1` or enter `workspace/project`.

Use `File > Upload to Roboflow...` to push images and bounding box annotations directly to a
target Roboflow project.

Use `File > Import COCO Dataset` to choose an annotations JSON, the source image directory, and a
new project destination. Supported categories are imported as bounding boxes; unsupported
categories and invalid records are reported and skipped. Images are copied into the project, so
the source dataset is never modified.

## Train/Validation/Test & Google Colab Export

Use `File > Export Dataset` and select from:
- `YOLO (Google Colab .zip)`: Creates a standalone ZIP archive containing `data.yaml`, `images/`, and `labels/` ready for 1-click Google Colab training (`model.train(data=".../data.yaml")`).
- `YOLOv11 Detection`, `YOLOv8 Detection`, `YOLOv26 Detection`: Direct directory export with both `dataset.yaml` and `data.yaml`.
- `COCO Detection`: Self-contained directory with `annotations.json`.

Choose `Train / validation / test split` with custom ratios (e.g. `0.8,0.1,0.1`) and random seed for reproducible dataset splits.

## Motorcycle and Rider Annotation

The supported classes include both `motorcycle` and `rider`. Keep the boxes separate:
the motorcycle box describes the motorcycle and the rider box describes the person riding it.
Overlapping motorcycle and rider boxes are preserved during duplicate cleanup. Use
`Annotation > DINO Annotate Entire Dataset` for Grounding DINO-only prompt-ensemble annotation.
The DINO dataset pass runs full-image and overlapping tiled inference, and accepts comma- or
period-separated prompts such as `motorcycle. rider. motorbike. motorcyclist.`
The selected annotation can be marked occluded or truncated from the Review & Cleanup actions.

Dense traffic annotation uses multi-scale YOLO proposals plus Grounding DINO proposals. The
combined dataset action keeps YOLO vehicle detections authoritative and uses DINO to supplement
motorcycles and riders. DINO-only annotation preserves existing YOLO boxes, so it can be used as
a second pass without replacing the baseline.

## Crop Assist

Use `Annotation > Crop Assist > Start Crop Assist` on crowded images. The app creates overlapping
temporary `640x640` crops with 20% overlap, or divides smaller images into four visible regions.
Existing boxes are assigned to one crop by object center, and each crop uses the normal box editor.
`Next Crop` and `Previous Crop` navigate the session; `Commit Crop Session` maps all local boxes
back to the current original image and removes crop-boundary duplicates. `Cancel Crop Session`
restores the original document without changing it.
