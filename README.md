# Anotara

> Universal AI-assisted visual annotation studio and computer vision dataset curation platform.

Anotara is a general-purpose, high-performance visual annotation platform designed to annotate **anything**—from everyday objects, people, and retail products to specialized industrial, biomedical, robotic, and agricultural imagery. Built with a modern desktop interface (Tauri v2 + React 19) backed by a local GPU-accelerated Python AI engine (YOLO11, SAM 2, Grounding DINO, and Florence-2 VLM).

---

## Quick Start (One-Click Launch)

**Already have Colab credits?** Click **Colab → Sign in with Google** in the studio
on Linux/macOS, paste Google's one-time code, then click **Start annotation**.
Anotara handles GPU setup, transfer, labeling, saving, and runtime release.
The first run defaults to three images. See the [Colab guide](docs/colab-annotation.md)
for details and the Windows notebook fallback.

**No local GPU?** Use the [cloud inference setup](docs/cloud-inference.md) to run
automatic annotation on a remote GPU with a lightweight local studio/backend.
Review, saving, and exports remain local. Cloud deployment requires a separate
provider account and configuration.

Anotara has one desktop interface: the Tauri/React studio. Source launches require Python 3.12+, uv, Node.js/npm, Rust/Cargo, and the platform's Tauri system dependencies.

| Platform | Launch |
| :--- | :--- |
| **Windows** | Double-click or run `run.bat` |
| **Linux** | Run `./run.sh` or `./run_desktop.sh` |

The launcher uses the locked Python environment, installs missing frontend dependencies, starts the local API, and opens Anotara. Missing dependencies produce an error rather than opening another interface.

---

## Table of Contents

- [Interface](#interface)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [One-Click Launch](#one-click-launch)
  - [Development Setup](#development-setup)
- [Standalone Distribution & Packaging](#standalone-distribution--packaging)
- [Core AI Engines & Programmatic API](#core-ai-engines--programmatic-api)

---

## Interface

The desktop studio uses the Anotara design: a light workspace with a sidebar and one page per
step of the dataset workflow. The interface copy is in Indonesian.

| Page | What it does |
| :--- | :--- |
| **Ringkasan** (overview) | Review progress, the next image to check, quick links to auto-label and export. |
| **Proyek** (projects) | The active dataset folder plus recently opened folders. |
| **Impor dataset** | Open a folder, drag in JPG/PNG/WEBP images, manage object classes. |
| **Auto-label** | Pick models, edit class prompts, preview on samples, run a batch and watch its progress. |
| **Review anotasi** | The annotation canvas, annotation list, audit (`Cek kualitas`), and image approval. |
| **Kualitas dataset** | Approved vs. pending counts, hardest images, and label distribution. |
| **Ekspor** | YOLO, COCO, Pascal VOC or CVAT export, optionally only approved images. |

Editing or saving annotations does not approve an image automatically. Use **Setujui gambar** to mark it as reviewed. Auto-labeled images remain pending until a person approves them.

## Keyboard Shortcuts

Press `?` for the current shortcut reference. Review shortcuts only operate on the review page:

- `V`: select a box; `B`: draw a box; `P`: draw a polygon; `H`: pan.
- `R`: run detection; `Q`: check annotation quality.
- `Tab` / `Shift+Tab`: navigate boxes; `Enter`: accept a box; `Delete`: remove a box.
- `Ctrl+S`: save and approve the image.
- `L`: auto-label; `I`: import; `E`: export.

## Key Features

- **Universal & Unbiased**: Annotate any objects without predefined class restrictions or domain constraints.
- **Multi-Model AI Foundation**: Integrates state-of-the-art architectures (Grounding DINO 1.5, YOLO11, SAM 2, Florence-2 VLM).
- **Active Learning & Disagreement Ranking**: Automatically scores image difficulty based on model disagreement, density, occlusion, and missing boxes to surface edge cases.
- **Smart Overlap Pruning & Deduplication**: Removes duplicate annotations across model proposals while preserving nested or multi-class valid detections.
- **Cross-Platform**: Standalone native desktop app for Windows (x64 installer / portable EXE) and Linux (Ubuntu 22.04 / 24.04 & Jetson Orin).

---

## Getting Started

### One-Click Launch (Unified Launcher)

Anotara provides a single polyglot launcher ([`run.bat`](file:///home/marsel/Work/ImageAnnotator/run.bat) / [`run.sh`](file:///home/marsel/Work/ImageAnnotator/run.sh)) that runs natively on both Windows and Linux.

#### On Windows
- **One-Click**: Simply double-click `run.bat` in Windows File Explorer.
- **Terminal**: Run `run.bat` from CMD or PowerShell.

#### On Linux
- **One-Click**: Double-click `run.bat` or `run.sh` in your desktop file manager (or right-click → *Run as a Program*).
- **Terminal**:
  ```bash
  ./run.sh          # Launches Tauri Desktop Studio + AI Engine
  ```

#### How the Launcher Works

1. Require uv and use `uv sync --locked` if the Python environment is missing.
2. Require the current desktop sources, Node.js/npm, Rust/Cargo, and the native Tauri prerequisites.
3. Start the FastAPI engine on `http://127.0.0.1:8765` if needed and wait for its session endpoint.
4. Launch the Anotara Tauri studio. Stop only a backend started by this launcher when the window closes.
5. Report startup failures in the application log directory; backend output is in `backend.log`.

---

### Development Setup

#### Prerequisites
- [uv](https://github.com/astral-sh/uv) (ultra-fast Python package installer)
- Python 3.12+
- Node.js & npm (compatible with the locked Vite version)
- Rust/Cargo and the platform Tauri development prerequisites

#### 1. Setup Backend
```bash
# Clone the repository
git clone https://github.com/Marsel204/VisionLabs.git
cd VisionLabs

# Sync locked Python dependencies
uv sync --extra dev

# Optional: Optuna (Bayesian) threshold search for the AI Auto-Tuner
uv sync --extra dev --extra tuning

# Run test suite
uv run pytest
```

#### 2. Setup Frontend Studio
```bash
cd desktop
npm install
npm run dev

# Frontend regression checks and production build
npm test
npm run build
npm run lint
```

The web studio is accessible at `http://localhost:1420`, connecting to the Python engine on `http://127.0.0.1:8765`.

### Annotation storage and compatibility

Anotara uses `AnnotationRepository`. Its authoritative files live at
`<dataset>/.visionlab/annotations/<relative image path including extension>.json`.
For example, `images/train/sample.jpg` becomes
`.visionlab/annotations/images/train/sample.jpg.json`. Documents retain object IDs,
confidence, provenance, review status, occlusion/truncation flags, and polygons.
SQLite stores a rebuildable image/review cache. Class IDs are the zero-based positions
in the dataset's `data.yaml` names list; adding a class preserves existing IDs.

Existing sibling, mirrored, nested, and flat YOLO/JSON labels are read on demand.
The first save creates a canonical document and refreshes derived YOLO detection
labels. Source images are preserved. Legacy labels shared by multiple image stems
are rejected because their owner cannot be determined reliably; separate them by
image before migrating. Back up the dataset before performing that reconciliation.
For colliding image names, exports allocate deterministic unique output names.
Polygons are retained in canonical JSON, YOLO export metadata, and COCO segmentation;
the YOLO `.txt` projection contains detection boxes.

Saves use atomic file replacement, a per-dataset process lock, and optimistic
revision checks. Both studios reject stale edits rather than overwrite a newer
revision. Web saves are queued per image, failed edits remain available when
navigating, and closing the page with unsaved drafts triggers a browser warning.
A failure to refresh a derived label is reported separately from a successful
canonical save; exporting always reads the canonical annotation document.

Batch jobs capture their dataset root and use their own index connection. They
preserve existing annotations, report per-image failures, and write completed
outcomes under `.visionlab/jobs/`. An empty "only unannotated" selection does no
work. Interrupted jobs are not automatically resumed; documents committed before
an interruption remain saved.

### Local API session

Keep the engine bound to `127.0.0.1`. The studio obtains a session token from
`GET /api/session`; API requests require the `X-VisionLab-Token` header. Image URLs
also accept the token query parameter. Tokens change when the backend restarts,
and the web client renews its session after a 401 response. CORS allows the local
studio development and Tauri origins. Image paths must resolve inside the active
dataset, including after symlink resolution.

Annotation clients should send `dataset_id` and `expected_revision` from their
last annotation read. A changed dataset or stale revision returns 409; reload and
reconcile the newer document before resubmitting. Geometry, confidence, and class
mapping errors return 422. Uploads validate image content and reject existing names.

Regression tests use temporary datasets and deterministic inference doubles for
HTTP contracts. Native Windows launcher execution, native browser pointer behavior,
and real GPU model accuracy still require validation on their target platforms.

---

## Standalone Distribution & Packaging

### Windows Executable (.exe)
To compile a portable Windows distribution, install Node.js/npm, Rust/Cargo, the Tauri build prerequisites, and the Python development dependencies. The build compiles the current Tauri frontend and packages it alongside the frozen Python launcher/API; no Python installation is needed on the target machine. The system webview runtime is still required:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_exe.ps1
```
Output:
- `dist/VisionLab/VisionLab.exe`: launcher that starts the local API and current studio.
- `dist/VisionLab/Anotara.exe`: current Tauri frontend; keep it beside the launcher.
- `dist/VisionLab-windows-x64.zip`: Portable ZIP release package.

### Windows Setup Wizard (.exe)
Compile `installer.iss` with [Inno Setup](https://jrsoftware.org/isinfo.php):
```cmd
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
```
Output: `dist/VisionLab-Setup.exe`.

---

## Core AI Engines & Programmatic API

Anotara also provides standalone Python services that can be imported directly into Python workflows:

### Label Fusion Engine
```python
from app.services.fusion import FusionEngine

# Fuse multi-model detections and eliminate hallucinations
result = FusionEngine().fuse([grounding_dino_detection, yolo_detection])
for item in result.detections:
    print(item.class_name, item.status, item.bbox)
```

### Active Learning Engine
```python
from app.services.active_learning import ActiveLearningConfig, ActiveLearningEngine

engine = ActiveLearningEngine(ActiveLearningConfig())
ranked_images = engine.score_many(analyses, max_workers=8)
for item in ranked_images[:10]:
    print(item.image_path, item.difficulty_score, item.recommended_action)
engine.close()
```

### AI Auto-Tuner (Optuna hyperparameter search)
The AI Auto-Tuner calibrates the confidence and IoU thresholds against your
ground-truth images. It also refines the class prompts. Thresholds are searched
over cached predictions, so inference does not run again. With the optional
`tuning` extra installed, the search uses Optuna's TPE sampler over continuous
ranges. Without it, the tuner falls back to a fixed 48-combination grid sweep.
The search always starts from your current thresholds and never returns a worse
result. Set `VISIONLAB_TUNER_SEARCH=auto|grid|optuna` in `.env` or configure it
in code:

```python
from app.services.ai_tuner import AITunerEngine, TunerConfig

config = TunerConfig.from_env(threshold_search="optuna", optuna_trials=100, optuna_seed=42)
result = AITunerEngine().run_tuning(samples, ground_truth, auto_label_config, config)
print(result.final_config.confidence_threshold, result.final_config.box_iou_threshold)
```

### Dataset Exporter
```python
from app.export.exporters import YoloExporter, split_documents

splits = split_documents(documents, train=0.7, val=0.2, test=0.1, seed=42)
exporter = YoloExporter(splits=splits)
exporter.export(documents, destination_dir)
```

---

## License

Apache 2.0 / MIT. Developed by Marsel204 for [VisionLabs](https://github.com/Marsel204/VisionLabs).

## Annotation Audit (missed objects and wrong boxes)

On the review page, press **Q** (or click **Cek kualitas** in the review toolbar) to compare the current image's boxes
against a low-threshold, tiled YOLO pass. Findings are drawn as dashed overlays and listed by
severity: possibly missed objects (relaxed thresholds when hidden behind another box),
misaligned boxes, wrong classes, duplicates, boxes with no detector support, and implausible
shapes. `J`/`K` step through findings, `Y` applies the suggested fix and `N` dismisses it.
Occlusion is estimated from box overlap (the box lower in the frame is nearer), and
**Tandai … kotak tertutup** sets the `Occluded` attribute on boxes that look at least 25% hidden.
The same engine is available as `POST /api/audit`.

The audit layers extra detector passes over the original image and merges them (pick the depth
in the panel): **Cepat** (original only), **Teliti** (default; adds a horizontally flipped
pass) or **Maksimal** (also sharpened and contrast-enhanced passes). On held-out traffic data a
flipped pass finds somewhat more real objects than just lowering the confidence threshold
(the margin varied between samples), whereas sharpen/contrast passes add recall about as cheaply
as a lower threshold would, with more review work, so they are opt-in. The **F** key / image dropdown is a separate display-only filter.

### Flip pass for detection and auto-label

The **+Flip** checkbox next to **Deteksi AI** (R key) and the flip-pass option on the
Auto-label page (preview and batch) run YOLO on a mirrored copy as well and merge the results;
boxes from the original pass are never changed, only unseen ones are added. It takes about
twice as long and produces more boxes to review, and it is local-backend only (the cloud and Colab
workers ignore it, so the UI disables it there). On held-out traffic data it raised motorcycle recall
by roughly 3 to 7 points over the original pass alone.
