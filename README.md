# Traffic Annotator

Production-oriented desktop annotation software for Indonesian traffic imagery.

## Development

```bash
uv sync --extra dev
uv run traffic-annotator
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
`nvidia-l4t-core` installed, it preserves the system CUDA-enabled PyTorch instead
of downloading an incompatible generic PyTorch wheel. Jetson CUDA-enabled PyTorch
must already be installed for the same Python interpreter as `python3`; otherwise
the installer stops and explains how to correct it. Use `--cpu-only` to force the
normal PyPI installation. Remove the application with:

```bash
~/.local/share/traffic-annotator/uninstall.sh
```

The installer does not remove datasets, cache, or logs.

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

## COCO Cleaning

Use `File > Import COCO Dataset` to choose an annotations JSON, the source image directory, and a
new project destination. Supported categories are imported as bounding boxes; unsupported
categories and invalid records are reported and skipped. Images are copied into the project, so
the source dataset is never modified. Import automatically removes overlapping boxes, including
different-class overlaps, using the configured IoU and containment thresholds. Use
`File > Export Cleaned COCO` to write a new COCO dataset with copied images and cleaned boxes.

## Crop Assist

Use `Annotation > Crop Assist > Start Crop Assist` on crowded images. The app creates overlapping
temporary `640x640` crops with 20% overlap, or divides smaller images into four visible regions.
Existing boxes are assigned to one crop by object center, and each crop uses the normal box editor.
`Next Crop` and `Previous Crop` navigate the session; `Commit Crop Session` maps all local boxes
back to the current original image and removes crop-boundary duplicates. `Cancel Crop Session`
restores the original document without changing it.
