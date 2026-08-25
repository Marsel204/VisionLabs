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
same-class overlaps, using the configured IoU and containment thresholds. Motorcycle and rider
boxes are intentionally preserved even when they overlap. Use
`File > Export Cleaned COCO` to write a new COCO dataset with copied images and cleaned boxes.

## Train/Validation/Test Export

Use `File > Export Dataset`, select an export format, and choose `Train / validation / test split`.
Enter ratios such as `0.8,0.1,0.1` and a seed to create a reproducible split. YOLO exports use
`images/{train,val,test}` and `labels/{train,val,test}` with matching paths in `dataset.yaml`.
COCO exports create one self-contained directory and `annotations.json` per split.

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
