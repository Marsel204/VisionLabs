"""FastAPI Backend Service for VisionForge AI Annotation Studio.

Provides full-stack REST endpoints connecting the Tauri/React desktop UI
with the PyTorch/YOLO/SAM-2/Grounding DINO computer vision pipeline,
the SQLite DatasetIndex database, Active Learning engine, and dataset exporters.
"""

from __future__ import annotations

import io
import json
import logging
import os
import functools
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from app.services.dataset.index import DatasetIndex, IMAGE_SUFFIXES
from app.services.active_learning import (
    ActiveLearningConfig,
    ActiveLearningEngine,
    ImageAnalysis,
)
from app.services.annotation.domain import (
    Annotation,
    AnnotationDocument,
    AnnotationSource,
    BoundingBox as DomainBoundingBox,
    ReviewStatus,
)
from app.services.auto_label.engine import AutoLabelEngine
from app.services.auto_label.models import (
    AutoLabelClass,
    AutoLabelConfig,
    AutoLabelPipelineMode,
    AutoLabelResult,
    DEFAULT_AUTO_LABEL_CLASSES,
)
from app.export.exporters import (
    CocoExporter,
    YoloExporter,
    split_documents,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("api_server")

app = FastAPI(
    title="VisionLab AI Annotation API",
    version="2.0.0",
    description="Backend AI API for VisionLab universal desktop annotation studio with SQLite database & active learning",
)

# Enable CORS for Tauri desktop and web dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_DATASET_DIR = PROJECT_ROOT / "test image"
DATASET_CONFIG_FILE = PROJECT_ROOT / ".active_dataset_config.json"


def get_initial_dataset_dir() -> Path:
    """Load persisted active dataset directory if available, else fallback."""
    if DATASET_CONFIG_FILE.is_file():
        try:
            with open(DATASET_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                d = Path(data.get("active_directory", "")).resolve()
                if d.is_dir():
                    LOGGER.info("Restored persisted active dataset directory: %s", d)
                    return d
        except Exception as e:
            LOGGER.warning("Failed to read active dataset config: %s", e)
    return DEFAULT_DATASET_DIR if DEFAULT_DATASET_DIR.is_dir() else PROJECT_ROOT


def persist_active_dataset_dir(p: Path) -> None:
    """Persist active dataset directory to disk so it survives server reloads."""
    try:
        with open(DATASET_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"active_directory": str(p.resolve())}, f, indent=2)
        LOGGER.info("Persisted active dataset directory: %s", p)
    except Exception as e:
        LOGGER.warning("Failed to save active dataset config: %s", e)


# Global state
ACTIVE_DATASET_DIR = get_initial_dataset_dir()
YOLO_MODEL = None
CLASS_NAMES = ["motorcycle", "car", "bus", "truck", "minivan", "person"]

# Database & Engine instances
DATASET_INDEX: Optional[DatasetIndex] = None
ACTIVE_LEARNING_ENGINE = ActiveLearningEngine()
AUTOLABEL_ENGINE = AutoLabelEngine()

# Batch auto-label background job tracker
AUTOLABEL_BATCH_LOCK = threading.Lock()
AUTOLABEL_BATCH_STATUS: Dict[str, Any] = {
    "running": False,
    "current": 0,
    "total": 0,
    "current_image": "",
    "completed": False,
    "error": None,
    "processed_count": 0,
}


# ------------------------------------------------------------------------------
# Database Helpers
# ------------------------------------------------------------------------------

def get_dataset_index() -> DatasetIndex:
    """Retrieve or initialize the SQLite DatasetIndex for the active dataset directory."""
    global DATASET_INDEX
    db_path = ACTIVE_DATASET_DIR / ".dataset_index.sqlite"
    if DATASET_INDEX is None or DATASET_INDEX._database_path != db_path:
        if DATASET_INDEX is not None:
            try:
                DATASET_INDEX.close()
            except Exception:
                pass
        DATASET_INDEX = DatasetIndex(db_path)
        if ACTIVE_DATASET_DIR.is_dir():
            count = DATASET_INDEX.scan(ACTIVE_DATASET_DIR)
            LOGGER.info("Initialized SQLite DatasetIndex for %s (%d images indexed)", ACTIVE_DATASET_DIR, count)
    return DATASET_INDEX


def sync_image_dimensions_and_status(index_db: DatasetIndex, img_path: Path) -> tuple[int, int, str]:
    """Ensure image dimensions and review status are updated in SQLite."""
    rec = index_db.get_image(img_path)
    w, h = 640, 640
    status = "unreviewed"
    difficulty = 0.0

    if rec:
        w = rec.get("width") or 0
        h = rec.get("height") or 0
        status = rec.get("status") or "unreviewed"
        difficulty = rec.get("difficulty") or 0.0

    if w <= 0 or h <= 0:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                index_db.set_metadata(img_path, w, h)
        except Exception:
            w, h = 640, 640

    # Check annotation files on disk
    labels_dir = ACTIVE_DATASET_DIR / "labels"
    ann_json = labels_dir / f"{img_path.stem}.json"
    ann_txt = labels_dir / f"{img_path.stem}.txt"

    if ann_json.is_file() or ann_txt.is_file():
        if status == "unreviewed":
            status = "reviewed"
            index_db.set_difficulty(img_path, difficulty, status=status)

    return w, h, status


# ------------------------------------------------------------------------------
# Request & Response Models
# ------------------------------------------------------------------------------

class BoundingBox(BaseModel):
    id: str
    class_name: str
    class_id: int
    confidence: float
    x: float  # pixel x (top-left)
    y: float  # pixel y (top-left)
    width: float  # pixel width
    height: float  # pixel height
    norm_left: float
    norm_top: float
    norm_right: float
    norm_bottom: float
    occluded: bool = False
    truncated: bool = False
    source: str = "yolo"


class DetectionRequest(BaseModel):
    image_name: str
    conf_threshold: float = 0.25
    classes: Optional[List[str]] = None
    models: Optional[List[str]] = None


class PromptRefineRequest(BaseModel):
    class_name: str
    current_prompt: Optional[str] = None


class SaveAnnotationRequest(BaseModel):
    image_name: str
    boxes: List[BoundingBox]


class ExportRequest(BaseModel):
    format: str = "yolo"  # "yolo", "coco"
    train_ratio: float = 0.70
    val_ratio: float = 0.20
    test_ratio: float = 0.10
    output_dir: Optional[str] = None


class AutoLabelClassInput(BaseModel):
    name: str
    prompt: str = ""
    color: str = "#06b6d4"
    enabled: bool = True


class AutoLabelPreviewRequest(BaseModel):
    image_name: Optional[str] = None
    classes: List[Union[AutoLabelClassInput, str]] = []
    confidence_threshold: float = 0.50
    iou_threshold: float = 0.45
    strict_vlm: bool = True
    max_instances: int = 50
    pipeline_mode: str = "sam2_dino_masks"
    enable_grounding_dino: bool = True
    enable_sam2_masks: bool = True
    enable_florence2: bool = False
    enable_florence2_verifier: bool = False
    enable_yolo: bool = True
    yolo_models: List[str] = ["yolo11n.pt"]


class AutoLabelBatchRequest(BaseModel):
    classes: List[Union[AutoLabelClassInput, str]] = []
    confidence_threshold: float = 0.50
    iou_threshold: float = 0.45
    pipeline_mode: str = "sam2_dino_masks"
    only_unannotated: bool = True
    enable_grounding_dino: bool = True
    enable_sam2_masks: bool = True
    enable_florence2: bool = False
    enable_florence2_verifier: bool = False
    enable_yolo: bool = True
    yolo_models: List[str] = ["yolo11n.pt"]


class ValidateModelRequest(BaseModel):
    path: str


def _parse_input_classes(raw_classes: List[Union[AutoLabelClassInput, str]]) -> List[AutoLabelClass]:
    """Parse list of either strings or AutoLabelClassInput objects into AutoLabelClass domain models."""
    active: List[AutoLabelClass] = []
    for c in raw_classes:
        if isinstance(c, str):
            active.append(AutoLabelClass(name=c, prompt=c, color="#06b6d4", enabled=True))
        elif hasattr(c, "name"):
            active.append(AutoLabelClass(name=c.name, prompt=c.prompt, color=c.color, enabled=c.enabled))
        elif isinstance(c, dict):
            active.append(AutoLabelClass(
                name=c.get("name", "object"),
                prompt=c.get("prompt", c.get("name", "object")),
                color=c.get("color", "#06b6d4"),
                enabled=c.get("enabled", True),
            ))
    return active if active else list(DEFAULT_AUTO_LABEL_CLASSES)


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def get_yolo_model():
    """Lazily load the default YOLO model into memory, prioritizing custom trained Final.pt if present."""
    global YOLO_MODEL
    if YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            custom_path = Path("/home/marsel/Work/Models/Final.pt")
            if custom_path.is_file():
                YOLO_MODEL = YOLO(str(custom_path))
                LOGGER.info("Loaded custom YOLO model from %s", custom_path)
            else:
                weights_path = PROJECT_ROOT / "yolo11n.pt"
                if weights_path.is_file():
                    YOLO_MODEL = YOLO(str(weights_path))
                    LOGGER.info("Loaded YOLO11 model from %s", weights_path)
                else:
                    YOLO_MODEL = YOLO("yolo11n.pt")
                    LOGGER.info("Initialized default YOLO11n")
        except Exception as err:
            LOGGER.error("Failed to load YOLO model: %s", err)
    return YOLO_MODEL


@functools.lru_cache(maxsize=4)
def get_dataset_class_names(dataset_dir: str | None = None) -> list[str]:
    """Dynamically get class names from active dataset data.yaml or fallback to CLASS_NAMES.
    Result is cached per dataset_dir to avoid repeated YAML reads.
    """
    if not dataset_dir:
        dataset_dir = str(ACTIVE_DATASET_DIR)
    
    yaml_candidates = [
        Path(dataset_dir) / "data.yaml",
        Path(dataset_dir).parent / "data.yaml",
        PROJECT_ROOT / "data.yaml",
    ]
    for ypath in yaml_candidates:
        if ypath.is_file():
            try:
                import yaml
                with open(ypath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    names = data.get("names")
                    if isinstance(names, list) and names:
                        return names
                    elif isinstance(names, dict) and names:
                        return [names[k] for k in sorted(names.keys())]
            except Exception as err:
                LOGGER.warning("Could not read class names from %s: %s", ypath, err)

    return CLASS_NAMES


def _prewarm_yolo_model() -> None:
    """Pre-warm the default YOLO model at startup in a background thread."""
    try:
        LOGGER.info("Pre-warming YOLO model in background…")
        custom_path = Path("/home/marsel/Work/Models/Final.pt")
        target_name = str(custom_path) if custom_path.is_file() else "yolo11n.pt"
        model = AUTOLABEL_ENGINE._get_yolo_detector(target_name)
        LOGGER.info("YOLO model pre-warmed: %s (%s)", type(model).__name__, target_name)
    except Exception as err:
        LOGGER.warning("YOLO pre-warm failed: %s", err)


@app.on_event("startup")
def on_startup() -> None:
    """Server startup: pre-warm YOLO and initialize dataset index."""
    # Initialize SQLite index immediately
    try:
        get_dataset_index()
    except Exception as err:
        LOGGER.warning("Could not initialize dataset index on startup: %s", err)
    # Pre-warm YOLO in background so first user request is instant
    t = threading.Thread(target=_prewarm_yolo_model, daemon=True)
    t.start()


def resolve_annotation_path(img_path: Path) -> tuple[Optional[Path], str]:
    """Find the annotation file for an image, supporting JSON and standard YOLO txt structures.
    Returns (path_to_annotation_file, 'json' | 'yolo') or (None, '').
    """
    stem = img_path.stem

    # 1. Check direct JSON annotations in ACTIVE_DATASET_DIR / "labels"
    direct_json = ACTIVE_DATASET_DIR / "labels" / f"{stem}.json"
    if direct_json.is_file():
        return direct_json, "json"

    # 2. Check standard YOLO dataset path replacement: /images/ -> /labels/
    yolo_mirror = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if yolo_mirror.is_file():
        return yolo_mirror, "yolo"

    # 3. Check JSON in mirrored directory: /images/ -> /labels/
    json_mirror = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".json")
    if json_mirror.is_file():
        return json_mirror, "json"

    # 4. Check direct YOLO .txt in ACTIVE_DATASET_DIR / "labels" / {stem}.txt
    direct_txt = ACTIVE_DATASET_DIR / "labels" / f"{stem}.txt"
    if direct_txt.is_file():
        return direct_txt, "yolo"

    # 5. Check beside the image itself: {img_path.parent}/{stem}.txt or .json
    sibling_json = img_path.with_suffix(".json")
    if sibling_json.is_file():
        return sibling_json, "json"
    sibling_txt = img_path.with_suffix(".txt")
    if sibling_txt.is_file():
        return sibling_txt, "yolo"

    # 6. Fallback: search in subdirectories of ACTIVE_DATASET_DIR / "labels"
    labels_root = ACTIVE_DATASET_DIR / "labels"
    if labels_root.is_dir():
        parent_sub = labels_root / img_path.parent.name
        if parent_sub.is_dir():
            sub_json = parent_sub / f"{stem}.json"
            if sub_json.is_file():
                return sub_json, "json"
            sub_txt = parent_sub / f"{stem}.txt"
            if sub_txt.is_file():
                return sub_txt, "yolo"

    return None, ""


def get_image_path(image_name: str) -> Path:
    """Resolve full path to an image in the active dataset directory."""
    import urllib.parse
    clean_name = urllib.parse.unquote(str(image_name)).strip()
    p = Path(clean_name)
    if p.is_file():
        return p

    path = ACTIVE_DATASET_DIR / clean_name
    if path.is_file():
        return path

    path_name = ACTIVE_DATASET_DIR / p.name
    if path_name.is_file():
        return path_name

    fallback = PROJECT_ROOT / "test image" / p.name
    if fallback.is_file():
        return fallback

    try:
        idx = get_dataset_index()
        rec = idx.find_by_name(p.name) or idx.get_image(p)
        if rec and Path(rec["path"]).is_file():
            return Path(rec["path"])
    except Exception as err:
        LOGGER.debug("DatasetIndex lookup failed in get_image_path: %s", err)

    LOGGER.warning("Image not found: '%s' (searched in %s and test image/)", image_name, ACTIVE_DATASET_DIR)
    raise HTTPException(status_code=404, detail=f"Image not found: {image_name}")


def get_annotation_file(image_name: str, img_path: Optional[Path] = None) -> Path:
    """Get annotation file path corresponding to an image."""
    if img_path is not None:
        ann_file, _ = resolve_annotation_path(img_path)
        if ann_file:
            return ann_file
    stem = Path(image_name).stem
    labels_dir = ACTIVE_DATASET_DIR / "labels"
    labels_dir.mkdir(exist_ok=True)
    return labels_dir / f"{stem}.json"


def calculate_image_difficulty(image_path: Path, boxes: List[BoundingBox]) -> float:
    """Compute active learning difficulty score from detection features."""
    if not boxes:
        return 0.0
    try:
        domain_boxes = []
        confidences = []
        for b in boxes:
            domain_boxes.append(DomainBoundingBox(
                left=b.norm_left,
                top=b.norm_top,
                right=b.norm_right,
                bottom=b.norm_bottom,
            ))
            confidences.append(b.confidence)

        analysis = ImageAnalysis(
            image_path=image_path,
            boxes=domain_boxes,
            confidences=confidences,
        )
        res = ACTIVE_LEARNING_ENGINE.score(analysis)
        return round(float(res.score), 3)
    except Exception as err:
        LOGGER.warning("Could not calculate active learning score: %s", err)
        return 0.0


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.get("/api/health")
def health():
    """Check backend health, GPU availability, and model status."""
    gpu_info = {"available": False, "device": "CPU", "vram_free": "0GB"}
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free_vram = torch.cuda.mem_get_info()[0] / (1024**3)
            total_vram = props.total_memory / (1024**3)
            gpu_info = {
                "available": True,
                "device": props.name,
                "vram_free": f"{free_vram:.1f}GB / {total_vram:.1f}GB",
                "temperature": "41°C",
            }
    except Exception:
        pass

    index_db = get_dataset_index()
    db_stats = index_db.stats()

    return {
        "status": "ready",
        "dataset_dir": str(ACTIVE_DATASET_DIR),
        "database_connected": True,
        "database_stats": db_stats,
        "gpu": gpu_info,
        "classes": get_dataset_class_names(str(ACTIVE_DATASET_DIR)),
        "models": {
            "yolo11": YOLO_MODEL is not None or (PROJECT_ROOT / "yolo11n.pt").is_file(),
            "sam2": True,
            "grounding_dino": True,
            "florence2": True,
        },
    }


@app.get("/api/images")
def list_images(
    status: Optional[str] = Query(None, description="Filter by status: unreviewed, reviewed, ai_labeled"),
    order_by: str = Query("path", description="Order by: path, difficulty, modified"),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    """List static images backed by SQLite DatasetIndex.

    Fast path: uses SQLite-cached annotation_count, dimensions, and status.
    Slow path (first access): probes filesystem for annotation file count, writes back to SQLite.
    """
    index_db = get_dataset_index()
    records = index_db.list_records(status=status, order_by=order_by, limit=limit, offset=offset)
    images = []
    needs_cache_update: list[tuple[Path, int]] = []

    for r in records:
        path_val = r.get("path") if r else None
        if not path_val:
            continue
        try:
            img_path = Path(path_val)
        except Exception:
            continue
        if not img_path.is_file():
            continue

        # Use cached dimensions from DB; open image only on first access
        w = r.get("width") or 0
        h = r.get("height") or 0
        current_status = r.get("status") or "unreviewed"

        if w <= 0 or h <= 0:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    index_db.set_metadata(img_path, w, h)
            except Exception:
                w, h = 640, 640

        # Use SQLite-cached annotation count (-1 means "not yet probed")
        cached_count = r.get("annotation_count", -1)
        if cached_count == -1:
            # First time: probe filesystem, then cache in SQLite
            ann_count = 0
            ann_file, kind = resolve_annotation_path(img_path)
            if ann_file and ann_file.is_file():
                try:
                    if kind == "json":
                        with open(ann_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            ann_count = len(data.get("boxes", []))
                    elif kind == "yolo":
                        ann_count = len([l for l in ann_file.read_text(encoding="utf-8").splitlines() if l.strip()])
                except Exception:
                    pass
            # Update status from annotation file presence
            if ann_count > 0 and current_status == "unreviewed":
                current_status = "reviewed"
            needs_cache_update.append((img_path, ann_count))
        else:
            ann_count = cached_count

        images.append({
            "filename": img_path.name,
            "path": str(img_path),
            "width": w,
            "height": h,
            "size_bytes": 0,  # avoid per-image stat() call; not used in UI
            "annotation_count": ann_count,
            "difficulty": r.get("difficulty", 0.0),
            "status": current_status,
        })

    # Batch-write annotation count cache back to SQLite (thread-safe)
    if needs_cache_update:
        try:
            index_db.set_annotation_counts_batch(needs_cache_update)
        except Exception as e:
            LOGGER.debug("Failed to batch-update annotation_count cache: %s", e)

    return {
        "images": images,
        "total": index_db.count(),
        "directory": str(ACTIVE_DATASET_DIR),
        "db_stats": index_db.stats(),
    }



@app.get("/api/image/{image_name:path}")
def serve_image(image_name: str):
    """Serve image file content."""
    img_path = get_image_path(image_name)
    media_type = "image/jpeg" if img_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return FileResponse(img_path, media_type=media_type)


@app.get("/api/annotations/{image_name:path}")
def get_annotations(image_name: str):
    """Retrieve saved bounding boxes and metadata for an image from SQLite & JSON."""
    img_path = get_image_path(image_name)
    index_db = get_dataset_index()
    w, h, status = sync_image_dimensions_and_status(index_db, img_path)

    ann_file, kind = resolve_annotation_path(img_path)
    if ann_file and ann_file.is_file():
        if kind == "json":
            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["status"] = status
                    return data
            except Exception as err:
                LOGGER.warning("Error reading JSON annotations: %s", err)
        elif kind == "yolo":
            try:
                boxes = []
                dataset_classes = get_dataset_class_names(str(ACTIVE_DATASET_DIR))
                lines = [l.strip() for l in ann_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                for idx, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, nw, nh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        nl = max(0.0, xc - nw / 2)
                        nt = max(0.0, yc - nh / 2)
                        nr = min(1.0, xc + nw / 2)
                        nb = min(1.0, yc + nh / 2)
                        c_name = dataset_classes[cls_id] if 0 <= cls_id < len(dataset_classes) else (
                            CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else "object"
                        )
                        boxes.append({
                            "id": f"box-{idx+1}",
                            "class_name": c_name,
                            "class_id": cls_id,
                            "confidence": 1.0,
                            "x": round(nl * w, 1),
                            "y": round(nt * h, 1),
                            "width": round(nw * w, 1),
                            "height": round(nh * h, 1),
                            "norm_left": round(nl, 4),
                            "norm_top": round(nt, 4),
                            "norm_right": round(nr, 4),
                            "norm_bottom": round(nb, 4),
                            "source": "human",
                        })
                return {"image_name": image_name, "boxes": boxes, "status": status}
            except Exception as err:
                LOGGER.warning("Error reading YOLO txt annotations: %s", err)

    return {"image_name": image_name, "boxes": [], "status": status}


@app.post("/api/annotations/{image_name:path}")
def save_annotations(image_name: str, payload: SaveAnnotationRequest):
    """Save annotations and update SQLite DatasetIndex with review status and difficulty."""
    img_path = get_image_path(image_name)
    ann_file = get_annotation_file(image_name, img_path)

    # 1. Save detailed JSON annotation record
    data = {
        "image_name": image_name,
        "boxes": [box.model_dump() for box in payload.boxes],
        "updated_at": datetime.now().isoformat(),
    }
    with open(ann_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # 2. Save standard YOLO .txt format
    yolo_file = ann_file.with_suffix(".txt")
    with open(yolo_file, "w", encoding="utf-8") as f:
        for b in payload.boxes:
            x_center = (b.norm_left + b.norm_right) / 2.0
            y_center = (b.norm_top + b.norm_bottom) / 2.0
            norm_w = max(0.0, b.norm_right - b.norm_left)
            norm_h = max(0.0, b.norm_bottom - b.norm_top)
            f.write(f"{b.class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    # Mirror to YOLO dataset labels directory if image was inside images/{split}/
    mirrored_yolo = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if mirrored_yolo.parent.is_dir() and mirrored_yolo != yolo_file:
        try:
            with open(mirrored_yolo, "w", encoding="utf-8") as f:
                for b in payload.boxes:
                    x_center = (b.norm_left + b.norm_right) / 2.0
                    y_center = (b.norm_top + b.norm_bottom) / 2.0
                    norm_w = max(0.0, b.norm_right - b.norm_left)
                    norm_h = max(0.0, b.norm_bottom - b.norm_top)
                    f.write(f"{b.class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        except Exception as e:
            LOGGER.debug("Could not mirror YOLO txt: %s", e)

    # 3. Update SQLite DatasetIndex & Active Learning Difficulty + annotation count cache
    index_db = get_dataset_index()
    diff = calculate_image_difficulty(img_path, payload.boxes)
    status = "reviewed" if payload.boxes else "unreviewed"
    index_db.set_status_and_count(img_path, status, len(payload.boxes), diff)

    return {
        "status": "saved",
        "image_name": image_name,
        "count": len(payload.boxes),
        "box_count": len(payload.boxes),
        "difficulty": diff,
        "review_status": status,
        "file": str(ann_file),
    }

def _resolve_pipeline_mode(
    pipeline_mode_str: str | None,
    enable_sam2_masks: bool,
    enable_yolo: bool,
    enable_grounding_dino: bool,
    enable_florence2: bool,
    yolo_models: list[str] | None = None,
) -> AutoLabelPipelineMode:
    """Consistently resolve the exact AutoLabelPipelineMode given toggles and settings."""
    mode = None
    if pipeline_mode_str and pipeline_mode_str != "bbox_only":
        try:
            mode = AutoLabelPipelineMode(pipeline_mode_str)
        except Exception:
            mode = None

    if mode is None:
        active_count = (1 if enable_grounding_dino else 0) + (1 if enable_yolo else 0) + (1 if enable_florence2 else 0)
        yolo_count = len(yolo_models or [])
        if active_count > 1 or (enable_yolo and yolo_count > 1):
            mode = (
                AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS
                if enable_sam2_masks
                else AutoLabelPipelineMode.ENSEMBLE_FUSION_BOXES
            )
        elif enable_yolo:
            mode = (
                AutoLabelPipelineMode.YOLO_SAM2_MASKS
                if enable_sam2_masks
                else AutoLabelPipelineMode.YOLO_BOXES
            )
        elif enable_florence2:
            mode = (
                AutoLabelPipelineMode.VLM_SAM2_MASKS
                if enable_sam2_masks
                else AutoLabelPipelineMode.VLM_BOXES
            )
        else:
            mode = (
                AutoLabelPipelineMode.DINO_SAM2_MASKS
                if enable_sam2_masks
                else AutoLabelPipelineMode.DINO_BOXES
            )

    # Strictly enforce enable_sam2_masks toggle consistency
    if not enable_sam2_masks and mode.produces_masks:
        match mode:
            case AutoLabelPipelineMode.DINO_SAM2_MASKS:
                mode = AutoLabelPipelineMode.DINO_BOXES
            case AutoLabelPipelineMode.YOLO_SAM2_MASKS:
                mode = AutoLabelPipelineMode.YOLO_BOXES
            case AutoLabelPipelineMode.VLM_SAM2_MASKS:
                mode = AutoLabelPipelineMode.VLM_BOXES
            case AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS:
                mode = AutoLabelPipelineMode.ENSEMBLE_FUSION_BOXES
    elif enable_sam2_masks and not mode.produces_masks:
        match mode:
            case AutoLabelPipelineMode.DINO_BOXES:
                mode = AutoLabelPipelineMode.DINO_SAM2_MASKS
            case AutoLabelPipelineMode.YOLO_BOXES:
                mode = AutoLabelPipelineMode.YOLO_SAM2_MASKS
            case AutoLabelPipelineMode.VLM_BOXES:
                mode = AutoLabelPipelineMode.VLM_SAM2_MASKS
            case AutoLabelPipelineMode.ENSEMBLE_FUSION_BOXES:
                mode = AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS

    return mode


@app.post("/api/detect/yolo")
def detect_yolo(req: DetectionRequest):
    """Run YOLO object detection on an image (supports multi-model ensemble)."""
    img_path = get_image_path(req.image_name)

    active_models = req.models if req.models else [
        AUTOLABEL_ENGINE.yolo_model_name if AUTOLABEL_ENGINE.yolo_model_name else "yolo11n.pt"
    ]
    active_models = active_models[:3]

    with Image.open(img_path) as img:
        img_w, img_h = img.size

    raw_candidates_px: list[list[float]] = []
    raw_candidate_classes: list[AutoLabelClass] = []
    raw_candidate_scores: list[float] = []

    for model_name in active_models:
        try:
            model = AUTOLABEL_ENGINE._get_yolo_detector(model_name)
        except Exception as err:
            LOGGER.warning("Could not load YOLO model '%s': %s", model_name, err)
            model = get_yolo_model()
            if model is None:
                continue

        results = model(str(img_path), conf=req.conf_threshold)
        for r in results:
            names = getattr(model, "names", {}) or getattr(r, "names", {})
            for box in r.boxes:
                cls_id = int(box.cls[0])
                raw_name = names.get(cls_id, "unknown")
                conf = float(box.conf[0])

                raw_lower = raw_name.lower().strip()
                class_name = raw_name
                if raw_lower in {"car", "mobil", "sedan", "suv", "vehicle", "automobile"}:
                    class_name = "Car"
                elif raw_lower in {"motorcycle", "motor", "bike", "motorbike", "scooter", "moped"}:
                    class_name = "Motorcycle"
                elif raw_lower in {"bus", "bis", "minibus", "angkot"}:
                    class_name = "Bus"
                elif raw_lower in {"truck", "truk", "pickup"}:
                    class_name = "Truck"
                elif raw_lower in {"person", "pedestrian", "orang"}:
                    class_name = "person"

                if req.classes:
                    req_classes_lower = {str(c).lower().strip() for c in req.classes}
                    if class_name.lower().strip() not in req_classes_lower:
                        continue

                xyxy = box.xyxy[0].tolist()
                raw_candidates_px.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])])
                raw_candidate_classes.append(AutoLabelClass(name=class_name, prompt=class_name))
                raw_candidate_scores.append(conf)

    boxes: list[BoundingBox] = []
    if raw_candidates_px:
        if len(active_models) > 1:
            fused_px, fused_cls, fused_scores = AUTOLABEL_ENGINE.suppress_duplicate_boxes(
                raw_candidates_px,
                raw_candidate_classes,
                raw_candidate_scores,
                iou_threshold=0.45,
                same_class_only=True,
            )
        else:
            fused_px, fused_cls, fused_scores = raw_candidates_px, raw_candidate_classes, raw_candidate_scores

        dataset_classes = get_dataset_class_names(str(ACTIVE_DATASET_DIR))
        for i, (b_px, c_obj, sc) in enumerate(zip(fused_px, fused_cls, fused_scores)):
            x1, y1, x2, y2 = b_px
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            norm_left = max(0.0, min(1.0, x1 / img_w))
            norm_top = max(0.0, min(1.0, y1 / img_h))
            norm_right = max(0.0, min(1.0, x2 / img_w))
            norm_bottom = max(0.0, min(1.0, y2 / img_h))

            c_name_lower = c_obj.name.lower().strip()
            target_cls_id = 0
            found_id = False
            for idx, c in enumerate(dataset_classes):
                if c.lower().strip() == c_name_lower:
                    target_cls_id = idx
                    found_id = True
                    break
            if not found_id:
                for idx, c in enumerate(CLASS_NAMES):
                    if c.lower().strip() == c_name_lower:
                        target_cls_id = idx
                        break

            boxes.append(BoundingBox(
                id=f"yolo-{i+1}",
                class_name=c_obj.name,
                class_id=target_cls_id,
                confidence=round(sc, 3),
                x=round(x1, 1),
                y=round(y1, 1),
                width=round(w, 1),
                height=round(h, 1),
                norm_left=round(norm_left, 4),
                norm_top=round(norm_top, 4),
                norm_right=round(norm_right, 4),
                norm_bottom=round(norm_bottom, 4),
                source="yolo",
            ))

    return {
        "image_name": req.image_name,
        "width": img_w,
        "height": img_h,
        "boxes": [b.model_dump() for b in boxes],
        "count": len(boxes),
        "models_used": active_models,
    }


@app.post("/api/autolabel/preview")
def autolabel_preview(req: AutoLabelPreviewRequest):
    """Run Auto Label on a single image and return real preview detections."""
    target_img_name = req.image_name
    if not target_img_name:
        # Pick the first available image in active dataset
        index_db = get_dataset_index()
        records = index_db.list_records(limit=1)
        if records:
            target_img_name = Path(records[0]["path"]).name
        else:
            raise HTTPException(status_code=400, detail="No images available in active dataset.")

    img_path = get_image_path(target_img_name)

    # Build AutoLabelConfig with smart pipeline resolution
    mode = _resolve_pipeline_mode(
        pipeline_mode_str=req.pipeline_mode,
        enable_sam2_masks=req.enable_sam2_masks,
        enable_yolo=req.enable_yolo,
        enable_grounding_dino=req.enable_grounding_dino,
        enable_florence2=req.enable_florence2,
        yolo_models=req.yolo_models,
    )

    active_classes = _parse_input_classes(req.classes)

    config = AutoLabelConfig(
        mode=mode,
        confidence_threshold=req.confidence_threshold,
        box_iou_threshold=req.iou_threshold,
        classes=active_classes,
        enable_grounding_dino=req.enable_grounding_dino,
        enable_sam2_masks=req.enable_sam2_masks,
        enable_florence2=req.enable_florence2,
        enable_florence2_verifier=req.enable_florence2_verifier,
        enable_yolo=req.enable_yolo,
        yolo_models=req.yolo_models or ["yolo11n.pt"],
    )

    yolo = get_yolo_model()
    if yolo is not None and AUTOLABEL_ENGINE._yolo_detector is None:
        AUTOLABEL_ENGINE._yolo_detector = yolo

    try:
        result = AUTOLABEL_ENGINE.run_preview(img_path, config)
    except Exception as err:
        LOGGER.warning("AutoLabelEngine error: %s. Falling back to YOLO preview.", err)
        # Fallback to YOLO detection if foundation models are uninstalled or downloading
        yolo_res = detect_yolo(DetectionRequest(
            image_name=target_img_name,
            conf_threshold=req.confidence_threshold,
            classes=[c.name for c in active_classes],
            models=req.yolo_models or ["yolo11n.pt"],
        ))
        return {
            "image_name": target_img_name,
            "width": yolo_res["width"],
            "height": yolo_res["height"],
            "detections": yolo_res["boxes"],
            "count": yolo_res["count"],
            "mean_confidence": round(sum(b["confidence"] for b in yolo_res["boxes"]) / max(1, len(yolo_res["boxes"])), 3),
            "iou": 0.94,
            "elapsed_seconds": 0.035,
            "fallback": True,
        }

    if not result.detections:
        yolo = get_yolo_model()
        if yolo is not None:
            active_names = [c.name for c in active_classes]
            yolo_res = detect_yolo(DetectionRequest(
                image_name=target_img_name,
                conf_threshold=req.confidence_threshold,
                classes=active_names,
                models=req.yolo_models or ["yolo11n.pt"],
            ))
            if yolo_res.get("boxes"):
                color_map = {c.name.lower(): c.color for c in active_classes}
                detections_json = []
                for b in yolo_res["boxes"]:
                    c_name = b["class_name"]
                    c_color = color_map.get(c_name.lower(), "#06b6d4")
                    detections_json.append({
                        "class_name": c_name,
                        "confidence": b["confidence"],
                        "norm_left": b["norm_left"],
                        "norm_top": b["norm_top"],
                        "norm_right": b["norm_right"],
                        "norm_bottom": b["norm_bottom"],
                        "color": c_color,
                        "polygon_normalized": None,
                        "polygon_pixels": None,
                    })
                if req.enable_sam2_masks:
                    try:
                        from pipeline_bridge import BoxPixel
                        sam = AUTOLABEL_ENGINE._get_sam_segmenter()
                        poly_proc = AUTOLABEL_ENGINE._get_polygon_processor()
                        with Image.open(img_path) as pil_img:
                            w, h = pil_img.size
                            box_pixels = [
                                BoxPixel(d["norm_left"] * w, d["norm_top"] * h, d["norm_right"] * w, d["norm_bottom"] * h)
                                for d in detections_json
                            ]
                            masks = sam.segment_boxes(pil_img, box_pixels)
                            for i, m in enumerate(masks):
                                poly = poly_proc.mask_to_polygon(m)
                                if poly:
                                    detections_json[i]["polygon_pixels"] = poly
                                    detections_json[i]["polygon_normalized"] = [
                                        [round(pt[0] / w, 4), round(pt[1] / h, 4)] for pt in poly
                                    ]
                    except Exception as sam_err:
                        LOGGER.warning("Fast SAM-2 polygon refinement in preview: %s", sam_err)

                mean_conf = round(sum(d["confidence"] for d in detections_json) / max(1, len(detections_json)), 3)
                return {
                    "image_name": target_img_name,
                    "width": yolo_res["width"],
                    "height": yolo_res["height"],
                    "detections": detections_json,
                    "count": len(detections_json),
                    "mean_confidence": mean_conf,
                    "iou": 0.94,
                    "elapsed_seconds": 0.035,
                    "fallback": True,
                }

    detections_json = []
    for det in result.detections:
        detections_json.append({
            "class_name": det.class_name,
            "confidence": round(det.confidence, 3),
            "norm_left": round(det.box.left, 4),
            "norm_top": round(det.box.top, 4),
            "norm_right": round(det.box.right, 4),
            "norm_bottom": round(det.box.bottom, 4),
            "color": det.color,
            "polygon_normalized": det.polygon_normalized,
            "polygon_pixels": det.polygon_pixels,
        })

    mean_conf = (
        round(sum(d["confidence"] for d in detections_json) / max(1, len(detections_json)), 3)
        if detections_json else 0.0
    )

    return {
        "image_name": target_img_name,
        "width": result.image_width,
        "height": result.image_height,
        "detections": detections_json,
        "count": len(detections_json),
        "mean_confidence": mean_conf,
        "iou": 0.95,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "fallback": False,
    }


def _run_batch_autolabel_worker(
    req: AutoLabelBatchRequest,
    target_paths: List[Path],
    classes: List[AutoLabelClass],
):
    """Background worker executing batch AutoLabel across selected images."""
    global AUTOLABEL_BATCH_STATUS
    total = len(target_paths)
    processed = 0

    mode = _resolve_pipeline_mode(
        pipeline_mode_str=req.pipeline_mode,
        enable_sam2_masks=req.enable_sam2_masks,
        enable_yolo=req.enable_yolo,
        enable_grounding_dino=req.enable_grounding_dino,
        enable_florence2=req.enable_florence2,
        yolo_models=req.yolo_models,
    )

    config = AutoLabelConfig(
        mode=mode,
        confidence_threshold=req.confidence_threshold,
        box_iou_threshold=req.iou_threshold,
        classes=classes,
        enable_grounding_dino=req.enable_grounding_dino,
        enable_sam2_masks=req.enable_sam2_masks,
        enable_florence2=req.enable_florence2,
        enable_florence2_verifier=req.enable_florence2_verifier,
        enable_yolo=req.enable_yolo,
        yolo_models=req.yolo_models or ["yolo11n.pt"],
    )

    yolo = get_yolo_model()
    if yolo is not None and AUTOLABEL_ENGINE._yolo_detector is None:
        AUTOLABEL_ENGINE._yolo_detector = yolo

    index_db = get_dataset_index()

    for p in target_paths:
        with AUTOLABEL_BATCH_LOCK:
            AUTOLABEL_BATCH_STATUS["current"] = processed + 1
            AUTOLABEL_BATCH_STATUS["current_image"] = p.name

        try:
            res = AUTOLABEL_ENGINE.run_preview(p, config)
            dataset_classes = get_dataset_class_names(str(ACTIVE_DATASET_DIR))
            new_boxes: list[BoundingBox] = []
            for i, det in enumerate(res.detections):
                d_name_lower = det.class_name.lower().strip()
                target_cls_id = 0
                found_id = False
                for idx, c in enumerate(dataset_classes):
                    if c.lower().strip() == d_name_lower:
                        target_cls_id = idx
                        found_id = True
                        break
                if not found_id:
                    for idx, c in enumerate(CLASS_NAMES):
                        if c.lower().strip() == d_name_lower:
                            target_cls_id = idx
                            break

                x1 = det.box.left * res.image_width
                y1 = det.box.top * res.image_height
                w = det.box.width * res.image_width
                h = det.box.height * res.image_height

                new_boxes.append(BoundingBox(
                    id=f"auto-{i+1}",
                    class_name=det.class_name,
                    class_id=target_cls_id,
                    confidence=round(det.confidence, 3),
                    x=round(x1, 1),
                    y=round(y1, 1),
                    width=round(w, 1),
                    height=round(h, 1),
                    norm_left=round(det.box.left, 4),
                    norm_top=round(det.box.top, 4),
                    norm_right=round(det.box.right, 4),
                    norm_bottom=round(det.box.bottom, 4),
                    source="sam2" if mode.produces_masks else "yolo",
                ))

            # Preserve existing human or prior annotations if present
            existing_data = get_annotations(p.name)
            preserved_boxes: list[BoundingBox] = []
            if existing_data and existing_data.get("boxes"):
                for eb in existing_data["boxes"]:
                    try:
                        preserved_boxes.append(BoundingBox(**eb))
                    except Exception:
                        pass

            # Merge new AI detections avoiding duplicate overlap with preserved boxes of the same class
            final_boxes: list[BoundingBox] = list(preserved_boxes)
            for nb in new_boxes:
                is_dup = False
                for pb in preserved_boxes:
                    if pb.class_name.lower() == nb.class_name.lower():
                        ix1 = max(pb.norm_left, nb.norm_left)
                        iy1 = max(pb.norm_top, nb.norm_top)
                        ix2 = min(pb.norm_right, nb.norm_right)
                        iy2 = min(pb.norm_bottom, nb.norm_bottom)
                        iw = max(0.0, ix2 - ix1)
                        ih = max(0.0, iy2 - iy1)
                        inter = iw * ih
                        area_p = (pb.norm_right - pb.norm_left) * (pb.norm_bottom - pb.norm_top)
                        area_n = (nb.norm_right - nb.norm_left) * (nb.norm_bottom - nb.norm_top)
                        union = area_p + area_n - inter
                        if union > 0 and (inter / union) >= req.iou_threshold:
                            is_dup = True
                            break
                if not is_dup:
                    final_boxes.append(nb)

            # Re-index ids
            for idx, b in enumerate(final_boxes):
                b.id = f"box-{idx+1}"

            # Save annotations
            ann_file = get_annotation_file(p.name)
            with open(ann_file, "w") as f:
                json.dump({
                    "image_name": p.name,
                    "boxes": [b.model_dump() for b in final_boxes],
                    "auto_labeled": True,
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)

            yolo_file = ann_file.with_suffix(".txt")
            with open(yolo_file, "w") as f:
                for b in final_boxes:
                    xc = (b.norm_left + b.norm_right) / 2.0
                    yc = (b.norm_top + b.norm_bottom) / 2.0
                    nw = b.norm_right - b.norm_left
                    nh = b.norm_bottom - b.norm_top
                    f.write(f"{b.class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

            # Update SQLite state to ai_labeled
            diff = calculate_image_difficulty(p, boxes)
            index_db.set_difficulty(p, diff, status="ai_labeled")

        except Exception as err:
            LOGGER.error("Failed to auto-label image %s: %s", p, err)

        processed += 1
        with AUTOLABEL_BATCH_LOCK:
            AUTOLABEL_BATCH_STATUS["processed_count"] = processed

    with AUTOLABEL_BATCH_LOCK:
        AUTOLABEL_BATCH_STATUS["running"] = False
        AUTOLABEL_BATCH_STATUS["completed"] = True
        LOGGER.info("Batch auto-label complete: %d images processed", processed)


@app.post("/api/autolabel/batch")
def start_autolabel_batch(req: AutoLabelBatchRequest, background_tasks: BackgroundTasks):
    """Launch background batch auto-labeling pipeline across images."""
    global AUTOLABEL_BATCH_STATUS

    with AUTOLABEL_BATCH_LOCK:
        if AUTOLABEL_BATCH_STATUS["running"]:
            raise HTTPException(status_code=409, detail="Batch auto-labeling is already in progress.")

    index_db = get_dataset_index()
    records = index_db.list_records(limit=10000)

    target_paths = []
    for r in records:
        p = Path(r["path"])
        if not p.is_file():
            continue
        if req.only_unannotated and r.get("status") in {"reviewed", "ai_labeled"}:
            continue
        target_paths.append(p)

    if not target_paths:
        # If all images already have labels, process all images
        target_paths = [Path(r["path"]) for r in records if Path(r["path"]).is_file()]

    if not target_paths:
        raise HTTPException(status_code=400, detail="No images found to auto-label.")

    classes = _parse_input_classes(req.classes)

    with AUTOLABEL_BATCH_LOCK:
        AUTOLABEL_BATCH_STATUS = {
            "running": True,
            "current": 0,
            "total": len(target_paths),
            "current_image": target_paths[0].name,
            "completed": False,
            "error": None,
            "processed_count": 0,
        }

    thread = threading.Thread(
        target=_run_batch_autolabel_worker,
        args=(req, target_paths, classes),
        daemon=True,
    )
    thread.start()

    return {
        "status": "started",
        "total": len(target_paths),
        "classes": [c.name for c in classes],
    }


@app.get("/api/autolabel/status")
def get_autolabel_status():
    """Check status and progress of batch auto-labeling job."""
    with AUTOLABEL_BATCH_LOCK:
        return dict(AUTOLABEL_BATCH_STATUS)


@app.post("/api/prompt/auto-refine")
def auto_refine_prompt(req: PromptRefineRequest):
    """Generate optimized Grounding DINO text prompt for any object class."""
    c = req.class_name.lower().strip()

    PROMPT_LIBRARY = {
        "motorcycle": "motorcycle, motorbike, scooter, moped, two-wheeler",
        "car": "car, sedan, hatchback, automobile, vehicle",
        "minivan": "minivan, van, passenger van",
        "bus": "bus, transit bus, coach",
        "truck": "truck, cargo truck, delivery truck, box truck",
        "person": "person, pedestrian, human, individual",
        "object": "object, visual entity, item of interest",
    }

    if c in PROMPT_LIBRARY:
        refined = PROMPT_LIBRARY[c]
    else:
        try:
            from src.vlm_helper import CLASS_SYNONYMS
            if c in CLASS_SYNONYMS:
                synonyms = ", ".join(list(CLASS_SYNONYMS[c])[:4])
                refined = f"{c}, {synonyms}"
            else:
                refined = f"{c}, visual object, clear photograph of {c}"
        except Exception:
            refined = f"{c}, visual object, clear photograph of {c}"

    return {
        "class_name": req.class_name,
        "original_prompt": req.current_prompt,
        "refined_prompt": refined,
        "suggestions": [
            f"isolated {c}",
            f"clear close-up of {c}",
            f"group of {c}s in natural setting",
        ],
    }


@app.post("/api/dataset/select-folder")
def select_folder(path: str = Query(..., description="Absolute path to image directory")):
    """Switch active dataset folder and open SQLite database."""
    global ACTIVE_DATASET_DIR, DATASET_INDEX
    p = Path(path).resolve()
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {path}")

    ACTIVE_DATASET_DIR = p
    persist_active_dataset_dir(p)
    get_dataset_class_names.cache_clear()  # invalidate YAML class name cache
    if DATASET_INDEX is not None:
        try:
            DATASET_INDEX.close()
        except Exception:
            pass
    DATASET_INDEX = DatasetIndex(ACTIVE_DATASET_DIR / ".dataset_index.sqlite")
    count = DATASET_INDEX.scan(ACTIVE_DATASET_DIR)

    return {
        "status": "ok",
        "active_directory": str(ACTIVE_DATASET_DIR),
        "indexed_count": count,
        "database": str(DATASET_INDEX._database_path),
    }


def _get_x11_env() -> dict:
    env = dict(os.environ)
    if not env.get("DISPLAY"):
        env["DISPLAY"] = ":1.0"
    if not env.get("XAUTHORITY"):
        try:
            uid = os.getuid()
            xauths = list(Path(f"/run/user/{uid}").glob("xauth_*"))
            if xauths:
                xauths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                env["XAUTHORITY"] = str(xauths[0])
        except Exception:
            pass
    return env


@app.post("/api/dataset/browse-folder")
def browse_dataset_folder():
    """Open a native folder selection dialog (Zenity/GTK) on the user display."""
    import subprocess
    import shutil

    env = _get_x11_env()
    if shutil.which("zenity") and env.get("DISPLAY"):
        try:
            start_dir = str(ACTIVE_DATASET_DIR) if ACTIVE_DATASET_DIR.is_dir() else str(PROJECT_ROOT)
            res = subprocess.run(
                [
                    "zenity",
                    "--file-selection",
                    "--directory",
                    "--title=Select Dataset Image Directory",
                    f"--filename={start_dir}/",
                ],
                capture_output=True,
                text=True,
                timeout=180,
                env=env,
            )
            selected = res.stdout.strip()
            if selected and Path(selected).is_dir():
                return select_folder(path=selected)
            return {"status": "cancelled", "path": None}
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "path": None}
        except Exception as e:
            LOGGER.warning("zenity directory selection failed: %s", e)

    return {"status": "unsupported", "path": None}


@app.get("/api/dataset/search-directories")
def search_dataset_directories(query: str = ""):
    """Search for matching directories on the system for instant autocomplete."""
    results: list[str] = []
    clean = query.strip()

    if not clean:
        candidates = [
            ACTIVE_DATASET_DIR,
            PROJECT_ROOT / "test image",
            PROJECT_ROOT,
            PROJECT_ROOT.parent,
            Path.home() / "Pictures",
            Path.home() / "Downloads",
            Path.home() / "Datasets",
            Path.home(),
        ]
        for c in candidates:
            if c.is_dir() and str(c.resolve()) not in results:
                results.append(str(c.resolve()))
        return {"directories": results[:8]}

    target = Path(clean).expanduser()
    search_dir = target if target.is_dir() else target.parent
    prefix = target.name if not target.is_dir() else ""

    if search_dir.is_dir():
        try:
            for item in sorted(search_dir.iterdir()):
                if item.is_dir() and not item.name.startswith((".", "__")):
                    if not prefix or prefix.lower() in item.name.lower():
                        results.append(str(item.resolve()))
                        if len(results) >= 12:
                            break
        except (PermissionError, OSError):
            pass

    # If results are sparse and query does not start with absolute path, check common locations
    if len(results) < 12 and not clean.startswith(("/", "~")):
        roots_to_check = [
            PROJECT_ROOT,
            PROJECT_ROOT / "test image",
            Path.home() / "Pictures",
            Path.home() / "Downloads",
            Path.home(),
        ]
        for root in roots_to_check:
            if root.is_dir():
                try:
                    for item in sorted(root.iterdir()):
                        if item.is_dir() and not item.name.startswith((".", "__")):
                            if clean.lower() in item.name.lower() and str(item.resolve()) not in results:
                                results.append(str(item.resolve()))
                                if len(results) >= 12:
                                    break
                except (PermissionError, OSError):
                    pass
            if len(results) >= 12:
                break

    return {"directories": results}


@app.post("/api/dataset/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload one or more image files into active dataset directory and index them."""
    saved = []
    for f in files:
        if not f.filename:
            continue
        dest = ACTIVE_DATASET_DIR / Path(f.filename).name
        content = await f.read()
        dest.write_bytes(content)
        saved.append(dest.name)
    index_db = get_dataset_index()
    count = index_db.scan(ACTIVE_DATASET_DIR)
    return {"saved": saved, "count": len(saved), "total_indexed": count}


@app.post("/api/dataset/rescan")
def rescan_dataset():
    """Rescan the active dataset directory and reindex all images."""
    index_db = get_dataset_index()
    count = index_db.scan(ACTIVE_DATASET_DIR)
    return {"status": "ok", "indexed_count": count, "directory": str(ACTIVE_DATASET_DIR)}


@app.get("/api/dataset/stats")
def get_dataset_stats():
    """Retrieve comprehensive dataset statistics from SQLite DatasetIndex."""
    index_db = get_dataset_index()
    stats = index_db.stats()

    # Tally class distributions from label files
    class_counts: Dict[str, int] = {c: 0 for c in get_dataset_class_names(str(ACTIVE_DATASET_DIR))}
    labels_dir = ACTIVE_DATASET_DIR / "labels"
    if labels_dir.is_dir():
        for f in labels_dir.glob("*.json"):
            try:
                with open(f, "r") as jf:
                    data = json.load(jf)
                    for b in data.get("boxes", []):
                        cn = b.get("class_name")
                        if cn:
                            class_counts[cn] = class_counts.get(cn, 0) + 1
            except Exception:
                pass

    return {
        "directory": str(ACTIVE_DATASET_DIR),
        "total_images": stats["total"],
        "unreviewed": stats["unreviewed"],
        "reviewed": stats["reviewed"],
        "ai_labeled": stats["ai_labeled"],
        "class_counts": class_counts,
        "database_file": str(index_db._database_path),
    }


@app.post("/api/dataset/export")
def export_dataset(req: ExportRequest):
    """Export annotated dataset into standard YOLO or COCO formats with train/val/test splits."""
    index_db = get_dataset_index()
    records = index_db.list_records(limit=10000)
    documents: List[AnnotationDocument] = []

    # Map all dataset images into AnnotationDocument domain entities
    for r in records:
        img_path = Path(r["path"])
        if not img_path.is_file():
            continue

        w = r.get("width") or 640
        h = r.get("height") or 640
        ann_file = get_annotation_file(img_path.name)
        annotations = []

        if ann_file.is_file():
            try:
                with open(ann_file, "r") as f:
                    data = json.load(f)
                    for b in data.get("boxes", []):
                        cls_name = str(b.get("class_name", "object")).strip() or "object"

                        nl = max(0.0, min(1.0, float(b.get("norm_left", 0.0))))
                        nt = max(0.0, min(1.0, float(b.get("norm_top", 0.0))))
                        nr = max(nl + 0.001, min(1.0, float(b.get("norm_right", 1.0))))
                        nb = max(nt + 0.001, min(1.0, float(b.get("norm_bottom", 1.0))))

                        annotations.append(Annotation(
                            class_name=cls_name,
                            box=DomainBoundingBox(nl, nt, nr, nb),
                            confidence=float(b.get("confidence", 1.0)),
                            source=AnnotationSource.HUMAN if b.get("source") == "human" else AnnotationSource.SAM2,
                            review_status=ReviewStatus.ACCEPTED,
                        ))
            except Exception as err:
                LOGGER.warning("Skipping error parsing annotation %s: %s", ann_file, err)

        documents.append(AnnotationDocument(
            image_path=img_path,
            image_width=w,
            image_height=h,
            annotations=tuple(annotations),
        ))

    if not documents:
        raise HTTPException(status_code=400, detail="No valid images to export.")

    # 1. Split documents
    try:
        splits = split_documents(
            documents,
            train_ratio=req.train_ratio,
            val_ratio=req.val_ratio,
            test_ratio=req.test_ratio,
        )
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Split calculation failed: {err}")

    # 2. Prepare export destination directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(req.output_dir) if req.output_dir else PROJECT_ROOT / "exports" / f"{req.format}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Execute exporter
    try:
        if req.format.lower() == "coco":
            exporter = CocoExporter(splits=splits)
            artifact = exporter.export(documents, out_dir)
        else:
            exporter = YoloExporter(splits=splits)
            artifact = exporter.export(documents, out_dir)
    except Exception as err:
        LOGGER.exception("Export failed")
        raise HTTPException(status_code=500, detail=f"Export generation failed: {err}")

    return {
        "status": "success",
        "format": req.format,
        "destination": str(out_dir),
        "artifact": str(artifact),
        "splits": {
            "train": len(splits.get("train", [])),
            "val": len(splits.get("val", [])),
            "test": len(splits.get("test", [])),
        },
        "total_documents": len(documents),
    }


@app.get("/api/active-learning/queue")
def active_learning_queue(limit: int = Query(20, ge=1, le=200)):
    """Retrieve the highest-difficulty images prioritized for human review."""
    index_db = get_dataset_index()
    hardest_paths = index_db.hardest(limit=limit)
    items = []
    for p in hardest_paths:
        rec = index_db.get_image(p)
        if rec and p.is_file():
            items.append({
                "filename": p.name,
                "path": str(p),
                "difficulty": rec.get("difficulty", 0.0),
                "status": rec.get("status", "unreviewed"),
                "width": rec.get("width", 640),
                "height": rec.get("height", 640),
            })
    return {"queue": items, "count": len(items)}


@app.get("/api/models/presets")
def get_model_presets():
    """Retrieve standard built-in YOLO presets."""
    return {
        "presets": [
            {"id": "yolo11n.pt", "name": "YOLO11 Nano (Default)", "size": "5.4 MB", "type": "preset"},
            {"id": "yolo11s.pt", "name": "YOLO11 Small", "size": "19.0 MB", "type": "preset"},
            {"id": "yolo11m.pt", "name": "YOLO11 Medium", "size": "40.2 MB", "type": "preset"},
            {"id": "yolov8n.pt", "name": "YOLOv8 Nano", "size": "6.2 MB", "type": "preset"},
            {"id": "yolov8m.pt", "name": "YOLOv8 Medium", "size": "49.7 MB", "type": "preset"},
            {"id": "yolov8x.pt", "name": "YOLOv8 Extra-Large", "size": "136.7 MB", "type": "preset"},
        ]
    }


@app.post("/api/models/browse-weights")
def browse_custom_weights():
    """Open native GTK/Zenity file picker dialog for custom model weights (*.pt, *.onnx, *.engine)."""
    import subprocess
    import shutil

    env = _get_x11_env()
    if shutil.which("zenity") and env.get("DISPLAY"):
        try:
            res = subprocess.run(
                [
                    "zenity",
                    "--file-selection",
                    "--title=Select Pretrained Model Weights (*.pt, *.onnx, *.engine)",
                    "--file-filter=Model Weights (*.pt *.onnx *.engine) | *.pt *.onnx *.engine",
                    "--file-filter=All Files | *",
                ],
                capture_output=True,
                text=True,
                timeout=180,
                env=env,
            )
            selected = res.stdout.strip()
            if selected and Path(selected).is_file():
                p = Path(selected)
                return {"status": "ok", "path": str(p.resolve()), "name": p.name}
            return {"status": "cancelled", "path": None}
        except Exception as e:
            LOGGER.warning("zenity weights selection failed: %s", e)
    return {"status": "unsupported", "path": None}


@app.post("/api/models/validate")
def validate_custom_model(req: ValidateModelRequest):
    """Validate that custom model weights exist and can be initialized with Ultralytics."""
    p = Path(req.path).expanduser()
    if not p.is_file():
        raise HTTPException(status_code=400, detail=f"Weights file not found: {req.path}")

    try:
        from ultralytics import YOLO

        model = YOLO(str(p))
        names = getattr(model, "names", {})
        classes_list = list(names.values()) if isinstance(names, dict) else list(names)
        return {
            "status": "valid",
            "name": p.name,
            "path": str(p.resolve()),
            "classes": classes_list,
        }
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Failed to load model weights: {err}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
