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
import threading
import time
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ConfigDict, model_validator
from PIL import Image

from app.services.dataset.index import DatasetIndex, IMAGE_SUFFIXES
from app.services.dataset.repository import AnnotationRepository, RepositoryError, RevisionConflict
from app.services.active_learning import (
    ActiveLearningEngine,
    ImageAnalysis,
)
from app.services.annotation.domain import (
    AnnotationDocument,
    AnnotationSource,
    BoundingBox as DomainBoundingBox,
)
from app.services.auto_label.engine import AutoLabelEngine
from app.services.auto_label.models import (
    AutoLabelClass,
    AutoLabelConfig,
    AutoLabelPipelineMode,
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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:1420", "http://127.0.0.1:1420", "http://tauri.localhost", "https://tauri.localhost", "tauri://localhost"],
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
DATASET_STATE_LOCK = threading.RLock()
INFERENCE_LOCK = threading.RLock()
SESSION_TOKEN = secrets.token_urlsafe(32)
ALLOWED_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:1420", "http://127.0.0.1:1420", "http://tauri.localhost", "https://tauri.localhost", "tauri://localhost"}


@app.middleware("http")
async def authorize_local_client(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        return JSONResponse({"detail": "Client origin is not allowed"}, status_code=403)
    if request.method != "OPTIONS" and request.url.path != "/api/session":
        token = request.headers.get("X-VisionLab-Token", "")
        if request.url.path.startswith("/api/image/"):
            token = token or request.query_params.get("token", "")
        if not secrets.compare_digest(token, SESSION_TOKEN):
            return JSONResponse({"detail": "A local session token is required"}, status_code=401)
    return await call_next(request)


@app.get("/api/session")
def create_session():
    return {"token": SESSION_TOKEN}


def dataset_operation(function):
    """Keep selection and a synchronous operation on the same dataset."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with DATASET_STATE_LOCK:
            return function(*args, **kwargs)
    return wrapped


def inference_operation(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with INFERENCE_LOCK:
            return function(*args, **kwargs)
    return wrapped

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
    repo = AnnotationRepository(ACTIVE_DATASET_DIR)
    data = repo.read(img_path)
    index_db.set_metadata(img_path, data["width"], data["height"])
    record = index_db.get_image(img_path) or {}
    index_db.set_status_and_count(img_path, data["status"], len(data["boxes"]), record.get("difficulty") or 0)
    return data["width"], data["height"], data["status"]

# ------------------------------------------------------------------------------
# Request & Response Models
# ------------------------------------------------------------------------------

class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    id: str
    class_name: str
    class_id: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
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
    polygon_normalized: Optional[List[List[float]]] = None
    polygon_pixels: Optional[List[List[float]]] = None

    @model_validator(mode="after")
    def validate_geometry(self):
        try:
            DomainBoundingBox(self.norm_left, self.norm_top, self.norm_right, self.norm_bottom)
        except Exception as err:
            raise ValueError(str(err)) from err
        if self.source == "manual":
            self.source = "human"
        if self.source not in {str(s) for s in AnnotationSource}:
            raise ValueError("unknown annotation source")
        return self


class DetectionRequest(BaseModel):
    image_name: str
    dataset_id: Optional[str] = None
    conf_threshold: float = 0.25
    classes: Optional[List[str]] = None
    models: Optional[List[str]] = None


class PromptRefineRequest(BaseModel):
    class_name: str
    current_prompt: Optional[str] = None


class SaveAnnotationRequest(BaseModel):
    image_name: str
    boxes: List[BoundingBox]
    expected_revision: Optional[int] = Field(default=None, ge=0)
    dataset_id: Optional[str] = None


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
    dataset_id: Optional[str] = None
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
    dataset_id: Optional[str] = None
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


def get_dataset_class_names(dataset_dir: str | None = None) -> list[str]:
    return AnnotationRepository(Path(dataset_dir) if dataset_dir else ACTIVE_DATASET_DIR).classes()

def _prewarm_yolo_model() -> None:
    """Pre-warm the default YOLO model at startup in a background thread."""
    try:
        LOGGER.info("Pre-warming YOLO model in background…")
        custom_path = Path("/home/marsel/Work/Models/Final.pt")
        target_name = str(custom_path) if custom_path.is_file() else "yolo11n.pt"
        with INFERENCE_LOCK:
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
    return AnnotationRepository(ACTIVE_DATASET_DIR).resolve(img_path)


def get_image_path(image_name: str) -> Path:
    try:
        return AnnotationRepository(ACTIVE_DATASET_DIR).image_path(image_name)
    except FileNotFoundError as err:
        raise HTTPException(404, "Image not found") from err
    except RepositoryError as err:
        raise HTTPException(422, str(err)) from err


def get_annotation_file(image_name: str, img_path: Optional[Path] = None) -> Path:
    repo = AnnotationRepository(ACTIVE_DATASET_DIR)
    return repo.canonical_path(img_path or get_image_path(image_name))


def check_dataset(dataset_id: str | None) -> None:
    if dataset_id is not None and dataset_id != str(ACTIVE_DATASET_DIR.resolve()):
        raise HTTPException(409, "Dataset changed; reload before continuing")


def all_records(index_db: DatasetIndex):
    offset = 0
    while True:
        page = index_db.list_records(limit=1000, offset=offset)
        if not page:
            return
        yield from page
        offset += len(page)

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
@dataset_operation
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
            "yolo11": YOLO_MODEL is not None or bool(AUTOLABEL_ENGINE._yolo_detectors) or AUTOLABEL_ENGINE._yolo_detector is not None,
            "sam2": AUTOLABEL_ENGINE._sam_segmenter is not None,
            "grounding_dino": AUTOLABEL_ENGINE._grounding_detector is not None,
            "florence2": AUTOLABEL_ENGINE._vlm_helper is not None,
        },
    }


@app.get("/api/images")
@dataset_operation
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
    repo = AnnotationRepository(ACTIVE_DATASET_DIR)

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

        cached_count = r.get("annotation_count", -1)
        if cached_count == -1:
            try:
                data = repo.read(img_path)
                ann_count = len(data["boxes"])
                current_status = data["status"]
                index_db.set_status_and_count(img_path, current_status, ann_count, r.get("difficulty") or 0)
            except Exception as err:
                raise HTTPException(422, f"Cannot load labels for {img_path.name}: {err}") from err
        else:
            ann_count = cached_count

        images.append({
            "filename": img_path.name,
            "image_id": str(img_path.relative_to(ACTIVE_DATASET_DIR.resolve())),
            "dataset_id": str(ACTIVE_DATASET_DIR.resolve()),
            "path": str(img_path),
            "width": w,
            "height": h,
            "size_bytes": 0,  # avoid per-image stat() call; not used in UI
            "annotation_count": ann_count,
            "difficulty": r.get("difficulty", 0.0),
            "status": current_status,
        })

    return {
        "images": images,
        "total": index_db.count(),
        "directory": str(ACTIVE_DATASET_DIR),
        "db_stats": index_db.stats(),
    }



@app.get("/api/image/{image_name:path}")
@dataset_operation
def serve_image(image_name: str):
    """Serve image file content."""
    img_path = get_image_path(image_name)
    media_type = "image/jpeg" if img_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return FileResponse(img_path, media_type=media_type)


@app.get("/api/annotations/{image_name:path}")
@dataset_operation
def get_annotations(image_name: str, dataset_id: Optional[str] = None):
    check_dataset(dataset_id or None)
    image = get_image_path(image_name)
    try:
        data = AnnotationRepository(ACTIVE_DATASET_DIR).read(image)
        index = get_dataset_index()
        index.set_metadata(image, data["width"], data["height"])
        record = index.get_image(image) or {}
        index.set_status_and_count(image, data["status"], len(data["boxes"]), record.get("difficulty") or 0)
        return data
    except Exception as err:
        raise HTTPException(422, f"Cannot load annotations: {err}") from err


@app.post("/api/annotations/{image_name:path}")
@dataset_operation
def save_annotations(image_name: str, payload: SaveAnnotationRequest):
    check_dataset(payload.dataset_id)
    if image_name != payload.image_name:
        raise HTTPException(422, "Path and payload image identities differ")
    image = get_image_path(image_name)
    try:
        repo = AnnotationRepository(ACTIVE_DATASET_DIR)
        data = repo.save(image, [b.model_dump() for b in payload.boxes],
                         expected_revision=payload.expected_revision)
    except RevisionConflict as err:
        raise HTTPException(409, str(err)) from err
    except Exception as err:
        raise HTTPException(422, f"Cannot save annotations: {err}") from err
    index = get_dataset_index()
    difficulty = calculate_image_difficulty(image, payload.boxes)
    index.set_status_and_count(image, "reviewed", len(data["boxes"]), difficulty)
    return {"status": "saved", "image_name": image_name, "count": len(data["boxes"]),
            "box_count": len(data["boxes"]), "difficulty": difficulty,
            "review_status": "reviewed", "revision": data["revision"], "boxes": data["boxes"],
            "file": str(repo.canonical_path(image)), "projection_warning": data.get("projection_warning")}

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
@dataset_operation
@inference_operation
def detect_yolo(req: DetectionRequest):
    """Run YOLO object detection on an image (supports multi-model ensemble)."""
    check_dataset(req.dataset_id)
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
                    class_name = "car"
                elif raw_lower in {"motorcycle", "motor", "bike", "motorbike", "scooter", "moped"}:
                    class_name = "motorcycle"
                elif raw_lower in {"bus", "bis", "minibus", "angkot"}:
                    class_name = "bus"
                elif raw_lower in {"truck", "truk", "pickup"}:
                    class_name = "truck"
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

        dataset_classes = AnnotationRepository(ACTIVE_DATASET_DIR).register_classes([c.name for c in fused_cls])
        for i, (b_px, c_obj, sc) in enumerate(zip(fused_px, fused_cls, fused_scores)):
            x1, y1, x2, y2 = b_px
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            norm_left = max(0.0, min(1.0, x1 / img_w))
            norm_top = max(0.0, min(1.0, y1 / img_h))
            norm_right = max(0.0, min(1.0, x2 / img_w))
            norm_bottom = max(0.0, min(1.0, y2 / img_h))

            canonical_name = next(c for c in dataset_classes if c.casefold() == c_obj.name.casefold())
            target_cls_id = dataset_classes.index(canonical_name)
            if norm_left >= norm_right or norm_top >= norm_bottom:
                continue

            boxes.append(BoundingBox(
                id=f"yolo-{i+1}",
                class_name=canonical_name,
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
@dataset_operation
@inference_operation
def autolabel_preview(req: AutoLabelPreviewRequest):
    """Run Auto Label on a single image and return real preview detections."""
    check_dataset(req.dataset_id)
    started_at = time.perf_counter()
    target_img_name = req.image_name
    if not target_img_name:
        # Pick the first available image in active dataset
        index_db = get_dataset_index()
        records = index_db.list_records(limit=1)
        if records:
            target_img_name = records[0]["path"]
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
            "iou": None,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
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
                    "iou": None,
                    "elapsed_seconds": round(time.perf_counter() - started_at, 3),
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
        "iou": None,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "fallback": False,
    }


def _execute_batch_autolabel_worker(
    req: AutoLabelBatchRequest,
    target_paths: List[Path],
    classes: List[AutoLabelClass],
    dataset_root: Optional[Path] = None,
    job_status: Optional[dict] = None,
):
    """Background worker executing batch AutoLabel across selected images."""
    if job_status is None:
        job_status = AUTOLABEL_BATCH_STATUS
    root = (dataset_root or ACTIVE_DATASET_DIR).resolve()
    repo = AnnotationRepository(root)
    processed = 0
    failures = []
    warnings = []
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

    index_db = DatasetIndex(root / ".dataset_index.sqlite")
    for p in target_paths:
        with AUTOLABEL_BATCH_LOCK:
            job_status["current"] = processed + 1
            job_status["current_image"] = p.name

        try:
            existing_data = repo.read(p)
            with INFERENCE_LOCK:
                res = AUTOLABEL_ENGINE.run_preview(p, config)
            dataset_classes = repo.register_classes([d.class_name for d in res.detections])
            new_boxes: list[BoundingBox] = []
            for i, det in enumerate(res.detections):
                canonical_name = next(c for c in dataset_classes if c.casefold() == det.class_name.casefold())
                target_cls_id = dataset_classes.index(canonical_name)

                x1 = det.box.left * res.image_width
                y1 = det.box.top * res.image_height
                w = det.box.width * res.image_width
                h = det.box.height * res.image_height

                new_boxes.append(BoundingBox(
                    id=secrets.token_hex(16),
                    class_name=canonical_name,
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
                    polygon_normalized=det.polygon_normalized,
                    polygon_pixels=det.polygon_pixels,
                ))

            # Preserve existing human or prior annotations if present
            preserved_boxes: list[BoundingBox] = []
            if existing_data and existing_data.get("boxes"):
                for eb in existing_data["boxes"]:
                    try:
                        preserved_boxes.append(BoundingBox(**eb))
                    except Exception as err:
                        raise RepositoryError("Cannot preserve an existing annotation") from err

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

            saved = repo.save(p, [b.model_dump() for b in final_boxes], status="ai_labeled",
                              expected_revision=existing_data["revision"])
            if saved.get("projection_warning"):
                warnings.append(saved["projection_warning"])
            diff = calculate_image_difficulty(p, final_boxes)
            index_db.set_status_and_count(p, "ai_labeled", len(final_boxes), diff)
        except Exception as err:
            LOGGER.error("Failed to auto-label image %s: %s", p, err)
            failures.append({"image": str(p.relative_to(root)), "error": str(err)})

        processed += 1
        with AUTOLABEL_BATCH_LOCK:
            job_status["processed_count"] = processed

    index_db.close()
    with AUTOLABEL_BATCH_LOCK:
        job_status["failures"] = failures
        job_status["warnings"] = warnings
        job_status["succeeded_count"] = processed - len(failures)
        job_status["failed_count"] = len(failures)
        job_status["error"] = f"{len(failures)} images failed" if failures else None
        job_status["running"] = False
        job_status["completed"] = True
        LOGGER.info("Batch auto-label complete: %d images processed", processed)


def _run_batch_autolabel_worker(req, target_paths, classes, dataset_root=None, job_status=None):
    if job_status is None:
        job_status = AUTOLABEL_BATCH_STATUS
    root = (dataset_root or ACTIVE_DATASET_DIR).resolve()
    try:
        _execute_batch_autolabel_worker(req, target_paths, classes, root, job_status)
    except Exception as err:
        LOGGER.exception("Batch setup failed")
        with AUTOLABEL_BATCH_LOCK:
            job_status.update(running=False, completed=True, error=str(err))
    finally:
        with AUTOLABEL_BATCH_LOCK:
            snapshot = dict(job_status)
        from app.services.dataset.repository import atomic_text
        job_id = snapshot.get("job_id", "last")
        try:
            atomic_text(root / ".visionlab" / "jobs" / (job_id + ".json"), json.dumps(snapshot, indent=2))
        except OSError as err:
            LOGGER.error("Cannot persist batch outcome: %s", err)
            with AUTOLABEL_BATCH_LOCK:
                job_status["error"] = f"{job_status.get('error') or ''} Cannot persist batch outcome: {err}".strip()


@app.post("/api/autolabel/batch")
@dataset_operation
def start_autolabel_batch(req: AutoLabelBatchRequest, background_tasks: BackgroundTasks):
    global AUTOLABEL_BATCH_STATUS
    check_dataset(req.dataset_id)
    with AUTOLABEL_BATCH_LOCK:
        if AUTOLABEL_BATCH_STATUS["running"]:
            raise HTTPException(409, "Batch auto-labeling is already in progress")
        AUTOLABEL_BATCH_STATUS = {"running": True, "current": 0, "total": 0,
            "current_image": "", "completed": False, "error": None, "processed_count": 0,
            "job_id": secrets.token_hex(12), "dataset_id": str(ACTIVE_DATASET_DIR.resolve())}
        job_status = AUTOLABEL_BATCH_STATUS
    try:
        root = ACTIVE_DATASET_DIR.resolve()
        repo = AnnotationRepository(root)
        classes = _parse_input_classes(req.classes)
        repo.register_classes([c.name for c in classes])
        records = list(all_records(get_dataset_index()))
        target_paths = []
        for record in records:
            path = Path(record["path"])
            if not path.is_file():
                continue
            if req.only_unannotated and (record.get("status") in {"reviewed", "ai_labeled"}
                                       or repo.read(path)["status"] in {"reviewed", "ai_labeled"}):
                continue
            target_paths.append(path)
        with AUTOLABEL_BATCH_LOCK:
            AUTOLABEL_BATCH_STATUS["total"] = len(target_paths)
            AUTOLABEL_BATCH_STATUS["current_image"] = target_paths[0].name if target_paths else ""
            if not target_paths:
                AUTOLABEL_BATCH_STATUS.update(running=False, completed=True)
        if target_paths:
            thread = threading.Thread(target=_run_batch_autolabel_worker,
                args=(req, target_paths, classes, root, job_status), daemon=True)
            thread.start()
        return {"status": "started" if target_paths else "completed", "total": len(target_paths),
                "classes": [c.name for c in classes]}
    except Exception:
        with AUTOLABEL_BATCH_LOCK:
            AUTOLABEL_BATCH_STATUS["running"] = False
        raise

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
@dataset_operation
def select_folder(path: str = Query(..., description="Absolute path to image directory")):
    """Switch active dataset folder and open SQLite database."""
    global ACTIVE_DATASET_DIR, DATASET_INDEX
    p = Path(path).resolve()
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {path}")

    ACTIVE_DATASET_DIR = p
    persist_active_dataset_dir(p)
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
@dataset_operation
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
@dataset_operation
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
    root = ACTIVE_DATASET_DIR.resolve()
    staged = []
    names = set()
    for upload in files:
        name = Path(upload.filename or "").name
        if not name or Path(name).suffix.lower() not in IMAGE_SUFFIXES:
            raise HTTPException(422, "Upload must be a supported image")
        if name.casefold() in names or (root / name).exists():
            raise HTTPException(409, f"Image already exists: {name}")
        names.add(name.casefold())
        content = await upload.read()
        try:
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
        except Exception as err:
            raise HTTPException(422, f"Invalid image: {name}") from err
        staged.append((name, content))
    saved = []
    try:
        for name, content in staged:
            # Exclusive creation prevents a collision between simultaneous uploads.
            with (root / name).open("xb") as stream:
                saved.append(name)
                stream.write(content)
    except Exception as err:
        for name in saved:
            (root / name).unlink(missing_ok=True)
        raise HTTPException(409, "Upload could not be committed") from err
    with DatasetIndex(root / ".dataset_index.sqlite") as index:
        count = index.scan(root)
    return {"saved": saved, "count": len(saved), "total_indexed": count}


@app.post("/api/dataset/classes")
@dataset_operation
def register_class(payload: Dict[str, str]):
    check_dataset(payload.get("dataset_id"))
    name = payload.get("name", "").strip()
    if not name:
        raise HTTPException(422, "Class name is required")
    return {"classes": AnnotationRepository(ACTIVE_DATASET_DIR).register_classes([name])}


@app.post("/api/dataset/rescan")
@dataset_operation
def rescan_dataset():
    """Rescan the active dataset directory and reindex all images."""
    index_db = get_dataset_index()
    count = index_db.scan(ACTIVE_DATASET_DIR)
    return {"status": "ok", "indexed_count": count, "directory": str(ACTIVE_DATASET_DIR)}


@app.get("/api/dataset/stats")
@dataset_operation
def get_dataset_stats():
    """Retrieve comprehensive dataset statistics from SQLite DatasetIndex."""
    index_db = get_dataset_index()
    stats = index_db.stats()

    class_counts: Dict[str, int] = {c: 0 for c in get_dataset_class_names(str(ACTIVE_DATASET_DIR))}
    repo = AnnotationRepository(ACTIVE_DATASET_DIR)
    for record in all_records(index_db):
        image = Path(record["path"])
        if image.is_file():
            data = repo.read(image)
            index_db.set_status_and_count(image, data["status"], len(data["boxes"]), record.get("difficulty") or 0)
            for box in data["boxes"]:
                name = box["class_name"]
                class_counts[name] = class_counts.get(name, 0) + 1
    stats = index_db.stats()

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
@dataset_operation
def export_dataset(req: ExportRequest):
    """Export annotated dataset into standard YOLO or COCO formats with train/val/test splits."""
    index_db = get_dataset_index()
    records = all_records(index_db)
    documents: List[AnnotationDocument] = []
    repo = AnnotationRepository(ACTIVE_DATASET_DIR)
    for record in records:
        image = Path(record["path"])
        if not image.is_file():
            raise HTTPException(422, "An indexed source image is missing; rescan before exporting")
        try:
            documents.append(repo.document(image))
        except Exception as err:
            raise HTTPException(422, f"Cannot export {image.name}: {err}") from err

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
            exporter = CocoExporter(splits=splits, class_order=repo.classes())
            artifact = exporter.export(documents, out_dir)
        else:
            exporter = YoloExporter(splits=splits, class_order=repo.classes())
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
@dataset_operation
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
@dataset_operation
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
@dataset_operation
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
@dataset_operation
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
