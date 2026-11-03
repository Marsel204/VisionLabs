"""FastAPI backend for the Anotara annotation studio.

Provides full-stack REST endpoints connecting the Tauri/React desktop UI
with the PyTorch/YOLO/SAM-2/Grounding DINO computer vision pipeline,
the SQLite DatasetIndex database, Active Learning engine, and dataset exporters.
"""

from __future__ import annotations

import io
import errno
import json
import logging
import os
import functools
import threading
import tempfile
import time
import secrets
import sqlite3
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, ConfigDict, model_validator
from PIL import Image
from anyio import CapacityLimiter, from_thread, to_thread

from app.services.dataset.index import DatasetIndex, IMAGE_SUFFIXES
from app.services.dataset.repository import AnnotationRepository, RepositoryError, RevisionConflict
from app.services.dataset.class_management import change_class
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
from app.configs.settings import PathSettings
from app.api.colab_routes import ColabJobRequest, create_router as create_colab_router
from app.services.colab_connection import ColabConnection, MAX_TRANSFER_BYTES
from app.services.auto_label.models import (
    AutoLabelClass,
    AutoLabelConfig,
    AutoLabelPipelineMode,
    DEFAULT_AUTO_LABEL_CLASSES,
)
from app.services.inference.cloud import CloudInferenceError, MAX_IMAGE_BYTES
from app.services.inference.preparation import ModelPreparation
from app.export.exporters import (
    CocoExporter,
    CvatExporter,
    PascalVocExporter,
    YoloExporter,
    split_documents,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("api_server")

app = FastAPI(
    title="Anotara AI Annotation API",
    version="2.0.0",
    description="Backend AI API for the Anotara desktop annotation app with SQLite database & active learning",
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


def dataset_state_paths(project_root: Path) -> tuple[Path, Path]:
    if getattr(sys, "frozen", False):
        paths = PathSettings()
        return paths.dataset_root, paths.cache_root / "active_dataset_config.json"
    return project_root / "test image", project_root / ".active_dataset_config.json"


DEFAULT_DATASET_DIR, DATASET_CONFIG_FILE = dataset_state_paths(PROJECT_ROOT)


def get_initial_dataset_dir() -> Path | None:
    """Load persisted active dataset directory if available, else fallback."""
    if DATASET_CONFIG_FILE.is_file():
        try:
            with open(DATASET_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("active_directory", "") is None:
                    return None
                d = Path(data.get("active_directory", "")).resolve()
                if d.is_dir():
                    LOGGER.info("Restored persisted active dataset directory: %s", d)
                    return d
        except Exception as e:
            LOGGER.warning("Failed to read active dataset config: %s", e)
    return DEFAULT_DATASET_DIR if DEFAULT_DATASET_DIR.is_dir() else PROJECT_ROOT


def persist_active_dataset_dir(p: Path | None) -> bool:
    """Persist active dataset directory to disk so it survives server reloads."""
    try:
        with open(DATASET_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"active_directory": str(p.resolve()) if p else None}, f, indent=2)
        LOGGER.info("Persisted active dataset directory: %s", p)
        return True
    except Exception as e:
        LOGGER.warning("Failed to save active dataset config: %s", e)
        return False


# Global state
ACTIVE_DATASET_DIR = get_initial_dataset_dir()
YOLO_MODEL = None
CLASS_NAMES = ["motorcycle", "car", "bus", "truck", "minivan", "person"]

# Database & Engine instances
DATASET_INDEX: Optional[DatasetIndex] = None
ACTIVE_LEARNING_ENGINE = ActiveLearningEngine()
AUTOLABEL_ENGINE = AutoLabelEngine()
COLAB_CONNECTION = ColabConnection(PathSettings().cache_root / "colab")
DATASET_STATE_LOCK = threading.RLock()
INFERENCE_LOCK = threading.RLock()
MODEL_PREPARATION = ModelPreparation(AUTOLABEL_ENGINE, INFERENCE_LOCK)
SESSION_TOKEN = secrets.token_urlsafe(32)
ALLOWED_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:1420", "http://127.0.0.1:1420", "http://tauri.localhost", "https://tauri.localhost", "tauri://localhost"}


@app.exception_handler(CloudInferenceError)
async def cloud_inference_error(request: Request, error: CloudInferenceError):
    return JSONResponse({"detail": str(error)}, status_code=503)


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
    check_dataset(None)
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
    with repo.lock, repo._write_lock():
        generation = index_db.annotation_generation()
        data = repo.read(img_path)
        index_db.set_metadata(img_path, data["width"], data["height"])
        record = index_db.get_image(img_path) or {}
        index_db.set_status_and_count(
            img_path, data["status"], len(data["boxes"]), record.get("difficulty") or 0,
            boxes=data["boxes"], expected_generation=generation,
        )
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


def _validated_views(views: Optional[List[str]]) -> tuple[str, ...]:
    """Validate requested detector views; extra views run only on the local backend."""
    from app.services.preprocess import ORIGINAL_ONLY, validate_views

    if not views:
        return ORIGINAL_ONLY
    try:
        result = validate_views(views)
    except ValueError as err:
        raise HTTPException(422, str(err)) from err
    if result != ORIGINAL_ONLY and AUTOLABEL_ENGINE.uses_cloud:
        raise HTTPException(400, "Layered detection passes run only on the local backend")
    return result


class DetectionRequest(BaseModel):
    image_name: str
    dataset_id: Optional[str] = None
    conf_threshold: float = 0.25
    classes: Optional[List[str]] = None
    models: Optional[List[str]] = None
    views: Optional[List[str]] = None


class PromptRefineRequest(BaseModel):
    class_name: str
    current_prompt: Optional[str] = None


ReviewState = Literal["reviewed", "unreviewed", "ai_labeled"]


class SaveAnnotationRequest(BaseModel):
    image_name: str
    boxes: List[BoundingBox]
    expected_revision: Optional[int] = Field(default=None, ge=0)
    dataset_id: Optional[str] = None
    # Editing and approving are different acts. Clients that omit this keep the historical
    # behaviour of a save counting as a review.
    status: Optional[ReviewState] = None


class ExportRequest(BaseModel):
    format: str = "yolo"  # "yolo", "coco", "voc", "cvat"
    only_reviewed: bool = False
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
    enable_locate_anything: bool = False
    enable_florence2_verifier: bool = False
    enable_yolo: bool = True
    yolo_models: List[str] = ["yolo11n.pt"]
    yolo_views: Optional[List[str]] = None


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
    enable_locate_anything: bool = False
    enable_florence2_verifier: bool = False
    enable_yolo: bool = True
    yolo_models: List[str] = ["yolo11n.pt"]
    yolo_views: Optional[List[str]] = None


class ValidateModelRequest(BaseModel):
    path: str


class PrepareModelsRequest(BaseModel):
    enable_grounding_dino: bool = True
    enable_sam2_masks: bool = True
    enable_florence2: bool = False
    enable_locate_anything: bool = False
    enable_florence2_verifier: bool = False
    enable_yolo: bool = True
    yolo_models: List[str] = Field(default_factory=lambda: ["yolo11n.pt"], max_length=3)


@app.get("/api/models/status")
def get_model_preparation_status():
    return MODEL_PREPARATION.snapshot()


@app.post("/api/models/prepare")
def prepare_autolabel_models(req: PrepareModelsRequest):
    model_ids = []
    if req.enable_grounding_dino:
        model_ids.append("grounding_dino")
    if req.enable_yolo:
        model_ids.extend(f"yolo:{name}" for name in (req.yolo_models or ["yolo11n.pt"]))
    if req.enable_florence2 or req.enable_florence2_verifier:
        model_ids.append("florence2")
    if req.enable_locate_anything:
        model_ids.append("locate_anything")
    if req.enable_sam2_masks:
        model_ids.append("sam2")
    return MODEL_PREPARATION.start(model_ids)


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
    if AUTOLABEL_ENGINE.uses_cloud:
        return None
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
    if not dataset_dir and ACTIVE_DATASET_DIR is None:
        return []
    return AnnotationRepository(Path(dataset_dir) if dataset_dir else ACTIVE_DATASET_DIR).classes()

@app.on_event("startup")
def on_startup() -> None:
    """Initialize the dataset index; prepare models only after explicit selection."""
    # Initialize SQLite index immediately
    if ACTIVE_DATASET_DIR is not None:
        try:
            get_dataset_index()
        except Exception as err:
            LOGGER.warning("Could not initialize dataset index on startup: %s", err)


def resolve_annotation_path(img_path: Path) -> tuple[Optional[Path], str]:
    return AnnotationRepository(ACTIVE_DATASET_DIR).resolve(img_path)


def get_image_path(image_name: str) -> Path:
    check_dataset(None)
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
    if ACTIVE_DATASET_DIR is None:
        raise HTTPException(409, "No active project. Open a dataset folder first.")
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


def select_colab_images(request: ColabJobRequest):
    with DATASET_STATE_LOCK:
        check_dataset(request.dataset_id)
        root = ACTIVE_DATASET_DIR.resolve()
        repository = AnnotationRepository(root)
        index_db = get_dataset_index()
        images = []
        for record in all_records(index_db):
            image = Path(record["path"])
            if not image.is_file():
                continue
            if request.only_unannotated and _already_annotated(repository, index_db, record, image):
                continue
            images.append(image)
            if len(images) >= request.limit:
                break
        if not images:
            raise HTTPException(422, "No matching images. Choose another dataset or include labeled images.")
        return root, images


app.include_router(create_colab_router(COLAB_CONNECTION, select_colab_images))


@app.on_event("shutdown")
def stop_owned_colab_job():
    COLAB_CONNECTION.cancel()
    worker = COLAB_CONNECTION._thread
    if worker is not None and worker.is_alive():
        worker.join(timeout=65)

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
    if not AUTOLABEL_ENGINE.uses_cloud:
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

    index_db = get_dataset_index() if ACTIVE_DATASET_DIR is not None else None
    db_stats = index_db.stats() if index_db else {}

    return {
        "status": "ready",
        "dataset_dir": str(ACTIVE_DATASET_DIR) if ACTIVE_DATASET_DIR else "",
        "database_connected": index_db is not None,
        "database_stats": db_stats,
        "gpu": gpu_info,
        "inference": {"backend": "cloud" if AUTOLABEL_ENGINE.uses_cloud else "local"},
        "classes": get_dataset_class_names(),
        "models": {
            "yolo11": YOLO_MODEL is not None or bool(AUTOLABEL_ENGINE._yolo_detectors) or AUTOLABEL_ENGINE._yolo_detector is not None,
            "sam2": AUTOLABEL_ENGINE._sam_segmenter is not None,
            "grounding_dino": AUTOLABEL_ENGINE._grounding_detector is not None,
            "florence2": AUTOLABEL_ENGINE._vlm_helper is not None,
            "locate_anything": AUTOLABEL_ENGINE._locate_anything_detector is not None,
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
    if ACTIVE_DATASET_DIR is None:
        return {"images": [], "total": 0, "directory": ""}
    index_db = get_dataset_index()
    records = index_db.list_records(status=status, order_by=order_by, limit=limit, offset=offset)
    images = []
    repo = AnnotationRepository(ACTIVE_DATASET_DIR)
    dataset_root = ACTIVE_DATASET_DIR.resolve()

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
                with repo.lock, repo._write_lock():
                    generation = index_db.annotation_generation()
                    data = repo.read(img_path)
                    ann_count = len(data["boxes"])
                    current_status = data["status"]
                    index_db.set_status_and_count(
                        img_path, current_status, ann_count, r.get("difficulty") or 0,
                        boxes=data["boxes"], expected_generation=generation,
                    )
            except Exception as err:
                raise HTTPException(422, f"Cannot load labels for {img_path.name}: {err}") from err
        else:
            ann_count = cached_count

        images.append({
            "filename": img_path.name,
            "image_id": str(img_path.relative_to(dataset_root)),
            "dataset_id": str(dataset_root),
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



@functools.lru_cache(maxsize=16)
def _enhanced_image_bytes(path: str, mtime_ns: int, preset: str, fmt: str) -> bytes:
    from app.services.preprocess import enhance, resolve_preset

    with Image.open(path) as source:
        result = enhance(source.convert("RGB"), resolve_preset(preset))
    buffer = io.BytesIO()
    result.save(buffer, format=fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    return buffer.getvalue()


@app.get("/api/image/{image_name:path}")
@dataset_operation
def serve_image(image_name: str, enhance: Optional[str] = None):
    """Serve image file content, optionally with a display-only enhancement preset."""
    img_path = get_image_path(image_name)
    is_jpeg = img_path.suffix.lower() in {".jpg", ".jpeg"}
    media_type = "image/jpeg" if is_jpeg else "image/png"
    if enhance and enhance.lower() != "none":
        from app.services.preprocess import PRESETS

        if enhance.lower() not in PRESETS:
            raise HTTPException(422, f"Unknown enhancement preset: {enhance}")
        data = _enhanced_image_bytes(
            str(img_path), img_path.stat().st_mtime_ns, enhance.lower(),
            "JPEG" if is_jpeg else "PNG",
        )
        return Response(data, media_type=media_type)
    return FileResponse(img_path, media_type=media_type)


@app.get("/api/annotations/{image_name:path}")
@dataset_operation
def get_annotations(image_name: str, dataset_id: Optional[str] = None):
    check_dataset(dataset_id or None)
    image = get_image_path(image_name)
    try:
        repo = AnnotationRepository(ACTIVE_DATASET_DIR)
        with repo.lock, repo._write_lock():
            index = get_dataset_index()
            generation = index.annotation_generation()
            data = repo.read(image)
            index.set_metadata(image, data["width"], data["height"])
            record = index.get_image(image) or {}
            index.set_status_and_count(
                image, data["status"], len(data["boxes"]), record.get("difficulty") or 0,
                boxes=data["boxes"], expected_generation=generation,
            )
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
    status = payload.status or "reviewed"
    try:
        repo = AnnotationRepository(ACTIVE_DATASET_DIR)
        with repo.lock, repo._write_lock():
            data = repo.save(image, [b.model_dump() for b in payload.boxes], status=status,
                             expected_revision=payload.expected_revision)
            index = get_dataset_index()
            difficulty = calculate_image_difficulty(image, payload.boxes)
            index.set_status_and_count(
                image, status, len(data["boxes"]), difficulty, boxes=data["boxes"],
            )
    except RevisionConflict as err:
        raise HTTPException(409, str(err)) from err
    except Exception as err:
        raise HTTPException(422, f"Cannot save annotations: {err}") from err
    return {"status": "saved", "image_name": image_name, "count": len(data["boxes"]),
            "box_count": len(data["boxes"]), "difficulty": difficulty,
            "review_status": status, "revision": data["revision"], "boxes": data["boxes"],
            "file": str(repo.canonical_path(image)), "projection_warning": data.get("projection_warning")}

def _resolve_pipeline_mode(
    pipeline_mode_str: str | None,
    enable_sam2_masks: bool,
    enable_yolo: bool,
    enable_grounding_dino: bool,
    enable_florence2: bool,
    yolo_models: list[str] | None = None,
    enable_locate_anything: bool = False,
) -> AutoLabelPipelineMode:
    """Consistently resolve the exact AutoLabelPipelineMode given toggles and settings."""
    mode = None
    if pipeline_mode_str and pipeline_mode_str != "bbox_only":
        try:
            mode = AutoLabelPipelineMode(pipeline_mode_str)
        except Exception:
            mode = None

    if mode is None or enable_locate_anything:
        active_count = sum((enable_grounding_dino, enable_yolo, enable_florence2,
                            enable_locate_anything))
        yolo_count = len(yolo_models or [])
        if active_count > 1 or (enable_yolo and yolo_count > 1):
            mode = (
                AutoLabelPipelineMode.ENSEMBLE_FUSION_SAM2_MASKS
                if enable_sam2_masks
                else AutoLabelPipelineMode.ENSEMBLE_FUSION_BOXES
            )
        elif enable_locate_anything:
            mode = (
                AutoLabelPipelineMode.LOCATE_ANYTHING_SAM2_MASKS
                if enable_sam2_masks
                else AutoLabelPipelineMode.LOCATE_ANYTHING_BOXES
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
            case AutoLabelPipelineMode.LOCATE_ANYTHING_SAM2_MASKS:
                mode = AutoLabelPipelineMode.LOCATE_ANYTHING_BOXES
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
            case AutoLabelPipelineMode.LOCATE_ANYTHING_BOXES:
                mode = AutoLabelPipelineMode.LOCATE_ANYTHING_SAM2_MASKS
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
    views = _validated_views(req.views)

    if AUTOLABEL_ENGINE.uses_cloud:
        classes = req.classes or get_dataset_class_names(str(ACTIVE_DATASET_DIR))
        config = AutoLabelConfig(
            mode=AutoLabelPipelineMode.YOLO_BOXES,
            classes=[AutoLabelClass(name=name) for name in classes],
            confidence_threshold=req.conf_threshold,
            yolo_models=(req.models or [AUTOLABEL_ENGINE.yolo_model_name])[:3],
            enable_grounding_dino=False, enable_florence2=False,
            enable_florence2_verifier=False, enable_sam2_masks=False, enable_yolo=True,
        )
        result = AUTOLABEL_ENGINE.run_preview(img_path, config)
        dataset_classes = AnnotationRepository(ACTIVE_DATASET_DIR).register_classes(
            [d.class_name for d in result.detections],
        )
        boxes = []
        for det in result.detections:
            name = next(c for c in dataset_classes if c.casefold() == det.class_name.casefold())
            boxes.append(BoundingBox(
                id=secrets.token_hex(16), class_name=name, class_id=dataset_classes.index(name),
                confidence=det.confidence,
                x=det.box.left * result.image_width, y=det.box.top * result.image_height,
                width=det.box.width * result.image_width, height=det.box.height * result.image_height,
                norm_left=det.box.left, norm_top=det.box.top,
                norm_right=det.box.right, norm_bottom=det.box.bottom, source="yolo",
            ))
        return {
            "image_name": req.image_name, "width": result.image_width,
            "height": result.image_height, "boxes": [b.model_dump() for b in boxes],
            "count": len(boxes), "models_used": config.yolo_models, "inference_backend": "cloud",
        }

    active_models = req.models if req.models else [
        AUTOLABEL_ENGINE.yolo_model_name if AUTOLABEL_ENGINE.yolo_model_name else "yolo11n.pt"
    ]
    active_models = active_models[:3]

    from app.services.preprocess import apply_view, merge_view_boxes, restore_box

    with Image.open(img_path) as img:
        img_w, img_h = img.size
        view_image = img.convert("RGB") if views != ("original",) else None

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

        per_view = []
        for view in views:
            source = str(img_path) if view == "original" else apply_view(view_image, view)
            view_boxes = []
            for r in model(source, conf=req.conf_threshold):
                names = getattr(model, "names", {}) or getattr(r, "names", {})
                for box in r.boxes:
                    view_boxes.append((
                        names.get(int(box.cls[0]), "unknown"), float(box.conf[0]),
                        restore_box(box.xyxy[0].tolist(), view, img_w),
                    ))
            per_view.append(view_boxes)
        layered = per_view[0]
        for extra in per_view[1:]:
            layered = merge_view_boxes(layered, extra)
        for raw_name, conf, xyxy in layered:
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
        enable_locate_anything=req.enable_locate_anything,
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
        enable_locate_anything=req.enable_locate_anything,
        enable_florence2_verifier=req.enable_florence2_verifier,
        enable_yolo=req.enable_yolo,
        yolo_models=req.yolo_models or ["yolo11n.pt"],
        yolo_views=_validated_views(req.yolo_views),
    )

    if req.enable_yolo:
        yolo = get_yolo_model()
        if yolo is not None and AUTOLABEL_ENGINE._yolo_detector is None:
            AUTOLABEL_ENGINE._yolo_detector = yolo

    try:
        result = AUTOLABEL_ENGINE.run_preview(img_path, config)
    except Exception as err:
        if AUTOLABEL_ENGINE.uses_cloud:
            detail = str(err) if isinstance(err, CloudInferenceError) else "Cloud inference failed"
            raise HTTPException(503, detail) from None
        if req.enable_locate_anything:
            raise HTTPException(503, f"Locate Anything 3B annotation failed: {err}") from None
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

    if not result.detections and not req.enable_locate_anything:
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
        enable_locate_anything=req.enable_locate_anything,
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
        enable_locate_anything=req.enable_locate_anything,
        enable_florence2_verifier=req.enable_florence2_verifier,
        enable_yolo=req.enable_yolo,
        yolo_models=req.yolo_models or ["yolo11n.pt"],
        yolo_views=tuple(req.yolo_views or ("original",)),
    )

    if req.enable_yolo:
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
                    source=str(mode.annotation_source),
                    polygon_normalized=det.polygon_normalized or None,
                    polygon_pixels=det.polygon_pixels or None,
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

            with repo.lock, repo._write_lock():
                saved = repo.save(p, [b.model_dump() for b in final_boxes], status="ai_labeled",
                                  expected_revision=existing_data["revision"])
                if saved.get("projection_warning"):
                    warnings.append(saved["projection_warning"])
                diff = calculate_image_difficulty(p, final_boxes)
                index_db.set_status_and_count(
                    p, "ai_labeled", len(final_boxes), diff, boxes=saved["boxes"],
                )
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


def _already_annotated(repo: AnnotationRepository, index_db: DatasetIndex,
                       record: dict, path: Path) -> bool:
    """Whether an image was reviewed, auto-labeled, or already has boxes (e.g. imported labels).

    The index answers for every image it has probed; only never-probed images (count -1) need
    their label files read, so starting a batch does not open every image in the dataset.
    """
    status = record.get("status")
    count = record.get("annotation_count", -1)
    if status in {"reviewed", "ai_labeled"}:
        return True
    if count is None or count < 0:
        with repo.lock, repo._write_lock():
            generation = index_db.annotation_generation()
            data = repo.read(path)
            status, count = data["status"], len(data["boxes"])
            index_db.set_status_and_count(
                path, status, count, record.get("difficulty") or 0,
                boxes=data["boxes"], expected_generation=generation,
            )
        if status in {"reviewed", "ai_labeled"}:
            return True
    return count > 0


@app.post("/api/autolabel/batch")
@dataset_operation
def start_autolabel_batch(req: AutoLabelBatchRequest, background_tasks: BackgroundTasks):
    global AUTOLABEL_BATCH_STATUS
    check_dataset(req.dataset_id)
    _validated_views(req.yolo_views)  # reject bad views before the job is marked running
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
            if req.only_unannotated and _already_annotated(repo, get_dataset_index(), record, path):
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


class CloseProjectRequest(BaseModel):
    dataset_id: str


@app.post("/api/dataset/close")
@dataset_operation
def close_project(req: CloseProjectRequest):
    """Close the current project, preserving every dataset file and annotation."""
    global ACTIVE_DATASET_DIR, DATASET_INDEX
    check_dataset(req.dataset_id)
    if AUTOLABEL_BATCH_STATUS.get("running") or IMPORT_STATUS.get("running"):
        raise HTTPException(409, "Tunggu auto-label atau impor anotasi selesai sebelum menghapus proyek.")
    if COLAB_CONNECTION.status().get("busy"):
        raise HTTPException(409, "Tunggu proses Colab selesai sebelum menghapus proyek.")
    if not persist_active_dataset_dir(None):
        raise HTTPException(500, "Proyek belum ditutup: pengaturan tidak bisa disimpan. Coba lagi.")
    if DATASET_INDEX is not None:
        DATASET_INDEX.close()
        DATASET_INDEX = None
    ACTIVE_DATASET_DIR = None
    return {"status": "closed", "dataset_id": req.dataset_id}


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
            start_dir = str(ACTIVE_DATASET_DIR) if ACTIVE_DATASET_DIR and ACTIVE_DATASET_DIR.is_dir() else str(PROJECT_ROOT)
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
            if c is not None and c.is_dir() and str(c.resolve()) not in results:
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


UPLOAD_CHUNK_BYTES = 1024 * 1024
# Upload workers await UploadFile.read(), which can itself need the default thread
# pool for disk-spooled multipart files. Keep a separate, bounded admission limit
# so simultaneous uploads cannot occupy every token needed by their own readers.
UPLOAD_WORKERS = CapacityLimiter(4)


def _check_upload_collisions(root: Path, names: list[str]) -> None:
    """Check case-insensitive collisions, including directories and dangling symlinks."""
    existing = {path.name.casefold() for path in root.iterdir()}
    for name in names:
        if name.casefold() in existing:
            raise HTTPException(409, f"Image already exists: {name}")


def _commit_uploads(root: Path, staged: list[tuple[str, Path, int, int]]) -> dict:
    """Publish verified files exclusively and index them as one batch, in a worker."""
    with DATASET_STATE_LOCK:
        check_dataset(str(root))
        _check_upload_collisions(root, [name for name, *_ in staged])
        index = get_dataset_index()
        saved = []
        try:
            for name, source, width, height in staged:
                target = root / name
                # Same-filesystem hard links publish the complete staged file atomically,
                # and unlike rename(), fail if any other writer created the target.
                try:
                    os.link(source, target)
                except FileExistsError:
                    raise
                except OSError as error:
                    if error.errno not in {
                        errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP,
                        errno.EXDEV, errno.ENOSYS, errno.EMLINK,
                    } and getattr(error, "winerror", None) not in {1, 5, 17, 50, 1142, 1314}:
                        raise
                    # Some filesystems cannot link. Exclusive creation retains collision
                    # protection; register ownership before copying for partial rollback.
                    with target.open("xb") as destination:
                        saved.append((target, width, height))
                        with source.open("rb") as origin:
                            while chunk := origin.read(UPLOAD_CHUNK_BYTES):
                                destination.write(chunk)
                else:
                    saved.append((target, width, height))
            count = index.add_paths(saved)
        except Exception as err:
            for target, *_ in saved:
                target.unlink(missing_ok=True)
            raise HTTPException(409, "Upload could not be committed") from err
        names = [target.name for target, *_ in saved]
        return {"saved": names, "count": len(names), "total_indexed": count}


def _stage_and_upload_images(files: List[UploadFile]) -> dict:
    """Bound memory and disk use; all filesystem, Pillow, and lock work stays here."""
    with DATASET_STATE_LOCK:
        check_dataset(None)
        root = ACTIVE_DATASET_DIR.resolve()

    names = []
    seen = set()
    for upload in files:
        name = Path(upload.filename or "").name
        if not name or Path(name).suffix.lower() not in IMAGE_SUFFIXES:
            raise HTTPException(422, "Upload must be a supported image")
        if name.casefold() in seen:
            raise HTTPException(409, f"Image already exists: {name}")
        names.append(name)
        seen.add(name.casefold())

    with DATASET_STATE_LOCK:
        check_dataset(str(root))
        _check_upload_collisions(root, names)
        # Initialize the index before publishing new files if this dataset is cold.
        get_dataset_index()

    # Hidden staging is excluded from scans and lives on the target filesystem.
    # TemporaryDirectory also cleans partial writes, read failures, and invalid batches.
    with tempfile.TemporaryDirectory(prefix=".visionlab-upload-", dir=root) as directory:
        staged = []
        total_bytes = 0
        for number, (upload, name) in enumerate(zip(files, names)):
            source = Path(directory) / str(number)
            file_bytes = 0
            with source.open("xb") as stream:
                while chunk := from_thread.run(upload.read, UPLOAD_CHUNK_BYTES):
                    file_bytes += len(chunk)
                    total_bytes += len(chunk)
                    # Share the existing image (20 MiB) and transfer (50 MiB) caps.
                    if file_bytes > MAX_IMAGE_BYTES or total_bytes > MAX_TRANSFER_BYTES:
                        raise HTTPException(413, "Upload exceeds image or batch size limit")
                    stream.write(chunk)
            try:
                with Image.open(source) as image:
                    width, height = image.size
                    image.verify()
            except Exception as err:
                raise HTTPException(422, f"Invalid image: {name}") from err
            staged.append((name, source, width, height))
        return _commit_uploads(root, staged)


@app.post("/api/dataset/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """Stream images through a worker; only bounded async reads run on the event loop."""
    return await to_thread.run_sync(_stage_and_upload_images, files, limiter=UPLOAD_WORKERS)


@app.post("/api/dataset/classes")
@dataset_operation
def register_class(payload: Dict[str, str]):
    check_dataset(payload.get("dataset_id"))
    name = payload.get("name", "").strip()
    if not name:
        raise HTTPException(422, "Class name is required")
    return {"classes": AnnotationRepository(ACTIVE_DATASET_DIR).register_classes([name])}


class ClassChangeRequest(BaseModel):
    dataset_id: str
    expected_classes: list[str]
    action: Literal["rename", "delete"]
    name: str
    target: str | None = None
    delete_annotations: bool = False


@app.patch("/api/dataset/classes")
@dataset_operation
def manage_class(req: ClassChangeRequest):
    check_dataset(req.dataset_id)
    if (
        AUTOLABEL_BATCH_STATUS.get("running")
        or IMPORT_STATUS.get("running")
        or COLAB_CONNECTION.status().get("busy")
    ):
        raise HTTPException(409, "Tunggu auto-label, impor, atau Colab selesai sebelum mengubah kelas.")
    if not INFERENCE_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Tunggu deteksi atau audit selesai sebelum mengubah kelas.")
    try:
        get_dataset_index().invalidate_annotations()
        return change_class(
            AnnotationRepository(ACTIVE_DATASET_DIR), req.action, req.name, req.target,
            req.expected_classes, delete_annotations=req.delete_annotations,
        )
    except RepositoryError as error:
        raise HTTPException(409, str(error)) from error
    except (OSError, yaml.YAMLError, sqlite3.Error) as error:
        raise HTTPException(500, f"Kelas belum diubah: {error}") from error
    finally:
        INFERENCE_LOCK.release()


@app.post("/api/dataset/rescan")
@dataset_operation
def rescan_dataset():
    """Rescan the active dataset directory and reindex all images."""
    index_db = get_dataset_index()
    count = index_db.scan(ACTIVE_DATASET_DIR, refresh_annotations=True)
    return {"status": "ok", "indexed_count": count, "directory": str(ACTIVE_DATASET_DIR)}


@app.get("/api/dataset/stats")
def get_dataset_stats():
    """Dataset counts for the quality page.

    This deliberately does not hold the dataset lock for its whole run: on a large dataset it
    takes seconds, and every other dataset call would freeze behind it. Images the index
    already knows have no boxes are skipped. Unknown class histograms are rebuilt from
    annotations; subsequent requests aggregate the cached histograms in SQL.
    """
    with DATASET_STATE_LOCK:
        if ACTIVE_DATASET_DIR is None:
            return {"directory": "", "total_images": 0, "total_annotations": 0,
                    "reviewed": 0, "unreviewed": 0, "ai_labeled": 0,
                    "class_counts": {}, "database_file": ""}
        root = ACTIVE_DATASET_DIR.resolve()
        index_db = get_dataset_index()
    repo = AnnotationRepository(root)

    def registry_identity():
        # Detect a class migration between fills without rereading the registry
        # or blocking repository writers throughout the whole scan.
        result = []
        for name in ("data.yaml", "dataset.yaml"):
            try:
                stat = (root / name).stat()
                result.append((stat.st_mtime_ns, stat.st_size))
            except FileNotFoundError:
                result.append(None)
        return result

    try:
        with repo.lock, repo._write_lock():
            classes = repo.classes()
            registry = registry_identity()
            generation = index_db.annotation_generation()
        class_counts: Dict[str, int] = {c: 0 for c in classes}
        for record in index_db.unknown_histograms():
            # Coordinate each fill with saves/migrations, including other
            # processes. Release these locks between images for other requests.
            with repo.lock, repo._write_lock():
                if registry_identity() != registry:
                    raise HTTPException(409, "Dataset classes changed; reload before continuing")
                image = Path(record["path"])
                try:
                    data = repo.read(image, classes=classes)
                except (RepositoryError, OSError, ValueError) as err:
                    LOGGER.warning("Skipping unreadable labels for %s: %s", image.name, err)
                    continue
                updated = index_db.set_status_and_count(
                    image, data["status"], len(data["boxes"]), record.get("difficulty") or 0,
                    boxes=data["boxes"], expected_generation=generation)
                if not updated:
                    raise HTTPException(409, "Dataset annotations changed; reload before continuing")
        with repo.lock, repo._write_lock():
            if registry_identity() != registry:
                raise HTTPException(409, "Dataset classes changed; reload before continuing")
            class_counts.update(index_db.class_counts())
            stats = index_db.stats()
            if index_db.annotation_generation() != generation:
                raise HTTPException(409, "Dataset annotations changed; reload before continuing")
    except sqlite3.ProgrammingError as err:  # the dataset was switched while this ran
        raise HTTPException(409, "Dataset changed; reload before continuing") from err
    return {
        "directory": str(root),
        "total_images": stats["total"],
        "unreviewed": stats["unreviewed"],
        "reviewed": stats["reviewed"],
        "ai_labeled": stats["ai_labeled"],
        "class_counts": class_counts,
        "total_annotations": sum(class_counts.values()),
        "database_file": str(index_db._database_path),
    }


@app.post("/api/dataset/export")
@dataset_operation
def export_dataset(req: ExportRequest):
    """Export the dataset as YOLO, COCO (with train/val/test splits), Pascal VOC or CVAT XML."""
    export_format = req.format.lower()
    if export_format not in {"yolo", "coco", "voc", "cvat"}:
        raise HTTPException(422, f"Unsupported export format: {req.format}")
    index_db = get_dataset_index()
    records = all_records(index_db)
    documents: List[AnnotationDocument] = []
    repo = AnnotationRepository(ACTIVE_DATASET_DIR)
    skipped_unreviewed = 0
    for record in records:
        if req.only_reviewed and record.get("status") != "reviewed":
            skipped_unreviewed += 1
            continue
        image = Path(record["path"])
        if not image.is_file():
            raise HTTPException(422, "An indexed source image is missing; rescan before exporting")
        try:
            documents.append(repo.document(image))
        except Exception as err:
            raise HTTPException(422, f"Cannot export {image.name}: {err}") from err

    if not documents:
        detail = ("No approved images to export yet." if req.only_reviewed and skipped_unreviewed
                  else "No valid images to export.")
        raise HTTPException(status_code=400, detail=detail)

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
    out_dir = Path(req.output_dir) if req.output_dir else PROJECT_ROOT / "exports" / f"{export_format}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Execute exporter
    try:
        if export_format == "coco":
            exporter = CocoExporter(splits=splits, class_order=repo.classes())
        elif export_format == "voc":
            exporter = PascalVocExporter()
        elif export_format == "cvat":
            exporter = CvatExporter()
        else:
            exporter = YoloExporter(splits=splits, class_order=repo.classes())
        artifact = exporter.export(documents, out_dir)
    except Exception as err:
        LOGGER.exception("Export failed")
        raise HTTPException(status_code=500, detail=f"Export generation failed: {err}")

    return {
        "status": "success",
        "format": export_format,
        "destination": str(out_dir),
        "artifact": str(artifact),
        "splits": {
            "train": len(splits.get("train", [])),
            "val": len(splits.get("val", [])),
            "test": len(splits.get("test", [])),
        },
        "total_documents": len(documents),
        "skipped_unreviewed": skipped_unreviewed,
    }


class AuditRequest(BaseModel):
    image_name: str
    dataset_id: Optional[str] = None
    conf_threshold: float = Field(default=0.08, ge=0.01, le=0.9)
    models: Optional[List[str]] = None
    tiled: bool = True
    views: Optional[List[str]] = None
    boxes: Optional[List[BoundingBox]] = None

    @model_validator(mode="after")
    def validate_views(self):
        from app.services.preprocess import VIEWS

        unknown = [view for view in self.views or [] if view not in VIEWS]
        if unknown:
            raise ValueError(f"unknown detection views: {unknown}; choose from {sorted(VIEWS)}")
        return self


def _audit_predictions(img_path: Path, req: AuditRequest, classes: list[str]):
    """Collect high-recall reference detections for the annotation audit."""
    from app.services.audit import Prediction
    from app.services.audit.detector import detect_reference, merge_predictions

    if AUTOLABEL_ENGINE.uses_cloud:
        config = AutoLabelConfig(
            mode=AutoLabelPipelineMode.YOLO_BOXES,
            classes=[AutoLabelClass(name=name) for name in classes],
            confidence_threshold=req.conf_threshold,
            yolo_models=(req.models or [AUTOLABEL_ENGINE.yolo_model_name])[:3],
            enable_grounding_dino=False, enable_florence2=False,
            enable_florence2_verifier=False, enable_sam2_masks=False, enable_yolo=True,
        )
        result = AUTOLABEL_ENGINE.run_preview(img_path, config)
        return [Prediction(d.class_name, d.box, d.confidence) for d in result.detections]

    predictions: list = []
    with Image.open(img_path) as source:
        image = source.convert("RGB")
    tile_size = 640 if req.tiled else max(image.size)
    for model_name in (req.models or [AUTOLABEL_ENGINE.yolo_model_name or "yolo11n.pt"])[:3]:
        try:
            model = AUTOLABEL_ENGINE._get_yolo_detector(model_name)
        except Exception as err:
            LOGGER.warning("Could not load YOLO model '%s': %s", model_name, err)
            model = get_yolo_model()
            if model is None:
                continue
        predictions.extend(detect_reference(
            model, image, confidence=req.conf_threshold, tile_size=tile_size,
            allowed_classes=classes or None,
            **({"views": req.views} if req.views else {}),
        ))
    return merge_predictions(predictions, 0.6)


@app.post("/api/audit")
@dataset_operation
@inference_operation
def audit_image(req: AuditRequest):
    """Flag likely-missed (often occluded) objects and suspicious boxes in one image."""
    from app.services.annotation.domain import BoundingBox as DomainBox
    from app.services.audit import AuditBox, audit_annotations
    from app.services.audit.payload import report_payload

    check_dataset(req.dataset_id)
    img_path = get_image_path(req.image_name)
    repo = AnnotationRepository(ACTIVE_DATASET_DIR)
    if req.boxes is not None:
        stored = [b.model_dump() for b in req.boxes]
    else:
        stored = repo.read(img_path)["boxes"]
    with Image.open(img_path) as img:
        img_w, img_h = img.size
    existing = [
        AuditBox(
            id=str(b["id"]), class_name=b["class_name"],
            box=DomainBox(b["norm_left"], b["norm_top"], b["norm_right"], b["norm_bottom"]),
            confidence=b.get("confidence"), source=b.get("source", "human"),
        )
        for b in stored
    ]
    classes = get_dataset_class_names(str(ACTIVE_DATASET_DIR))
    predictions = _audit_predictions(img_path, req, classes)
    report = audit_annotations(existing, predictions, img_w, img_h)
    payload = report_payload(report, img_w, img_h)
    payload.update(image_name=req.image_name, width=img_w, height=img_h,
                   prediction_count=len(predictions))
    return payload


# ------------------------------------------------------------------------------
# Import existing annotations (COCO)
# ------------------------------------------------------------------------------

IMPORT_LOCK = threading.Lock()
IMPORT_STATUS: Dict[str, Any] = {"running": False, "completed": False, "error": None}


class AnnotationImportRequest(BaseModel):
    sources: List[str] = Field(min_length=1, max_length=20)
    overwrite: bool = False
    dataset_id: Optional[str] = None


def _source_payload(source, root: Path) -> dict:
    def relative(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(root))
        except ValueError:
            return str(path)

    return {
        "path": str(source.path),
        "relative": relative(source.path),
        "image_dir": relative(source.image_root),
        "images": source.images,
        "found_images": source.found_images,
        "annotations": source.annotations,
        "categories": list(source.categories),
    }


@app.get("/api/dataset/annotation-sources")
def list_annotation_sources():
    """COCO annotation files found in the active dataset that can be imported."""
    from app.services.dataset.annotation_import import (
        CocoFormatError,
        describe_source,
        find_coco_files,
    )

    with DATASET_STATE_LOCK:
        if ACTIVE_DATASET_DIR is None:
            return {"directory": "", "sources": [], "skipped": []}
        root = ACTIVE_DATASET_DIR.resolve()
    sources, skipped = [], []
    for path in find_coco_files(root):
        try:
            sources.append(_source_payload(describe_source(path), root))
        except (CocoFormatError, OSError) as err:
            skipped.append({"path": str(path), "error": str(err)})
    return {"directory": str(root), "sources": sources, "skipped": skipped}


def _import_job_snapshot() -> dict:
    with IMPORT_LOCK:
        return dict(IMPORT_STATUS)


def _run_annotation_import(root: Path, files: List[Path], overwrite: bool, job: dict) -> None:
    from app.services.dataset.annotation_import import ImportCounts, import_coco

    from app.services.dataset.annotation_import import describe_source

    repo = AnnotationRepository(root)
    index_db = DatasetIndex(root / ".dataset_index.sqlite")
    # The grand total is known up front so the progress bar moves smoothly across several files.
    counts = ImportCounts(total_images=sum(describe_source(f).images for f in files))
    by_name: Dict[str, List[Path]] = {}

    def locate_by_name(name: str) -> Optional[Path]:
        """Last resort for a file name that is not where the annotation file says."""
        if not by_name:
            for record in all_records(index_db):
                by_name.setdefault(Path(record["path"]).name, []).append(Path(record["path"]))
        matches = by_name.get(name, [])
        return matches[0] if len(matches) == 1 else None

    def publish(label: str) -> None:
        with IMPORT_LOCK:
            job.update(
                current_file=label, total=counts.total_images, processed=counts.processed,
                imported=counts.imported, boxes=counts.boxes,
                skipped_existing=counts.skipped_existing,
                skipped_conflict=counts.skipped_conflict,
                missing_images=counts.missing_images, invalid_boxes=counts.invalid_boxes,
                orphan_boxes=counts.orphan_boxes, classes_added=list(counts.classes_added),
                failed_count=len(counts.failures),
            )

    def on_saved(image: Path, saved: dict) -> None:
        boxes = [BoundingBox(**box) for box in saved["boxes"]]
        index_db.set_status_and_count(image, saved["status"], len(boxes),
                                      calculate_image_difficulty(image, boxes), boxes=saved["boxes"])

    try:
        for annotation_file in files:
            import_coco(
                repo, annotation_file, overwrite=overwrite, counts=counts,
                progress=lambda c, label: publish(label), on_saved=on_saved,
                locate_fallback=locate_by_name, count_total=False,
            )
        with IMPORT_LOCK:
            job["failures"] = list(counts.failures)
        publish("")
    except Exception as err:
        LOGGER.exception("Annotation import failed")
        with IMPORT_LOCK:
            job["error"] = str(err)
    finally:
        index_db.close()
        with IMPORT_LOCK:
            job.update(running=False, completed=True)


@app.post("/api/dataset/import-annotations")
def start_annotation_import(req: AnnotationImportRequest):
    """Import COCO annotation files into the active dataset as a background job."""
    from app.services.dataset.annotation_import import CocoFormatError, describe_source

    with DATASET_STATE_LOCK:  # only to pin the dataset; parsing the files happens unlocked
        check_dataset(req.dataset_id)
        root = ACTIVE_DATASET_DIR.resolve()
    files: List[Path] = []
    for value in req.sources:
        path = Path(value)
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_file() or path.suffix.lower() != ".json":
            raise HTTPException(422, f"Annotation file not found: {value}")
        try:
            describe_source(path)
        except CocoFormatError as err:
            raise HTTPException(422, str(err)) from err
        files.append(path)
    with IMPORT_LOCK:
        if IMPORT_STATUS.get("running"):
            raise HTTPException(409, "An annotation import is already running")
        IMPORT_STATUS.clear()
        IMPORT_STATUS.update(
            running=True, completed=False, error=None, job_id=secrets.token_hex(12),
            dataset_id=str(root), total=0, processed=0, imported=0, boxes=0,
            skipped_existing=0, skipped_conflict=0, missing_images=0, invalid_boxes=0,
            orphan_boxes=0, classes_added=[], failed_count=0, failures=[], current_file="",
            files=[str(f) for f in files],
        )
        job = IMPORT_STATUS
    threading.Thread(target=_run_annotation_import, args=(root, files, req.overwrite, job),
                     daemon=True).start()
    return {"status": "started", "job_id": job["job_id"], "files": len(files)}


@app.get("/api/dataset/import-annotations/status")
def annotation_import_status():
    return _import_job_snapshot()


@app.get("/api/active-learning/queue")
@dataset_operation
def active_learning_queue(limit: int = Query(20, ge=1, le=200)):
    """Retrieve the highest-difficulty images prioritized for human review."""
    if ACTIVE_DATASET_DIR is None:
        return {"queue": [], "count": 0}
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
    if AUTOLABEL_ENGINE.uses_cloud:
        raise HTTPException(422, "Cloud mode uses worker model weights; laptop paths are not supported")
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
