"""Main application window and initial dock layout."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any

from PySide6.QtCore import QEvent, QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QBrush,
    QColor,
    QIcon,
    QImage,
    QImageReader,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QInputDialog,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.runtime import detect_gpu
from app.export.exporters import CocoExporter, YoloExporter, split_documents
from app.models.contracts import Detection as ModelDetection
from app.services.active_learning import ActiveLearningConfig, ActiveLearningEngine, ImageAnalysis
from app.services.active_learning.active_learning_models import DifficultyResult
from app.services.annotation.domain import (
    Annotation,
    AnnotationDocument,
    AnnotationSource,
    BoundingBox,
)
from app.services.annotation.history import (
    AddAnnotationCommand,
    AnnotationHistory,
    RemoveAnnotationCommand,
    ReplaceDocumentCommand,
    UpdateAnnotationCommand,
)
from app.services.crop_assisted import CropGenerator, CropMerger, CropSession
from app.services.dataset.coco_importer import CocoImporter, CocoImportResult
from app.services.dataset.index import IMAGE_SUFFIXES
from app.services.fusion import (
    FusionConfig,
    FusionEngine,
    FusionResult,
    FusionStatus,
    remove_overlapping_annotations,
)
from app.services.inference.dense_motorcycle import DenseInferenceConfig, DenseMotorcycleInference
from app.services.inference.grounding import grounding_class, prompt_variants, tile_positions
from app.ui.canvas.annotation_canvas import AnnotationCanvas, CanvasMode
from app.ui.views.image_browser import ImageBrowser

LOGGER = logging.getLogger(__name__)


class _ActiveLearningSignals(QObject):
    """Signals emitted by a background active-learning calculation."""

    completed = Signal(object)
    failed = Signal(str)


class _ActiveLearningTask(QRunnable):
    """Run one active-learning score without blocking the Qt event loop."""

    def __init__(self, engine: ActiveLearningEngine, analysis: ImageAnalysis) -> None:
        super().__init__()
        self.signals = _ActiveLearningSignals()
        self._engine = engine
        self._analysis = analysis

    def run(self) -> None:
        try:
            self.signals.completed.emit(self._engine.score(self._analysis))
        except Exception as error:
            LOGGER.exception("active-learning calculation failed")
            self.signals.failed.emit(str(error))


class _DatasetAnnotationSignals(QObject):
    """Signals emitted while processing a complete dataset."""

    progress = Signal(int, str)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class _DatasetProgressDialog(QDialog):
    """Readable, cancellable progress window for long dataset operations."""

    cancelled = Signal()

    def __init__(self, title: str, total: int, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(560)
        self.setModal(False)
        layout = QVBoxLayout(self)
        self._phase = QLabel("Preparing...")
        self._phase.setWordWrap(True)
        self._phase.setMinimumHeight(42)
        self._progress = QProgressBar()
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._cancel = QPushButton("Cancel")
        self._cancel.clicked.connect(self.cancelled.emit)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self._cancel)
        layout.addWidget(self._phase)
        layout.addWidget(self._progress)
        layout.addLayout(buttons)

    def update_progress(self, value: int, message: str) -> None:
        self._progress.setValue(value)
        self._phase.setText(message)


class _DatasetAnnotationTask(QRunnable):
    """Run configurable Grounding DINO and YOLO dataset inference off the UI thread."""

    def __init__(
        self,
        documents: list[AnnotationDocument],
        grounding_model,
        grounding_processor,
        yolo_model,
        prompt: str,
        confidence: float,
        iou_threshold: float,
        containment_threshold: float,
        use_yolo: bool = True,
        tile_size: int = 512,
        tile_overlap: float = 0.25,
        enabled_classes: set[str] | None = None,
        sam2_model=None,
        sam2_processor=None,
        use_sam2: bool = False,
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.signals = _DatasetAnnotationSignals()
        self._documents = documents
        self._grounding_model = grounding_model
        self._grounding_processor = grounding_processor
        self._sam2_model = sam2_model
        self._sam2_processor = sam2_processor
        self._use_sam2 = use_sam2
        self._yolo_model = yolo_model
        self._prompt = prompt
        self._prompts = prompt_variants(prompt)
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._containment_threshold = containment_threshold
        self._use_yolo = use_yolo
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._dense_inference = DenseMotorcycleInference(
            grounding_model,
            grounding_processor,
            yolo_model,
            DenseInferenceConfig(
                yolo_confidence=min(confidence, 0.05),
                dino_box_threshold=min(confidence, 0.12),
                dino_text_threshold=min(confidence, 0.18),
                enabled_classes=frozenset(enabled_classes)
                if enabled_classes
                else frozenset({"motorcycle", "car", "bus", "truck"}),
            ),
        )
        self._cancel_requested = Event()

    def cancel(self) -> None:
        """Request cancellation after the current model inference finishes."""
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            from PIL import Image

            results: dict[Path, AnnotationDocument] = {}
            added_total = 0
            removed_total = 0
            for index, document in enumerate(self._documents, start=1):
                if self._cancel_requested.is_set():
                    self.signals.cancelled.emit()
                    return
                preserved = tuple(
                    item
                    for item in document.annotations
                    if item.source
                    not in {AnnotationSource.GROUNDING_DINO, AnnotationSource.FUSED}
                    and not (
                        self._use_yolo and item.source is AnnotationSource.YOLO
                    )
                )
                inference_document = AnnotationDocument(
                    document.image_path,
                    document.image_width,
                    document.image_height,
                    preserved,
                )
                yolo_predictions, dino_predictions = self._dense_inference.predict(
                    document,
                    self._prompt,
                    use_yolo=self._use_yolo,
                )
                if self._use_yolo:
                    predictions = self._supplement_yolo(yolo_predictions, dino_predictions)
                else:
                    predictions = self._resolve_class_conflicts(dino_predictions)
                additions = self._merge_predictions(
                    inference_document,
                    predictions,
                    same_class_only=True,
                )
                kept = tuple(
                    self._nms_predictions(additions, self._dense_inference.config.nms_iou)
                )
                if self._use_sam2 and self._sam2_model is not None and self._sam2_processor is not None:
                    image = Image.open(document.image_path).convert("RGB")
                    kept = tuple(self._refine_annotations_sam2(image, list(kept)))
                removed = len(additions) - len(kept)
                updated = AnnotationDocument(
                    document.image_path,
                    document.image_width,
                    document.image_height,
                    (*preserved, *kept),
                )
                results[document.image_path] = updated
                added_total += len(kept)
                removed_total += removed
                self.signals.progress.emit(
                    index,
                    f"{document.image_path.name} | YOLO {len(yolo_predictions)} | "
                    f"DINO {len(dino_predictions)} | added {len(kept)} | "
                    f"removed {removed}",
                )
            self.signals.completed.emit((results, added_total, removed_total))
        except Exception as error:
            LOGGER.exception("dataset annotation failed")
            self.signals.failed.emit(str(error))

    def _refine_annotations_sam2(self, image, annotations: list[Annotation]) -> list[Annotation]:  # type: ignore[no-untyped-def]
        if not annotations:
            return []
        import torch

        try:
            boxes = [
                [
                    ann.box.left * image.width,
                    ann.box.top * image.height,
                    ann.box.right * image.width,
                    ann.box.bottom * image.height,
                ]
                for ann in annotations
            ]
            pixel_boxes = [boxes]
            inputs = self._sam2_processor(
                images=image, input_boxes=pixel_boxes, return_tensors="pt"
            )
            device = next(self._sam2_model.parameters()).device
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.no_grad():
                outputs = self._sam2_model(**inputs, multimask_output=False)
            masks = self._sam2_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"]
            )
            mask_batch = masks[0]
            refined: list[Annotation] = []
            for idx, ann in enumerate(annotations):
                mask = mask_batch[idx].squeeze()
                rows, columns = torch.where(mask > 0)
                if rows.numel() == 0:
                    refined.append(ann)
                else:
                    refined_box = BoundingBox(
                        float(columns.min()) / image.width,
                        float(rows.min()) / image.height,
                        float(columns.max() + 1) / image.width,
                        float(rows.max() + 1) / image.height,
                    )
                    refined.append(replace(ann, box=refined_box, source=AnnotationSource.SAM2))
            return refined
        except Exception:
            return annotations

    def _grounding_detections(self, document: AnnotationDocument) -> list[Annotation]:
        import torch
        from PIL import Image

        image = Image.open(document.image_path).convert("RGB")
        detections = []
        for prompt in self._prompts:
            for crop, offset_x, offset_y, is_tile in self._grounding_crops(image):
                inputs = self._grounding_processor(
                    images=crop,
                    text=prompt,
                    return_tensors="pt",
                )
                device = next(self._grounding_model.parameters()).device
                inputs = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
                with torch.no_grad():
                    outputs = self._grounding_model(**inputs)
                result = self._post_process_grounding(
                    outputs,
                    inputs["input_ids"],
                    (crop.height, crop.width),
                )
                labels = result.get("text_labels", result.get("labels", ()))
                for index, (box, score) in enumerate(
                    zip(result["boxes"], result["scores"], strict=True)
                ):
                    if index >= len(labels):
                        continue
                    class_name = grounding_class(str(labels[index]))
                    if class_name is None:
                        continue
                    left, top, right, bottom = box.tolist()
                    if is_tile and (
                        left <= 2
                        or top <= 2
                        or right >= crop.width - 2
                        or bottom >= crop.height - 2
                    ):
                        continue
                    annotation = self._annotation(
                        class_name,
                        left + offset_x,
                        top + offset_y,
                        right + offset_x,
                        bottom + offset_y,
                        float(score),
                        AnnotationSource.GROUNDING_DINO,
                        document,
                    )
                    if annotation is not None and annotation.box.area <= 0.75:
                        detections.append(annotation)
        return self._nms_predictions(detections, self._iou_threshold)

    @staticmethod
    def _nms_predictions(
        predictions: list[Annotation], iou_threshold: float
    ) -> list[Annotation]:
        """Suppress same-class duplicates without containment suppression."""
        kept: list[Annotation] = []
        for prediction in sorted(
            predictions,
            key=lambda item: item.confidence if item.confidence is not None else 0.0,
            reverse=True,
        ):
            if any(
                existing.class_name == prediction.class_name
                and MainWindow._box_iou(existing.box, prediction.box) >= iou_threshold
                for existing in kept
            ):
                continue
            kept.append(prediction)
        return kept

    def _supplement_yolo(
        self,
        yolo_predictions: list[Annotation],
        dino_predictions: list[Annotation],
    ) -> list[Annotation]:
        """Keep YOLO boxes authoritative and add only useful DINO proposals."""
        result = list(yolo_predictions)
        for prediction in dino_predictions:
            if prediction.class_name != "motorcycle":
                continue
            if any(
                item.class_name == "motorcycle"
                and MainWindow._box_iou(item.box, prediction.box) >= 0.3
                for item in yolo_predictions
            ):
                continue
            result.append(prediction)
        return self._resolve_class_conflicts(result, preserve_yolo=True)

    def _resolve_class_conflicts(
        self,
        predictions: list[Annotation],
        preserve_yolo: bool = False,
    ) -> list[Annotation]:
        """Remove competing classes."""
        ordered = sorted(
            predictions,
            key=lambda item: (
                1 if preserve_yolo and item.source is AnnotationSource.YOLO else 0,
                item.confidence if item.confidence is not None else 0.0,
            ),
            reverse=True,
        )
        kept: list[Annotation] = []
        for prediction in ordered:
            conflict = any(
                existing.class_name != prediction.class_name
                and MainWindow._box_iou(existing.box, prediction.box) >= self._iou_threshold
                for existing in kept
            )
            if not conflict:
                kept.append(prediction)
        return kept

    def _post_process_grounding(self, outputs, input_ids, target_size):  # type: ignore[no-untyped-def]
        """Support both Transformers Grounding DINO threshold parameter names."""
        try:
            return self._grounding_processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                threshold=self._confidence,
                text_threshold=self._confidence,
                target_sizes=[target_size],
            )[0]
        except TypeError:
            return self._grounding_processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                box_threshold=self._confidence,
                text_threshold=self._confidence,
                target_sizes=[target_size],
            )[0]

    def _grounding_crops(self, image):  # type: ignore[no-untyped-def]
        if image.width <= self._tile_size and image.height <= self._tile_size:
            return [(image, 0, 0, False)]
        stride = max(1, int(self._tile_size * (1.0 - self._tile_overlap)))
        x_positions = tile_positions(image.width, self._tile_size, stride)
        y_positions = tile_positions(image.height, self._tile_size, stride)
        crops = [(image, 0, 0, False)]
        for top in y_positions:
            for left in x_positions:
                right = min(image.width, left + self._tile_size)
                bottom = min(image.height, top + self._tile_size)
                crops.append((image.crop((left, top, right, bottom)), left, top, True))
        return crops

    def _yolo_detections(self, document: AnnotationDocument) -> list[Annotation]:
        device = 0 if detect_gpu().device == "cuda" else "cpu"
        result = self._yolo_model(
            str(document.image_path),
            device=device,
            conf=self._confidence,
            verbose=False,
        )[0]
        names = self._yolo_model.names
        detections = []
        for box in result.boxes:
            class_name = str(names[int(box.cls[0])])
            if class_name not in {"motorcycle", "car", "bus", "truck"}:
                continue
            left, top, right, bottom = box.xyxy[0].tolist()
            detections.append(
                self._annotation(
                    class_name,
                    left,
                    top,
                    right,
                    bottom,
                    float(box.conf[0]),
                    AnnotationSource.YOLO,
                    document,
                )
            )
        return [item for item in detections if item is not None]

    @staticmethod
    def _annotation(
        class_name: str,
        left: float,
        top: float,
        right: float,
        bottom: float,
        confidence: float,
        source: AnnotationSource,
        document: AnnotationDocument,
    ) -> Annotation | None:
        left = max(0.0, min(left, document.image_width))
        top = max(0.0, min(top, document.image_height))
        right = max(0.0, min(right, document.image_width))
        bottom = max(0.0, min(bottom, document.image_height))
        if left >= right or top >= bottom:
            return None
        return Annotation(
            class_name,
            BoundingBox(
                left / document.image_width,
                top / document.image_height,
                right / document.image_width,
                bottom / document.image_height,
            ),
            confidence=confidence,
            source=source,
        )

    @staticmethod
    def _merge_predictions(
        document: AnnotationDocument,
        predictions: list[Annotation],
        same_class_only: bool = True,
    ) -> list[Annotation]:
        """Preserve manual boxes and reject predictions already covered by them."""
        additions = []
        for prediction in predictions:
            if any(
                (not same_class_only or existing.class_name == prediction.class_name)
                and MainWindow._box_iou(existing.box, prediction.box) >= 0.5
                for existing in document.annotations
            ):
                continue
            additions.append(prediction)
        return additions


class MainWindow(QMainWindow):
    """Primary shell; feature views can be added without changing application startup."""

    def __init__(
        self,
        fusion_config: FusionConfig | None = None,
        active_learning_config: ActiveLearningConfig | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Traffic Annotator")
        self.resize(1440, 900)
        self._build_central_view()
        QApplication.instance().installEventFilter(self)
        self._yolo_model = None
        self._yolo_model_path: Path | None = None
        self._grounding_processor = None
        self._grounding_model = None
        self._grounding_model_id = "IDEA-Research/grounding-dino-tiny"
        self._sam2_processor = None
        self._sam2_model = None
        self._sam2_model_id = "facebook/sam2.1-hiera-tiny"
        self._vlm_helper = None
        self._vlm_model_id = "microsoft/Florence-2-base"
        self._vlm_filter_enabled = True
        self._confidence_threshold = 0.25
        self._grounding_prompt = "motorcycle. motorbike. scooter. car. bus. truck."
        self._grounding_detections: list[ModelDetection] = []
        self._yolo_detections: list[ModelDetection] = []
        self._fusion_result: FusionResult | None = None
        self._fusion_config = fusion_config or FusionConfig()
        self._active_learning_engine = ActiveLearningEngine(active_learning_config)
        self._active_learning_result: DifficultyResult | None = None
        self._active_learning_task: _ActiveLearningTask | None = None
        self._dataset_annotation_task: _DatasetAnnotationTask | None = None
        self._dataset_progress: _DatasetProgressDialog | None = None
        self._project_documents: dict[Path, AnnotationDocument] = {}
        self._project_root: Path | None = None
        self._crop_session: CropSession | None = None
        self._crop_original_document: AnnotationDocument | None = None
        self._crop_original_history: AnnotationHistory | None = None
        self._crop_index = 0
        self._crop_directory: Path | None = None
        self._enabled_classes = {"motorcycle", "car", "bus", "truck"}
        self._build_docks()
        self._document: AnnotationDocument | None = None
        self._history: AnnotationHistory | None = None
        self._selected_class = "car"
        self._selected_annotation_id = None
        self.setStatusBar(QStatusBar(self))
        gpu = detect_gpu()
        self.statusBar().showMessage(f"Ready | Device: {gpu.device} ({gpu.name})")
        LOGGER.info("main window initialized")

    def _build_central_view(self) -> None:
        self.canvas = AnnotationCanvas()
        self.setCentralWidget(self.canvas)

    @staticmethod
    def _create_class_icon(color_hex: str, size: int = 14) -> QIcon:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(color_hex)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(1, 1, size - 2, size - 2)
        painter.end()
        return QIcon(pixmap)

    def _build_docks(self) -> None:
        class_list = QTreeWidget()
        class_list.setObjectName("classList")
        class_list.setHeaderHidden(True)
        class_list.setIndentation(0)
        class_list.setUniformRowHeights(True)
        class_list.setIconSize(QSize(14, 14))

        for name in ("motorcycle", "car", "bus", "truck"):
            color = AnnotationCanvas.CLASS_COLORS.get(name, "#29b6f6")
            item = QTreeWidgetItem(class_list, [name])
            item.setIcon(0, self._create_class_icon(color))
            item.setData(0, Qt.ItemDataRole.UserRole, name)

        class_list.itemSelectionChanged.connect(self._class_changed)
        class_list.setCurrentItem(class_list.topLevelItem(0))
        classes_dock = self._dock("Classes", class_list)
        classes_dock.setMinimumWidth(250)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, classes_dock)
        self.image_browser = ImageBrowser()
        images_dock = self._dock("Images", self.image_browser)
        images_dock.setMinimumWidth(250)
        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, images_dock
        )
        properties = QWidget()
        self._properties_layout = QVBoxLayout(properties)
        self._properties_layout.setContentsMargins(6, 6, 6, 6)
        self._properties_layout.setSpacing(8)
        self._property_group_layouts: dict[str, QVBoxLayout] = {}
        properties_scroll = QScrollArea()
        properties_scroll.setWidgetResizable(True)
        properties_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        properties_scroll.setWidget(properties)
        properties_dock = self._dock("Properties", properties_scroll)
        properties_dock.setMinimumWidth(320)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            properties_dock,
        )
        self.resizeDocks([classes_dock, properties_dock], [250, 320], Qt.Orientation.Horizontal)
        self.resizeDocks([classes_dock, images_dock], [200, 450], Qt.Orientation.Vertical)
        self._build_shortcuts()
        self.image_browser.image_selected.connect(self._load_image)
        self.image_browser.delete_requested.connect(self._delete_picture_from_database)
        self.canvas.box_created.connect(self._add_box)
        self.canvas.box_selected.connect(self._select_annotation)
        self.canvas.box_resized.connect(self._resize_box)
        self.canvas.box_deleted.connect(self._delete_box)
        self._build_file_actions()

    def _build_file_actions(self) -> None:
        import_folder = QAction("Import Folder", self)
        import_folder.triggered.connect(self._import_folder)
        import_coco = QAction("Import COCO Dataset", self)
        import_coco.triggered.connect(self._import_coco_dataset)
        export_dataset = QAction("Export Dataset", self)
        export_dataset.triggered.connect(self._export_dataset)
        crop_start = QAction("Start Crop Assist", self)
        crop_start.triggered.connect(self._start_crop_assist)
        crop_previous = QAction("Previous Crop", self)
        crop_previous.setShortcut("Alt+Left")
        crop_previous.triggered.connect(self._previous_crop)
        crop_next = QAction("Next Crop", self)
        crop_next.setShortcut("Alt+Right")
        crop_next.triggered.connect(self._next_crop)
        crop_commit = QAction("Commit Crop Session", self)
        crop_commit.triggered.connect(self._commit_crop_assist)
        crop_cancel = QAction("Cancel Crop Session", self)
        crop_cancel.triggered.connect(self._cancel_crop_assist)
        load_model = QAction("Load YOLO Model", self)
        load_model.setShortcut("Ctrl+L")
        load_model.triggered.connect(self._load_yolo_model)
        auto_annotate = QAction("YOLO Annotate", self)
        auto_annotate.setShortcut("Ctrl+Shift+A")
        auto_annotate.triggered.connect(self._auto_annotate)
        load_grounding = QAction("Load Grounding DINO", self)
        load_grounding.setShortcut("Ctrl+G")
        load_grounding.triggered.connect(self._load_grounding_model)
        grounding_annotate = QAction("DINO Annotate", self)
        grounding_annotate.setShortcut("Ctrl+Shift+G")
        grounding_annotate.triggered.connect(self._grounding_annotate)
        load_sam2 = QAction("Load SAM2", self)
        load_sam2.setShortcut("Ctrl+Shift+L")
        load_sam2.triggered.connect(self._load_sam2_model)
        load_vlm = QAction("Load Florence-2 VLM", self)
        load_vlm.setShortcut("Ctrl+Shift+U")
        load_vlm.triggered.connect(self._load_vlm_model)
        vlm_annotate = QAction("VLM Auto-Annotate", self)
        vlm_annotate.setShortcut("Ctrl+Shift+V")
        vlm_annotate.setToolTip("Run Florence-2 object detection to generate new annotations on active image (Ctrl+Shift+V)")
        vlm_annotate.triggered.connect(self._vlm_auto_annotate)
        toggle_vlm_filter = QAction("VLM Auto-Filter (DINO+SAM)", self)
        toggle_vlm_filter.setCheckable(True)
        toggle_vlm_filter.setChecked(True)
        toggle_vlm_filter.setToolTip("Filter candidate boxes through Florence-2 captioning before SAM2 segmentation")
        toggle_vlm_filter.toggled.connect(self._toggle_vlm_filtering)
        refine_sam2 = QAction("Refine Selection (SAM2)", self)
        refine_sam2.setShortcut("Ctrl+Shift+S")
        refine_sam2.triggered.connect(self._refine_with_sam2)
        fuse = QAction("Label Fusion", self)
        fuse.setShortcut("Ctrl+Shift+F")
        fuse.triggered.connect(self._run_fusion)
        cleanup = QAction("Remove Overlapping Duplicates", self)
        cleanup.setShortcut("Ctrl+Shift+D")
        cleanup.triggered.connect(self._remove_overlapping)
        cleanup_dataset = QAction("Remove Database Duplicates", self)
        cleanup_dataset.setShortcut("Ctrl+Shift+Alt+D")
        cleanup_dataset.triggered.connect(self._remove_database_duplicates)
        delete_all_annotations = QAction("Delete All Annotations (Current Image)", self)
        delete_all_annotations.setShortcut("Ctrl+Shift+X")
        delete_all_annotations.setToolTip("Delete all annotations on current image (Ctrl+Shift+X)")
        delete_all_annotations.triggered.connect(self._delete_all_annotations)
        delete_picture = QAction("Delete Picture from Database", self)
        delete_picture.setShortcuts(["Ctrl+Delete", "Shift+Delete"])
        delete_picture.setToolTip("Delete current picture from dataset database and disk (Ctrl+Delete)")
        delete_picture.triggered.connect(lambda: self._delete_picture_from_database())
        toggle_occluded = QAction("Toggle Selected Occluded", self)
        toggle_occluded.triggered.connect(self._toggle_selected_occluded)
        toggle_truncated = QAction("Toggle Selected Truncated", self)
        toggle_truncated.triggered.connect(self._toggle_selected_truncated)
        fusion_colors = QAction("Show Fusion Status Colors", self)
        fusion_colors.setCheckable(True)
        fusion_colors.setChecked(True)
        fusion_colors.toggled.connect(self.canvas.set_fusion_colors_enabled)
        active_learning = QAction("Score Review Difficulty", self)
        active_learning.setShortcut("Ctrl+Shift+R")
        active_learning.triggered.connect(self._score_active_image)
        dataset_annotate = QAction("Annotate Entire Dataset", self)
        dataset_annotate.triggered.connect(self._annotate_entire_dataset)
        dino_dataset_annotate = QAction("DINO Annotate Entire Dataset", self)
        dino_dataset_annotate.triggered.connect(self._annotate_entire_dataset_dino)
        dino_sam_annotate = QAction("DINO + SAM Auto-Annotate", self)
        dino_sam_annotate.setShortcut("Ctrl+Shift+Z")
        dino_sam_annotate.setToolTip("Run Grounding DINO detection and refine with SAM2 in one step (Ctrl+Shift+Z)")
        dino_sam_annotate.triggered.connect(self._zero_shot_dino_sam_annotate)
        dino_sam_dataset_annotate = QAction("DINO + SAM Annotate Entire Dataset", self)
        dino_sam_dataset_annotate.setToolTip("Run Zero-Shot Grounding DINO + SAM2 auto-annotation across the entire dataset")
        dino_sam_dataset_annotate.triggered.connect(self._annotate_entire_dataset_dino_sam)
        auto_label_workspace = QAction("⚡ Auto Label...", self)
        auto_label_workspace.setShortcut("Ctrl+Shift+A")
        auto_label_workspace.setToolTip("Open Roboflow-style Auto Label workspace with DINO, SAM2, VLM, custom visual prompts & live preview (Ctrl+Shift+A)")
        auto_label_workspace.triggered.connect(self._open_auto_label_dialog)
        self._filter_actions = {}
        for label, status in (
            ("Show Accepted", FusionStatus.ACCEPTED),
            ("Show Needs Review", FusionStatus.NEEDS_REVIEW),
            ("Show Conflicts", FusionStatus.CONFLICT),
        ):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(True)
            action.toggled.connect(self._update_fusion_filter)
            self._filter_actions[status] = action
        save = QAction("Save", self)
        save.setShortcut("Ctrl+S")
        save.triggered.connect(self._save_annotations)

        draw_tool = QAction("Draw Box Tool", self)
        draw_tool.setCheckable(True)
        draw_tool.setChecked(True)
        draw_tool.setShortcut("V")
        draw_tool.setToolTip("Select and draw bounding boxes (V)")
        draw_tool.triggered.connect(self.canvas.set_draw_mode)

        pan_tool = QAction("Pan Tool", self)
        pan_tool.setCheckable(True)
        pan_tool.setShortcut("H")
        pan_tool.setToolTip("Pan / Hand Tool (H) - Drag canvas to pan (Or Space+drag / Middle-click drag)")
        pan_tool.triggered.connect(self.canvas.set_pan_mode)

        tool_group = QActionGroup(self)
        tool_group.addAction(draw_tool)
        tool_group.addAction(pan_tool)
        tool_group.setExclusive(True)

        def _sync_canvas_mode(mode: CanvasMode) -> None:
            if mode == CanvasMode.DRAW:
                draw_tool.setChecked(True)
                if hasattr(self, "_draw_tool_btn"):
                    self._draw_tool_btn.setChecked(True)
            elif mode == CanvasMode.PAN:
                pan_tool.setChecked(True)
                if hasattr(self, "_pan_tool_btn"):
                    self._pan_tool_btn.setChecked(True)

        self.canvas.mode_changed.connect(_sync_canvas_mode)

        fit_view = QAction("Fit to View", self)
        fit_view.setShortcut("F")
        fit_view.setToolTip("Fit image to canvas view (F)")
        fit_view.triggered.connect(self.canvas.reset_view)

        zoom_in = QAction("Zoom In", self)
        zoom_in.setShortcuts(["Ctrl++", "Ctrl+=", "+", "="])
        zoom_in.setToolTip("Zoom in (+ / Ctrl++)")
        zoom_in.triggered.connect(lambda: self.canvas.zoom_in())

        zoom_out = QAction("Zoom Out", self)
        zoom_out.setShortcuts(["Ctrl+-", "-"])
        zoom_out.setToolTip("Zoom out (- / Ctrl+-)")
        zoom_out.triggered.connect(lambda: self.canvas.zoom_out())

        zoom_actual = QAction("Actual Size (100%)", self)
        zoom_actual.setShortcut("Ctrl+1")
        zoom_actual.setToolTip("Zoom to 100% scale (Ctrl+1)")
        zoom_actual.triggered.connect(self.canvas.zoom_actual_size)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(import_folder)
        file_menu.addAction(import_coco)
        file_menu.addAction(export_dataset)
        file_menu.addAction(save)
        file_menu.addSeparator()
        file_menu.addAction(delete_picture)
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(draw_tool)
        view_menu.addAction(pan_tool)
        view_menu.addSeparator()
        view_menu.addAction(fit_view)
        view_menu.addAction(zoom_in)
        view_menu.addAction(zoom_out)
        view_menu.addAction(zoom_actual)
        view_menu.addSeparator()
        view_menu.addAction(fusion_colors)
        filter_menu = view_menu.addMenu("Fusion Filters")
        for action in self._filter_actions.values():
            filter_menu.addAction(action)
        model_menu = self.menuBar().addMenu("Model")
        model_menu.addAction(load_model)
        model_menu.addAction(load_grounding)
        model_menu.addAction(load_sam2)
        model_menu.addAction(load_vlm)
        annotation_menu = self.menuBar().addMenu("Annotation")
        annotation_menu.addAction(auto_label_workspace)
        annotation_menu.addSeparator()
        annotation_menu.addAction(auto_annotate)
        annotation_menu.addAction(grounding_annotate)
        annotation_menu.addAction(refine_sam2)
        annotation_menu.addAction(dino_sam_annotate)
        annotation_menu.addAction(vlm_annotate)
        annotation_menu.addAction(toggle_vlm_filter)
        annotation_menu.addAction(dataset_annotate)
        annotation_menu.addAction(dino_dataset_annotate)
        annotation_menu.addAction(dino_sam_dataset_annotate)
        annotation_menu.addAction(fuse)
        annotation_menu.addAction(cleanup)
        annotation_menu.addAction(cleanup_dataset)
        annotation_menu.addAction(delete_all_annotations)
        annotation_menu.addAction(delete_picture)
        annotation_menu.addAction(toggle_occluded)
        annotation_menu.addAction(toggle_truncated)
        annotation_menu.addAction(fusion_colors)
        annotation_menu.addAction(active_learning)
        crop_menu = annotation_menu.addMenu("Crop Assist")
        for action in (crop_start, crop_previous, crop_next, crop_commit, crop_cancel):
            crop_menu.addAction(action)

        self._setup_properties_panel(
            draw_tool=draw_tool,
            pan_tool=pan_tool,
            auto_label_workspace=auto_label_workspace,
            fit_view=fit_view,
            zoom_in=zoom_in,
            zoom_out=zoom_out,
            zoom_actual=zoom_actual,
            refine_sam2=refine_sam2,
            fuse=fuse,
            cleanup=cleanup,
            cleanup_dataset=cleanup_dataset,
            toggle_occluded=toggle_occluded,
            toggle_truncated=toggle_truncated,
            fusion_colors=fusion_colors,
            active_learning=active_learning,
            crop_start=crop_start,
            crop_previous=crop_previous,
            crop_next=crop_next,
            crop_commit=crop_commit,
            crop_cancel=crop_cancel,
            import_folder=import_folder,
            import_coco=import_coco,
            save=save,
            export_dataset=export_dataset,
        )

    def _setup_properties_panel(
        self,
        draw_tool: QAction,
        pan_tool: QAction,
        auto_label_workspace: QAction,
        fit_view: QAction,
        zoom_in: QAction,
        zoom_out: QAction,
        zoom_actual: QAction,
        refine_sam2: QAction,
        fuse: QAction,
        cleanup: QAction,
        cleanup_dataset: QAction,
        toggle_occluded: QAction,
        toggle_truncated: QAction,
        fusion_colors: QAction,
        active_learning: QAction,
        crop_start: QAction,
        crop_previous: QAction,
        crop_next: QAction,
        crop_commit: QAction,
        crop_cancel: QAction,
        import_folder: QAction,
        import_coco: QAction,
        save: QAction,
        export_dataset: QAction,
    ) -> None:
        """Construct a clean, consolidated, grouped Properties panel."""
        # 1. Canvas Tools & View
        self._tools_group = QGroupBox("Canvas Tools")
        tools_layout = QVBoxLayout(self._tools_group)
        tools_layout.setContentsMargins(8, 10, 8, 8)
        tools_layout.setSpacing(6)

        # Mode row: Draw & Pan (50% / 50% equal width)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self._draw_tool_btn = QToolButton(self)
        self._draw_tool_btn.setText("✏️ Draw (V)")
        self._draw_tool_btn.setCheckable(True)
        self._draw_tool_btn.setChecked(True)
        self._draw_tool_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        self._draw_tool_btn.setToolTip("Select and draw bounding boxes (V)")
        self._draw_tool_btn.clicked.connect(self.canvas.set_draw_mode)
        mode_row.addWidget(self._draw_tool_btn, 1)

        self._pan_tool_btn = QToolButton(self)
        self._pan_tool_btn.setText("✋ Pan (H)")
        self._pan_tool_btn.setCheckable(True)
        self._pan_tool_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        self._pan_tool_btn.setToolTip("Pan / Hand Tool (H) - Drag canvas to pan")
        self._pan_tool_btn.clicked.connect(self.canvas.set_pan_mode)
        mode_row.addWidget(self._pan_tool_btn, 1)
        tools_layout.addLayout(mode_row)

        canvas_mode_group = QButtonGroup(self)
        canvas_mode_group.addButton(self._draw_tool_btn)
        canvas_mode_group.addButton(self._pan_tool_btn)
        canvas_mode_group.setExclusive(True)

        draw_tool.toggled.connect(self._draw_tool_btn.setChecked)
        pan_tool.toggled.connect(self._pan_tool_btn.setChecked)

        # Auto Label button
        auto_label_btn = QToolButton(self)
        auto_label_btn.setText("⚡ Auto Label Workspace")
        auto_label_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        auto_label_btn.setToolTip(
            "Open Roboflow-style Auto Label workspace with DINO, SAM2, VLM & live preview (Ctrl+Shift+A)"
        )
        auto_label_btn.clicked.connect(self._open_auto_label_dialog)
        tools_layout.addWidget(auto_label_btn)

        # Zoom / View row (4 strictly equal 25% symmetrical buttons)
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(4)

        fit_btn = QToolButton(self)
        fit_btn.setText("Fit")
        fit_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        fit_btn.setToolTip("Fit image to canvas view (F)")
        fit_btn.clicked.connect(self.canvas.reset_view)
        zoom_row.addWidget(fit_btn, 1)

        zoom_in_btn = QToolButton(self)
        zoom_in_btn.setText("+")
        zoom_in_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        zoom_in_btn.setToolTip("Zoom in (+ / Ctrl++)")
        zoom_in_btn.clicked.connect(lambda: self.canvas.zoom_in())
        zoom_row.addWidget(zoom_in_btn, 1)

        zoom_out_btn = QToolButton(self)
        zoom_out_btn.setText("-")
        zoom_out_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        zoom_out_btn.setToolTip("Zoom out (- / Ctrl+-)")
        zoom_out_btn.clicked.connect(lambda: self.canvas.zoom_out())
        zoom_row.addWidget(zoom_out_btn, 1)

        zoom_actual_btn = QToolButton(self)
        zoom_actual_btn.setText("1:1")
        zoom_actual_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        zoom_actual_btn.setToolTip("Zoom to 100% scale (Ctrl+1)")
        zoom_actual_btn.clicked.connect(self.canvas.zoom_actual_size)
        zoom_row.addWidget(zoom_actual_btn, 1)

        tools_layout.addLayout(zoom_row)
        self._properties_layout.addWidget(self._tools_group)

        # 2. Selected Annotation (Properties)
        self._selection_group = QGroupBox("Selected Annotation")
        selection_layout = QVBoxLayout(self._selection_group)
        selection_layout.setContentsMargins(8, 10, 8, 8)
        selection_layout.setSpacing(6)

        self._selection_info_label = QLabel("No annotation selected")
        self._selection_info_label.setStyleSheet(
            "color: #9aa0a6; font-size: 12px; font-weight: normal; padding: 2px;"
        )
        selection_layout.addWidget(self._selection_info_label)

        flags_row = QHBoxLayout()
        flags_row.setSpacing(6)

        self._occluded_btn = QToolButton(self)
        self._occluded_btn.setText("👁 Occluded")
        self._occluded_btn.setCheckable(True)
        self._occluded_btn.setEnabled(False)
        self._occluded_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        self._occluded_btn.setToolTip("Toggle occluded flag on selected annotation")
        self._occluded_btn.clicked.connect(self._toggle_selected_occluded)
        flags_row.addWidget(self._occluded_btn, 1)

        self._truncated_btn = QToolButton(self)
        self._truncated_btn.setText("✂ Truncated")
        self._truncated_btn.setCheckable(True)
        self._truncated_btn.setEnabled(False)
        self._truncated_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        self._truncated_btn.setToolTip("Toggle truncated flag on selected annotation")
        self._truncated_btn.clicked.connect(self._toggle_selected_truncated)
        flags_row.addWidget(self._truncated_btn, 1)
        selection_layout.addLayout(flags_row)

        self._refine_sam2_btn = QToolButton(self)
        self._refine_sam2_btn.setText("🎯 Refine Selection (SAM2)")
        self._refine_sam2_btn.setEnabled(False)
        self._refine_sam2_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        self._refine_sam2_btn.setToolTip(
            "Refine selected bounding box with SAM2 segmentation (Ctrl+Shift+S)"
        )
        self._refine_sam2_btn.clicked.connect(self._refine_with_sam2)
        selection_layout.addWidget(self._refine_sam2_btn)

        self._properties_layout.addWidget(self._selection_group)

        # 3. Review & Cleanup (Grouped 2-column tools)
        self._review_group = QGroupBox("Review && Cleanup")
        review_layout = QVBoxLayout(self._review_group)
        review_layout.setContentsMargins(8, 10, 8, 8)
        review_layout.setSpacing(6)

        ai_row = QHBoxLayout()
        ai_row.setSpacing(6)
        fuse_btn = QToolButton(self)
        fuse_btn.setText("⚡ Label Fusion")
        fuse_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        fuse_btn.setToolTip(
            "Run label fusion algorithm to merge model predictions (Ctrl+Shift+F)"
        )
        fuse_btn.clicked.connect(self._run_fusion)
        ai_row.addWidget(fuse_btn, 1)

        score_btn = QToolButton(self)
        score_btn.setText("📊 Difficulty")
        score_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        score_btn.setToolTip(
            "Score active image review difficulty for active learning (Ctrl+Shift+R)"
        )
        score_btn.clicked.connect(self._score_active_image)
        ai_row.addWidget(score_btn, 1)
        review_layout.addLayout(ai_row)

        clean_row = QHBoxLayout()
        clean_row.setSpacing(6)
        clean_overlap_btn = QToolButton(self)
        clean_overlap_btn.setText("🗂 Remove Overlaps")
        clean_overlap_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        clean_overlap_btn.setToolTip(
            "Remove overlapping duplicate boxes on current image (Ctrl+Shift+D)"
        )
        clean_overlap_btn.clicked.connect(self._remove_overlapping)
        clean_row.addWidget(clean_overlap_btn, 1)

        clean_db_btn = QToolButton(self)
        clean_db_btn.setText("🗄 Clean DB Dups")
        clean_db_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        clean_db_btn.setToolTip(
            "Remove database duplicate boxes across dataset (Ctrl+Shift+Alt+D)"
        )
        clean_db_btn.clicked.connect(self._remove_database_duplicates)
        clean_row.addWidget(clean_db_btn, 1)
        review_layout.addLayout(clean_row)

        delete_btn_row = QHBoxLayout()
        delete_btn_row.setSpacing(6)
        delete_all_btn = QToolButton(self)
        delete_all_btn.setText("🗑 Clear Annotations")
        delete_all_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        delete_all_btn.setToolTip(
            "Delete all bounding box annotations on the current image only (Ctrl+Shift+X)"
        )
        delete_all_btn.clicked.connect(self._delete_all_annotations)
        delete_btn_row.addWidget(delete_all_btn, 1)

        delete_pic_btn = QToolButton(self)
        delete_pic_btn.setText("🗑 Delete Picture")
        delete_pic_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        delete_pic_btn.setToolTip(
            "Delete active picture and its annotations from dataset database (Ctrl+Delete)"
        )
        delete_pic_btn.clicked.connect(lambda: self._delete_picture_from_database())
        delete_btn_row.addWidget(delete_pic_btn, 1)
        review_layout.addLayout(delete_btn_row)

        self._fusion_colors_btn = QToolButton(self)
        self._fusion_colors_btn.setText("🎨 Show Fusion Colors")
        self._fusion_colors_btn.setCheckable(True)
        self._fusion_colors_btn.setChecked(fusion_colors.isChecked())
        self._fusion_colors_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        self._fusion_colors_btn.setToolTip("Toggle fusion status overlay colors on canvas")
        self._fusion_colors_btn.toggled.connect(self.canvas.set_fusion_colors_enabled)
        fusion_colors.toggled.connect(self._fusion_colors_btn.setChecked)
        self._fusion_colors_btn.toggled.connect(fusion_colors.setChecked)
        review_layout.addWidget(self._fusion_colors_btn)

        self._properties_layout.addWidget(self._review_group)

        # 4. Crop Assist
        self._crop_group = QGroupBox("Crop Assist")
        crop_layout = QVBoxLayout(self._crop_group)
        crop_layout.setContentsMargins(8, 10, 8, 8)
        crop_layout.setSpacing(6)

        crop_start_btn = QToolButton(self)
        crop_start_btn.setText("✂ Start Crop Assist")
        crop_start_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        crop_start_btn.setToolTip("Start crop-assisted dense annotation session")
        crop_start_btn.clicked.connect(self._start_crop_assist)
        crop_layout.addWidget(crop_start_btn)

        crop_nav_row = QHBoxLayout()
        crop_nav_row.setSpacing(6)
        crop_prev_btn = QToolButton(self)
        crop_prev_btn.setText("◀ Prev")
        crop_prev_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        crop_prev_btn.setToolTip("Previous Crop (Alt+Left)")
        crop_prev_btn.clicked.connect(self._previous_crop)
        crop_nav_row.addWidget(crop_prev_btn, 1)

        crop_next_btn = QToolButton(self)
        crop_next_btn.setText("Next ▶")
        crop_next_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        crop_next_btn.setToolTip("Next Crop (Alt+Right)")
        crop_next_btn.clicked.connect(self._next_crop)
        crop_nav_row.addWidget(crop_next_btn, 1)
        crop_layout.addLayout(crop_nav_row)

        crop_act_row = QHBoxLayout()
        crop_act_row.setSpacing(6)
        crop_commit_btn = QToolButton(self)
        crop_commit_btn.setText("✓ Commit")
        crop_commit_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        crop_commit_btn.setToolTip("Commit Crop Session")
        crop_commit_btn.clicked.connect(self._commit_crop_assist)
        crop_act_row.addWidget(crop_commit_btn, 1)

        crop_cancel_btn = QToolButton(self)
        crop_cancel_btn.setText("✕ Cancel")
        crop_cancel_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        crop_cancel_btn.setToolTip("Cancel Crop Session")
        crop_cancel_btn.clicked.connect(self._cancel_crop_assist)
        crop_act_row.addWidget(crop_cancel_btn, 1)
        crop_layout.addLayout(crop_act_row)

        self._properties_layout.addWidget(self._crop_group)

        # 5. Project & Dataset
        self._project_group = QGroupBox("Project && Dataset")
        project_layout = QVBoxLayout(self._project_group)
        project_layout.setContentsMargins(8, 10, 8, 8)
        project_layout.setSpacing(6)

        import_row = QHBoxLayout()
        import_row.setSpacing(6)
        import_folder_btn = QToolButton(self)
        import_folder_btn.setText("📁 Import Folder")
        import_folder_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        import_folder_btn.setToolTip("Import folder containing image files")
        import_folder_btn.clicked.connect(self._import_folder)
        import_row.addWidget(import_folder_btn, 1)

        import_coco_btn = QToolButton(self)
        import_coco_btn.setText("📦 Import COCO")
        import_coco_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        import_coco_btn.setToolTip("Import COCO dataset annotations JSON")
        import_coco_btn.clicked.connect(self._import_coco_dataset)
        import_row.addWidget(import_coco_btn, 1)
        project_layout.addLayout(import_row)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(6)
        save_btn = QToolButton(self)
        save_btn.setText("💾 Save")
        save_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        save_btn.setToolTip("Save Annotations (Ctrl+S)")
        save_btn.clicked.connect(self._save_annotations)
        actions_row.addWidget(save_btn, 1)

        export_btn = QToolButton(self)
        export_btn.setText("⤓ Export")
        export_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Fixed)
        export_btn.setToolTip("Export Dataset")
        export_btn.clicked.connect(self._export_dataset)
        actions_row.addWidget(export_btn, 1)
        project_layout.addLayout(actions_row)

        self._properties_layout.addWidget(self._project_group)
        self._properties_layout.addStretch()

        # Backward compatibility for _property_group_layouts
        self._property_group_layouts["Review && Cleanup"] = review_layout
        self._property_group_layouts["Crop Assist"] = crop_layout
        self._property_group_layouts["Project"] = project_layout

    def _update_selection_properties(self) -> None:
        """Synchronize the Selected Annotation section in the Properties panel with current selection."""
        if not hasattr(self, "_selection_info_label"):
            return
        if self._document is None or self._selected_annotation_id is None:
            self._selection_info_label.setText("No annotation selected")
            if hasattr(self, "_occluded_btn"):
                self._occluded_btn.blockSignals(True)
                self._occluded_btn.setChecked(False)
                self._occluded_btn.setEnabled(False)
                self._occluded_btn.blockSignals(False)
            if hasattr(self, "_truncated_btn"):
                self._truncated_btn.blockSignals(True)
                self._truncated_btn.setChecked(False)
                self._truncated_btn.setEnabled(False)
                self._truncated_btn.blockSignals(False)
            if hasattr(self, "_refine_sam2_btn"):
                self._refine_sam2_btn.setEnabled(False)
            return

        annotation = next(
            (
                item
                for item in self._document.annotations
                if item.annotation_id == self._selected_annotation_id
            ),
            None,
        )
        if annotation is None:
            self._selection_info_label.setText("No annotation selected")
            self._selection_info_label.setStyleSheet(
                "color: #8b90a0; font-size: 12px; font-weight: normal; padding: 2px;"
            )
            if hasattr(self, "_occluded_btn"):
                self._occluded_btn.blockSignals(True)
                self._occluded_btn.setChecked(False)
                self._occluded_btn.setEnabled(False)
                self._occluded_btn.blockSignals(False)
            if hasattr(self, "_truncated_btn"):
                self._truncated_btn.blockSignals(True)
                self._truncated_btn.setChecked(False)
                self._truncated_btn.setEnabled(False)
                self._truncated_btn.blockSignals(False)
            if hasattr(self, "_refine_sam2_btn"):
                self._refine_sam2_btn.setEnabled(False)
            return

        ann_id_short = (
            str(annotation.annotation_id)[:8]
            if annotation.annotation_id
            else ""
        )
        color = AnnotationCanvas.CLASS_COLORS.get(annotation.class_name, "#4fc3f7")
        self._selection_info_label.setText(
            f"● {annotation.class_name.upper()} (ID: {ann_id_short})"
        )
        self._selection_info_label.setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: 600; padding: 2px;"
        )
        if hasattr(self, "_occluded_btn"):
            self._occluded_btn.blockSignals(True)
            self._occluded_btn.setEnabled(True)
            self._occluded_btn.setChecked(bool(annotation.occluded))
            self._occluded_btn.blockSignals(False)
        if hasattr(self, "_truncated_btn"):
            self._truncated_btn.blockSignals(True)
            self._truncated_btn.setEnabled(True)
            self._truncated_btn.setChecked(bool(annotation.truncated))
            self._truncated_btn.blockSignals(False)
        if hasattr(self, "_refine_sam2_btn"):
            self._refine_sam2_btn.setEnabled(True)

    def _add_property_action(self, group: str, action: QAction) -> None:
        """Add an action to its grouped tool section in the Properties dock."""
        if hasattr(self, "_property_group_layouts") and group in self._property_group_layouts:
            button = QToolButton(self)
            button.setDefaultAction(action)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            self._property_group_layouts[group].addWidget(button)

    def _set_confidence_threshold(self, value: float) -> None:
        self._confidence_threshold = value

    def _set_enabled_classes(self, value: str) -> None:
        supported = {"motorcycle", "car", "bus", "truck"}
        selected = {
            item.strip().lower()
            for item in value.split(",")
            if item.strip().lower() in supported
        }
        if selected:
            self._enabled_classes = selected

    def _set_grounding_prompt(self, value: str) -> None:
        self._grounding_prompt = value

    def _update_fusion_filter(self) -> None:
        """Apply checked fusion status filters to the canvas."""
        if not hasattr(self, "_filter_actions"):
            return
        statuses = {
            status for status, action in self._filter_actions.items() if action.isChecked()
        }
        self.canvas.set_status_filter(statuses if self._fusion_result is not None else None)

    def _select_annotation(self, annotation_id) -> None:  # type: ignore[no-untyped-def]
        self._selected_annotation_id = annotation_id
        self._update_selection_properties()

    def _class_changed(self) -> None:
        selected = self.sender().currentItem()  # type: ignore[union-attr]
        if selected is not None:
            self._selected_class = selected.data(0, Qt.ItemDataRole.UserRole) or selected.text(0)

    def _import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Import image folder")
        if not folder:
            return
        paths = sorted(
            path for path in Path(folder).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        self._project_documents = {
            path: AnnotationDocument(path, 1920, 1080) for path in paths
        }
        self._project_root = None
        self._refresh_image_browser_order(preserve_current=False)
        if paths:
            self._load_image(paths[0])
        self.statusBar().showMessage(f"Imported {len(paths)} images")

    def _import_coco_dataset(self) -> None:
        """Import a COCO detection dataset into a copied cleaning project."""
        annotation_name, _ = QFileDialog.getOpenFileName(
            self, "Choose COCO annotations", "", "COCO JSON (*.json)"
        )
        if not annotation_name:
            return
        annotation_path = Path(annotation_name)
        image_root_name = QFileDialog.getExistingDirectory(
            self, "Choose COCO image directory", str(annotation_path.parent)
        )
        if not image_root_name:
            return
        parent_name = QFileDialog.getExistingDirectory(self, "Choose new project destination")
        if not parent_name:
            return
        destination = self._new_project_path(Path(parent_name), annotation_path.stem)
        try:
            result = CocoImporter().import_dataset(
                annotation_path,
                Path(image_root_name),
                destination,
                remove_overlaps=True,
                overlap_iou_threshold=self._fusion_config.overlap_removal_iou_threshold,
                containment_threshold=self._fusion_config.overlap_removal_containment_threshold,
            )
        except Exception as error:
            LOGGER.exception("COCO import failed")
            QMessageBox.critical(self, "COCO import failed", str(error))
            return
        self._set_imported_project(result)
        report = result.report
        QMessageBox.information(
            self,
            "COCO import completed",
            f"Images copied: {report.images_imported}/{report.images_found}\n"
            f"Annotations imported: {report.annotations_imported}\n"
            f"Overlaps removed: {report.overlapping_removed}\n"
            f"Unsupported categories skipped: {report.unsupported_categories}\n"
            f"Missing images: {report.missing_images}\n"
            f"Invalid records: {report.invalid_annotations}",
        )

    def _export_dataset(self) -> None:
        """Export the current project in a user-selected dataset format."""
        if self._crop_session is not None:
            self.statusBar().showMessage("Commit or cancel Crop Assist before exporting")
            return
        if not self._project_documents:
            self.statusBar().showMessage("Import a folder or dataset before exporting")
            return
        formats = ["COCO Detection", "YOLOv8 Detection", "YOLOv11 Detection", "YOLOv26 Detection"]
        selected, accepted = QInputDialog.getItem(
            self,
            "Export Dataset",
            "Choose export format:",
            formats,
            0,
            False,
        )
        if not accepted:
            return
        split_choice, accepted = QInputDialog.getItem(
            self,
            "Export Split",
            "Dataset layout:",
            ["All images", "Train / validation / test split"],
            0,
            False,
        )
        if not accepted:
            return
        slug = {
            "COCO Detection": "coco",
            "YOLOv8 Detection": "yolov8",
            "YOLOv11 Detection": "yolov11",
            "YOLOv26 Detection": "yolov26",
        }[selected]
        parent_name = QFileDialog.getExistingDirectory(
            self,
            f"Choose {selected} destination",
        )
        if not parent_name:
            return
        documents = list(self._project_documents.values())
        splits = None
        suffix = ""
        if split_choice != "All images":
            ratio_text, accepted = QInputDialog.getText(
                self,
                "Split Ratios",
                "Train, validation, test ratios:",
                QLineEdit.EchoMode.Normal,
                "0.8,0.1,0.1",
            )
            if not accepted:
                return
            try:
                ratios = tuple(float(value.strip()) for value in ratio_text.split(","))
                if len(ratios) != 3:
                    raise ValueError
                seed, accepted = QInputDialog.getInt(
                    self,
                    "Split Seed",
                    "Random seed:",
                    42,
                    0,
                    2**31 - 1,
                )
                if not accepted:
                    return
                splits = split_documents(documents, *ratios, seed=seed)
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Invalid split ratios",
                    "Enter three non-negative ratios that add up to 1.0.",
                )
                return
            suffix = "-split"
        destination = self._new_project_path(Path(parent_name), f"exported-{slug}{suffix}")
        exporter = (
            CocoExporter(splits=splits)
            if slug == "coco"
            else YoloExporter(variant=slug, splits=splits)
        )
        try:
            result = exporter.export(documents, destination)
        except Exception as error:
            LOGGER.exception("dataset export failed")
            QMessageBox.critical(self, "Dataset export failed", str(error))
            return
        self.statusBar().showMessage(f"Exported {selected}: {result}")

    def _set_imported_project(self, result: CocoImportResult) -> None:
        self._project_root = result.project_root
        self._project_documents = {
            document.image_path: document for document in result.documents
        }
        self._refresh_image_browser_order(preserve_current=False)
        if self._project_documents:
            first_path = next(iter(self._project_documents.keys()))
            self._load_image(first_path)

    @staticmethod
    def _new_project_path(parent: Path, stem: str) -> Path:
        """Return a new destination without overwriting an existing project."""
        parent.mkdir(parents=True, exist_ok=True)
        candidate = parent / stem
        suffix = 2
        while candidate.exists():
            candidate = parent / f"{stem}-{suffix}"
            suffix += 1
        return candidate

    def _start_crop_assist(self) -> None:
        """Start a temporary overlapping crop annotation session."""
        if self._document is None or self._history is None:
            self.statusBar().showMessage("Select an image before starting Crop Assist")
            return
        if self._crop_session is not None:
            self.statusBar().showMessage("A Crop Assist session is already active")
            return
        try:
            self._crop_directory = Path(tempfile.mkdtemp(prefix="traffic-annotator-crops-"))
            self._crop_original_document = self._document
            self._crop_original_history = self._history
            self._crop_session = CropGenerator().generate(
                self._document,
                self._crop_directory,
                tile_size=640,
                overlap=0.20,
            )
            self._crop_index = 0
            self._load_crop(0)
        except Exception as error:
            LOGGER.exception("crop session creation failed")
            self._cleanup_crop_session()
            self.statusBar().showMessage(f"Crop Assist failed: {error}")

    def _load_crop(self, index: int) -> None:
        """Switch to a crop document, saving edits from the previous crop."""
        if self._crop_session is None:
            return
        self._save_current_crop()
        if not 0 <= index < len(self._crop_session.documents):
            return
        self._crop_index = index
        document = self._crop_session.documents[index]
        self._document = document
        self._history = AnnotationHistory(document)
        self.canvas.clear_fusion_statuses()
        self.canvas.set_document(document)
        self.statusBar().showMessage(
            f"Crop Assist {index + 1}/{len(self._crop_session.documents)} | "
            f"{document.image_width}x{document.image_height}"
        )

    def _save_current_crop(self) -> None:
        """Store the active crop document in the current session."""
        if self._crop_session is None or self._document is None:
            return
        if self._document.image_path != self._crop_session.regions[self._crop_index].image_path:
            return
        documents = list(self._crop_session.documents)
        documents[self._crop_index] = self._document
        self._crop_session = replace(self._crop_session, documents=tuple(documents))

    def _previous_crop(self) -> None:
        """Open the previous crop in the active session."""
        if self._crop_session is None:
            self.statusBar().showMessage("Start Crop Assist first")
            return
        self._load_crop(max(0, self._crop_index - 1))

    def _next_crop(self) -> None:
        """Open the next crop in the active session."""
        if self._crop_session is None:
            self.statusBar().showMessage("Start Crop Assist first")
            return
        self._load_crop(min(len(self._crop_session.documents) - 1, self._crop_index + 1))

    def _commit_crop_assist(self) -> None:
        """Merge all crop edits back into the original image document."""
        if self._crop_session is None or self._crop_original_document is None:
            self.statusBar().showMessage("Start Crop Assist first")
            return
        self._save_current_crop()
        merged = CropMerger().merge(
            self._crop_original_document,
            self._crop_session.regions,
            self._crop_session.documents,
        )
        if self._crop_original_history is not None and self._crop_original_document is not None:
            self._history = self._crop_original_history
            self._document = self._history.execute(
                ReplaceDocumentCommand(self._crop_original_document, merged)
            )
        else:
            self._document = merged
            self._history = AnnotationHistory(merged)
        self._remember_current_document()
        self._finish_crop_session()
        self.canvas.set_document(merged)
        self.statusBar().showMessage(f"Crop Assist committed {len(merged.annotations)} boxes")

    def _cancel_crop_assist(self) -> None:
        """Discard crop edits and restore the original document state."""
        if self._crop_session is None:
            return
        if self._crop_original_document is not None:
            self._document = self._crop_original_document
        if self._crop_original_history is not None:
            self._history = self._crop_original_history
        self._finish_crop_session()
        if self._document is not None:
            self.canvas.set_document(self._document)
        self.statusBar().showMessage("Crop Assist cancelled")

    def _finish_crop_session(self) -> None:
        """Release temporary crop state and files."""
        self._cleanup_crop_session()
        self._crop_session = None
        self._crop_original_document = None
        self._crop_original_history = None
        self._crop_index = 0

    def _cleanup_crop_session(self) -> None:
        if self._crop_directory is not None:
            shutil.rmtree(self._crop_directory, ignore_errors=True)
            self._crop_directory = None

    def _load_image(self, path: Path) -> None:
        if self._crop_session is not None:
            self._cancel_crop_assist()
        image = QImage(str(path))
        if image.isNull():
            self.statusBar().showMessage(f"Could not load {path.name}")
            return
        self._document = self._project_documents.get(
            path,
            AnnotationDocument(path, image.width(), image.height()),
        )
        self._project_documents[path] = self._document
        self._history = AnnotationHistory(self._document)
        self._grounding_detections = []
        self._yolo_detections = []
        self._fusion_result = None
        self.canvas.clear_fusion_statuses()
        self._selected_annotation_id = None
        self.canvas.set_document(self._document)
        self.statusBar().showMessage(f"Annotating {path.name} | class: {self._selected_class}")
        self._update_selection_properties()

    def _add_box(self, box) -> None:  # type: ignore[no-untyped-def]
        if self._history is None:
            return
        self._document = self._history.execute(
            AddAnnotationCommand(Annotation(self._selected_class, box))
        )
        self._remember_current_document()
        self.canvas.set_document(self._document)
        self.statusBar().showMessage(f"Added {self._selected_class} box")
        self._update_selection_properties()

    def _resize_box(self, annotation_id, box) -> None:  # type: ignore[no-untyped-def]
        if self._history is None:
            return
        previous = next(
            (
                item
                for item in self._history.document.annotations
                if item.annotation_id == annotation_id
            ),
            None,
        )
        if previous is None:
            return
        self._document = self._history.execute(
            UpdateAnnotationCommand(previous, previous.modify(box))
        )
        self._remember_current_document()
        self.canvas.set_document(self._document)
        self.statusBar().showMessage("Resized annotation")

    def _delete_box(self, annotation_id) -> None:  # type: ignore[no-untyped-def]
        if self._history is None:
            return
        annotation = next(
            (
                item
                for item in self._history.document.annotations
                if item.annotation_id == annotation_id
            ),
            None,
        )
        if annotation is None:
            return
        if self._selected_annotation_id == annotation_id:
            self._selected_annotation_id = None
        self._document = self._history.execute(RemoveAnnotationCommand(annotation))
        self._remember_current_document()
        self.canvas.set_document(self._document)
        self.statusBar().showMessage("Deleted annotation")
        self._update_selection_properties()

    def _save_annotations(self) -> None:
        if self._crop_session is not None:
            self.statusBar().showMessage("Commit or cancel Crop Assist before saving")
            return
        if self._document is None:
            self.statusBar().showMessage("Import a folder and select an image first")
            return
        class_order = ("motorcycle", "car", "bus", "truck")
        lines = []
        for annotation in self._document.annotations:
            center_x, center_y, width, height = annotation.box.to_yolo()
            class_id = class_order.index(annotation.class_name)
            lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        label_path = self._document.image_path.with_suffix(".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.statusBar().showMessage(f"Saved {len(lines)} annotations to {label_path.name}")

    def _remember_current_document(self) -> None:
        """Persist the active document in the current imported project session."""
        if self._document is not None and self._document.image_path in self._project_documents:
            self._project_documents[self._document.image_path] = self._document
            self.image_browser.update_annotation_count(
                self._document.image_path, len(self._document.annotations)
            )

    def _refresh_image_browser_order(self, preserve_current: bool = True) -> None:
        """Sort and refresh the Image Browser to keep annotated images at the top."""
        if not self._project_documents:
            return

        current_path = self._document.image_path if self._document else None

        annotated_paths = [
            p
            for p, doc in self._project_documents.items()
            if doc and doc.annotations
        ]
        unannotated_paths = [
            p
            for p, doc in self._project_documents.items()
            if not doc or not doc.annotations
        ]

        # Put annotated images at the top!
        sorted_paths = sorted(annotated_paths) + sorted(unannotated_paths)
        counts = {
            p: len(doc.annotations) for p, doc in self._project_documents.items() if doc
        }

        # Fast path: If the paths in the browser already match sorted_paths,
        # simply update the annotation count badges in-place without rebuilding the widget!
        existing_paths = list(self.image_browser._items_by_path.keys())
        if existing_paths == sorted_paths:
            for p, count in counts.items():
                self.image_browser.update_annotation_count(p, count)
            return

        self.image_browser.set_paths(sorted_paths, annotation_counts=counts)

        self.image_browser.blockSignals(True)
        if preserve_current and current_path in sorted_paths:
            row = sorted_paths.index(current_path)
            self.image_browser.setCurrentRow(row)
        elif sorted_paths:
            self.image_browser.setCurrentRow(0)
        self.image_browser.blockSignals(False)

    def _load_yolo_model(self) -> None:
        """Load YOLO weights once for reuse across images."""
        model_path, _ = QFileDialog.getOpenFileName(
            self, "Choose YOLO model weights", "", "YOLO weights (*.pt *.onnx *.engine);;All Files (*)"
        )
        if not model_path:
            return
        try:
            from ultralytics import YOLO

            self._yolo_model = YOLO(model_path)
            self._yolo_model_path = Path(model_path)
            if not hasattr(self, "_loaded_yolo_models"):
                self._loaded_yolo_models = []
            if str(model_path) not in self._loaded_yolo_models:
                self._loaded_yolo_models.append(str(model_path))
            self.statusBar().showMessage(f"Loaded model: {self._yolo_model_path.name}")
        except Exception as error:
            self._yolo_model = None
            self._yolo_model_path = None
            LOGGER.exception("YOLO model loading failed")
            self.statusBar().showMessage(f"Model loading failed: {error}")

    def _load_grounding_model(self) -> None:
        """Load the Hugging Face Grounding DINO model once for reuse."""
        try:
            import torch
            from transformers import AutoProcessor, GroundingDinoForObjectDetection

            device = "cuda" if detect_gpu().device == "cuda" else "cpu"
            dtype = torch.float32
            self._grounding_processor = AutoProcessor.from_pretrained(self._grounding_model_id)
            self._grounding_model = GroundingDinoForObjectDetection.from_pretrained(
                self._grounding_model_id,
                torch_dtype=dtype,
            ).to(torch.device(device))
            self._grounding_model.eval()
            self.statusBar().showMessage(f"Loaded Grounding DINO: {self._grounding_model_id}")
        except Exception as error:
            self._grounding_processor = None
            self._grounding_model = None
            LOGGER.exception("Grounding DINO loading failed")
            self.statusBar().showMessage(f"Grounding DINO loading failed: {error}")

    def _load_sam2_model(self) -> None:
        """Load the Hugging Face SAM2 model once for box refinement."""
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor

            device = "cuda" if detect_gpu().device == "cuda" else "cpu"
            dtype = torch.float32
            self._sam2_processor = Sam2Processor.from_pretrained(self._sam2_model_id)
            self._sam2_model = Sam2Model.from_pretrained(
                self._sam2_model_id,
                torch_dtype=dtype,
            ).to(torch.device(device))
            self._sam2_model.eval()
            self.statusBar().showMessage(f"Loaded SAM2: {self._sam2_model_id}")
        except Exception as error:
            self._sam2_processor = None
            self._sam2_model = None
            LOGGER.exception("SAM2 loading failed")
            self.statusBar().showMessage(f"SAM2 loading failed: {error}")

    def _load_vlm_model(self) -> None:
        """Load the Hugging Face Florence-2 model once for verification."""
        try:
            from src.vlm_helper import Florence2VLM

            self._vlm_helper = Florence2VLM(model_id=self._vlm_model_id)
            self._vlm_helper.ensure_loaded()
            self.statusBar().showMessage(f"Loaded Florence-2 VLM: {self._vlm_model_id}")
        except Exception as error:
            self._vlm_helper = None
            LOGGER.exception("Florence-2 VLM loading failed")
            self.statusBar().showMessage(f"Florence-2 VLM loading failed: {error}")

    def _toggle_vlm_filtering(self, enabled: bool) -> None:
        """Toggle automatic VLM verification during DINO + SAM auto-annotation."""
        self._vlm_filter_enabled = enabled
        status = "enabled" if enabled else "disabled"
        self.statusBar().showMessage(f"VLM auto-filter {status}")

    def _vlm_auto_annotate(self) -> None:
        """Run Florence-2 object detection to generate new annotations on the active image."""
        if self._document is None or self._history is None:
            self.statusBar().showMessage("Select an image before running VLM auto-annotate")
            return

        if self._vlm_helper is None:
            self._load_vlm_model()
        if self._vlm_helper is None:
            return

        from app.services.annotation.domain import Annotation, AnnotationSource
        from app.services.annotation.history import AddAnnotationCommand
        from src.vlm_helper import generate_annotations

        try:
            from PIL import Image

            image = Image.open(self._document.image_path).convert("RGB")

            candidates = generate_annotations(
                image=image,
                image_width=image.width,
                image_height=image.height,
                vlm=self._vlm_helper,
                enabled_classes=self._enabled_classes,
            )

            added = 0
            for class_name, bbox in candidates:
                # Skip if overlapping with existing annotation of the same class
                if any(
                    existing.class_name == class_name
                    and self._box_iou(existing.box, bbox) >= 0.5
                    for existing in self._document.annotations
                ):
                    continue

                annotation = Annotation(
                    class_name=class_name,
                    box=bbox,
                    confidence=None,
                    source=AnnotationSource.FLORENCE2,
                )
                self._document = self._history.execute(AddAnnotationCommand(annotation))
                added += 1

            self.canvas.set_document(self._document)
            msg = f"VLM Auto-Annotate: added {added} new annotations ({len(candidates)} detected)"
            LOGGER.info(msg)
            self.statusBar().showMessage(msg)
        except Exception as error:
            LOGGER.exception("VLM auto-annotate failed")
            self.statusBar().showMessage(f"VLM auto-annotate failed: {error}")


    def _annotate_entire_dataset(self) -> None:
        """Run both detectors over every imported image with cancellable progress."""
        if self._crop_session is not None:
            self.statusBar().showMessage("Commit or cancel Crop Assist first")
            return
        if self._dataset_annotation_task is not None:
            self.statusBar().showMessage("Dataset annotation is already running")
            return
        if not self._project_documents:
            self.statusBar().showMessage("Import a folder or dataset first")
            return
        if not self._grounding_prompt.strip():
            self.statusBar().showMessage("Enter a Grounding DINO prompt first")
            return
        if self._grounding_model is None or self._grounding_processor is None:
            self._load_grounding_model()
        if self._grounding_model is None or self._grounding_processor is None:
            return
        if self._yolo_model is None:
            self._load_yolo_model()
        if self._yolo_model is None:
            return

        documents = list(self._project_documents.values())
        task = _DatasetAnnotationTask(
            documents,
            self._grounding_model,
            self._grounding_processor,
            self._yolo_model,
            self._normalize_grounding_prompt(self._grounding_prompt),
            self._confidence_threshold,
            self._fusion_config.overlap_removal_iou_threshold,
            self._fusion_config.overlap_removal_containment_threshold,
            enabled_classes=self._enabled_classes,
        )
        progress = _DatasetProgressDialog("Annotate Entire Dataset", len(documents), self)
        progress.cancelled.connect(task.cancel)
        task.signals.progress.connect(progress.update_progress)
        task.signals.completed.connect(self._dataset_annotation_completed)
        task.signals.cancelled.connect(self._dataset_annotation_cancelled)
        task.signals.failed.connect(self._dataset_annotation_failed)
        self._dataset_progress = progress
        self._dataset_annotation_task = task
        progress.show()
        self.statusBar().showMessage(f"Annotating dataset: 0/{len(documents)}")
        QThreadPool.globalInstance().start(task)

    def _annotate_entire_dataset_dino(self) -> None:
        """Run Grounding DINO prompt-ensemble inference over every image."""
        if self._crop_session is not None:
            self.statusBar().showMessage("Commit or cancel Crop Assist first")
            return
        if self._dataset_annotation_task is not None:
            self.statusBar().showMessage("Dataset annotation is already running")
            return
        if not self._project_documents:
            self.statusBar().showMessage("Import a folder or dataset first")
            return
        if not self._grounding_prompt.strip():
            self.statusBar().showMessage("Enter a Grounding DINO prompt first")
            return
        if self._grounding_model is None or self._grounding_processor is None:
            self._load_grounding_model()
        if self._grounding_model is None or self._grounding_processor is None:
            return

        documents = list(self._project_documents.values())
        task = _DatasetAnnotationTask(
            documents,
            self._grounding_model,
            self._grounding_processor,
            None,
            self._grounding_prompt,
            self._confidence_threshold,
            self._fusion_config.overlap_removal_iou_threshold,
            self._fusion_config.overlap_removal_containment_threshold,
            use_yolo=False,
            enabled_classes=self._enabled_classes,
        )
        progress = _DatasetProgressDialog(
            "DINO Annotate Entire Dataset", len(documents), self
        )
        progress.cancelled.connect(task.cancel)
        task.signals.progress.connect(progress.update_progress)
        task.signals.completed.connect(self._dataset_annotation_completed)
        task.signals.cancelled.connect(self._dataset_annotation_cancelled)
        task.signals.failed.connect(self._dataset_annotation_failed)
        self._dataset_progress = progress
        self._dataset_annotation_task = task
        progress.show()
        self.statusBar().showMessage(f"DINO annotating dataset: 0/{len(documents)}")
        QThreadPool.globalInstance().start(task)

    def _annotate_entire_dataset_dino_sam(self) -> None:
        """Run Zero-Shot Grounding DINO + SAM2 auto-annotation across every image."""
        if self._crop_session is not None:
            self.statusBar().showMessage("Commit or cancel Crop Assist first")
            return
        if self._dataset_annotation_task is not None:
            self.statusBar().showMessage("Dataset annotation is already running")
            return
        if not self._project_documents:
            self.statusBar().showMessage("Import a folder or dataset first")
            return
        if not self._grounding_prompt.strip():
            self.statusBar().showMessage("Enter a Grounding DINO prompt first")
            return
        if self._grounding_model is None or self._grounding_processor is None:
            self._load_grounding_model()
        if self._grounding_model is None or self._grounding_processor is None:
            return
        if self._sam2_model is None or self._sam2_processor is None:
            self._load_sam2_model()
        if self._sam2_model is None or self._sam2_processor is None:
            return

        documents = list(self._project_documents.values())
        task = _DatasetAnnotationTask(
            documents,
            self._grounding_model,
            self._grounding_processor,
            None,
            self._grounding_prompt,
            self._confidence_threshold,
            self._fusion_config.overlap_removal_iou_threshold,
            self._fusion_config.overlap_removal_containment_threshold,
            use_yolo=False,
            enabled_classes=self._enabled_classes,
            sam2_model=self._sam2_model,
            sam2_processor=self._sam2_processor,
            use_sam2=True,
        )
        progress = _DatasetProgressDialog(
            "DINO + SAM Annotate Entire Dataset", len(documents), self
        )
        progress.cancelled.connect(task.cancel)
        task.signals.progress.connect(progress.update_progress)
        task.signals.completed.connect(self._dataset_annotation_completed)
        task.signals.cancelled.connect(self._dataset_annotation_cancelled)
        task.signals.failed.connect(self._dataset_annotation_failed)
        self._dataset_progress = progress
        self._dataset_annotation_task = task
        progress.show()
        self.statusBar().showMessage(f"DINO + SAM annotating dataset: 0/{len(documents)}")
        QThreadPool.globalInstance().start(task)

    def _dataset_annotation_completed(self, result) -> None:  # type: ignore[no-untyped-def]
        documents, added, removed = result
        current_path = self._document.image_path if self._document is not None else None
        self._project_documents.update(documents)
        if current_path is not None and current_path in documents:
            updated = documents[current_path]
            if self._history is not None and self._document is not None:
                self._document = self._history.execute(
                    ReplaceDocumentCommand(self._document, updated)
                )
            else:
                self._document = updated
                self._history = AnnotationHistory(updated)
            self.canvas.set_document(self._document)
        self._finish_dataset_annotation()
        self.statusBar().showMessage(
            f"Dataset annotation complete: added {added} boxes, removed {removed} duplicates"
        )

    def _dataset_annotation_cancelled(self) -> None:
        self._finish_dataset_annotation()
        self.statusBar().showMessage("Dataset annotation cancelled")

    def _dataset_annotation_failed(self, message: str) -> None:
        self._finish_dataset_annotation()
        QMessageBox.critical(self, "Dataset annotation failed", message)
        self.statusBar().showMessage("Dataset annotation failed")

    def _finish_dataset_annotation(self) -> None:
        if self._dataset_progress is not None:
            self._dataset_progress.close()
            self._dataset_progress.deleteLater()
            self._dataset_progress = None
        self._dataset_annotation_task = None

    def _refine_with_sam2(self) -> None:
        """Refine the selected annotation using its box as a SAM2 prompt."""
        if self._document is None or self._history is None:
            self.statusBar().showMessage("Select an image before using SAM2")
            return
        if self._selected_annotation_id is None:
            self.statusBar().showMessage("Select a box before using SAM2")
            return
        if self._sam2_model is None or self._sam2_processor is None:
            self._load_sam2_model()
        if self._sam2_model is None or self._sam2_processor is None:
            return
        try:
            import torch
            from PIL import Image

            previous = next(
                (
                    item
                    for item in self._document.annotations
                    if item.annotation_id == self._selected_annotation_id
                ),
                None,
            )
            if previous is None:
                self.statusBar().showMessage("Select an annotation first")
                return
            image = Image.open(self._document.image_path).convert("RGB")
            box = previous.box
            pixel_box = [[[
                box.left * image.width,
                box.top * image.height,
                box.right * image.width,
                box.bottom * image.height,
            ]]]
            inputs = self._sam2_processor(
                images=image, input_boxes=pixel_box, return_tensors="pt"
            )
            device = next(self._sam2_model.parameters()).device
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.no_grad():
                outputs = self._sam2_model(**inputs, multimask_output=False)
            masks = self._sam2_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"]
            )
            mask = masks[0].squeeze()
            rows, columns = torch.where(mask > 0)
            if rows.numel() == 0:
                self.statusBar().showMessage("SAM2 did not find a mask for the selected box")
                return
            refined_box = BoundingBox(
                float(columns.min()) / image.width,
                float(rows.min()) / image.height,
                float(columns.max() + 1) / image.width,
                float(rows.max() + 1) / image.height,
            )
            updated = replace(previous, box=refined_box, source=AnnotationSource.SAM2)
            self._document = self._history.execute(
                UpdateAnnotationCommand(previous, updated)
            )
            self.canvas.set_document(self._document)
            self.statusBar().showMessage("Selected box refined with SAM2")
        except Exception as error:
            LOGGER.exception("SAM2 refinement failed")
            self.statusBar().showMessage(f"SAM2 refinement failed: {error}")

    def _grounding_annotate(self) -> None:
        """Use Grounding DINO text prompts to add boxes to the active image."""
        if self._document is None or self._history is None:
            self.statusBar().showMessage("Select an image before annotating")
            return
        if not self._grounding_prompt.strip():
            self.statusBar().showMessage("Enter a Grounding DINO prompt first")
            return
        if self._grounding_model is None or self._grounding_processor is None:
            self._load_grounding_model()
        if self._grounding_model is None or self._grounding_processor is None:
            return
        try:
            import torch
            from PIL import Image

            image = Image.open(self._document.image_path).convert("RGB")
            prompt = self._normalize_grounding_prompt(self._grounding_prompt)
            inputs = self._grounding_processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            )
            device = next(self._grounding_model.parameters()).device
            model_dtype = getattr(self._grounding_model, "dtype", None)
            inputs_device = {}
            for key, value in inputs.items():
                if hasattr(value, "to"):
                    if (
                        model_dtype is not None
                        and hasattr(value, "dtype")
                        and value.dtype in (torch.float32, torch.float64)
                        and model_dtype in (torch.float16, torch.bfloat16)
                    ):
                        inputs_device[key] = value.to(device=device, dtype=model_dtype)
                    else:
                        inputs_device[key] = value.to(device)
                else:
                    inputs_device[key] = value

            with torch.inference_mode():
                outputs = self._grounding_model(**inputs_device)
            try:
                results = self._grounding_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=self._confidence_threshold,
                    text_threshold=self._confidence_threshold,
                    target_sizes=[(image.height, image.width)],
                )[0]
            except TypeError:
                results = self._grounding_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    box_threshold=self._confidence_threshold,
                    text_threshold=self._confidence_threshold,
                    target_sizes=[(image.height, image.width)],
                )[0]
            added = self._add_grounding_results(results, image.width, image.height)
            self.canvas.set_document(self._document)
            self.statusBar().showMessage(f"Grounding DINO added {added} boxes")
        except Exception as error:
            LOGGER.exception("Grounding DINO annotation failed")
            self.statusBar().showMessage(f"Grounding DINO annotation failed: {error}")

    def _add_grounding_results(self, results, image_width: int, image_height: int) -> int:  # type: ignore[no-untyped-def]
        self._grounding_detections = []
        added = 0
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["text_labels"] if "text_labels" in results else results["labels"]
        for index, (box, score) in enumerate(zip(boxes, scores, strict=True)):
            if index >= len(labels):
                continue
            label = labels[index]
            class_name = grounding_class(str(label)) or ""
            if not class_name or class_name not in self._enabled_classes:
                continue
            left, top, right, bottom = box.tolist()
            left = max(0.0, min(float(left), float(image_width)))
            top = max(0.0, min(float(top), float(image_height)))
            right = max(0.0, min(float(right), float(image_width)))
            bottom = max(0.0, min(float(bottom), float(image_height)))
            if left >= right or top >= bottom:
                continue
            annotation = Annotation(
                class_name=class_name,
                box=BoundingBox(
                    left / image_width,
                    top / image_height,
                    right / image_width,
                    bottom / image_height,
                ),
                confidence=float(score),
                source=AnnotationSource.GROUNDING_DINO,
            )
            self._grounding_detections.append(
                ModelDetection(
                    class_name=class_name,
                    box=annotation.box,
                    confidence=annotation.confidence or 0.0,
                    source=AnnotationSource.GROUNDING_DINO,
                )
            )
            if self._document is not None and any(
                existing.class_name == annotation.class_name
                and self._box_iou(existing.box, annotation.box) >= 0.5
                for existing in self._document.annotations
            ):
                continue
            if self._history is not None:
                self._document = self._history.execute(AddAnnotationCommand(annotation))
                added += 1
        return added

    def _zero_shot_dino_sam_annotate(self) -> None:
        """Run Grounding DINO detection and refine with SAM2 in a single zero-shot step."""
        if self._document is None or self._history is None:
            self.statusBar().showMessage("Select an image before annotating")
            return
        if not self._grounding_prompt.strip():
            self.statusBar().showMessage("Enter a Grounding DINO prompt first")
            return
        if self._grounding_model is None or self._grounding_processor is None:
            self._load_grounding_model()
        if self._grounding_model is None or self._grounding_processor is None:
            return
        if self._sam2_model is None or self._sam2_processor is None:
            self._load_sam2_model()
        if self._sam2_model is None or self._sam2_processor is None:
            return

        try:
            import torch
            from PIL import Image

            image = Image.open(self._document.image_path).convert("RGB")
            prompt = self._normalize_grounding_prompt(self._grounding_prompt)
            inputs = self._grounding_processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            )
            device = next(self._grounding_model.parameters()).device
            model_dtype = getattr(self._grounding_model, "dtype", None)
            inputs_device = {}
            for key, value in inputs.items():
                if hasattr(value, "to"):
                    if (
                        model_dtype is not None
                        and hasattr(value, "dtype")
                        and value.dtype in (torch.float32, torch.float64)
                        and model_dtype in (torch.float16, torch.bfloat16)
                    ):
                        inputs_device[key] = value.to(device=device, dtype=model_dtype)
                    else:
                        inputs_device[key] = value.to(device)
                else:
                    inputs_device[key] = value

            with torch.inference_mode():
                outputs = self._grounding_model(**inputs_device)
            try:
                results = self._grounding_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=self._confidence_threshold,
                    text_threshold=self._confidence_threshold,
                    target_sizes=[(image.height, image.width)],
                )[0]
            except TypeError:
                results = self._grounding_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    box_threshold=self._confidence_threshold,
                    text_threshold=self._confidence_threshold,
                    target_sizes=[(image.height, image.width)],
                )[0]

            boxes = results.get("boxes", [])
            scores = results.get("scores", [])
            labels = results.get("text_labels", results.get("labels", []))

            if len(boxes) == 0:
                LOGGER.warning("Zero detections for text_prompt on %s", self._document.image_path.name)
                self.statusBar().showMessage(
                    f"Zero detections for '{self._grounding_prompt}' on {self._document.image_path.name}"
                )
                return

            candidate_boxes: list[list[float]] = []
            candidate_labels: list[str] = []
            candidate_scores: list[float] = []

            for index, (box, score) in enumerate(zip(boxes, scores, strict=True)):
                if index >= len(labels):
                    continue
                label = labels[index]
                class_name = grounding_class(str(label)) or ""
                if not class_name or class_name not in self._enabled_classes:
                    continue

                left, top, right, bottom = box.tolist()
                left = max(0.0, min(float(left), image.width))
                top = max(0.0, min(float(top), image.height))
                right = max(0.0, min(float(right), image.width))
                bottom = max(0.0, min(float(bottom), image.height))
                if left >= right or top >= bottom:
                    continue

                candidate_boxes.append([left, top, right, bottom])
                candidate_labels.append(class_name)
                candidate_scores.append(float(score))

            if not candidate_boxes:
                self.statusBar().showMessage("No valid detections matched enabled classes")
                return

            if self._vlm_filter_enabled and candidate_boxes:
                if self._vlm_helper is None:
                    self._load_vlm_model()
                if self._vlm_helper is not None:
                    from src.vlm_helper import crop_image, verify_crop_classes_batch

                    crops = [
                        crop_image(image, b_coords, normalized=False)
                        for b_coords in candidate_boxes
                    ]
                    matches = verify_crop_classes_batch(
                        crops, candidate_labels, vlm=self._vlm_helper
                    )
                    vlm_boxes: list[list[float]] = []
                    vlm_labels: list[str] = []
                    vlm_scores: list[float] = []
                    for b_coords, c_name, s_val, is_matched in zip(
                        candidate_boxes, candidate_labels, candidate_scores, matches, strict=True
                    ):
                        if is_matched:
                            vlm_boxes.append(b_coords)
                            vlm_labels.append(c_name)
                            vlm_scores.append(s_val)
                        else:
                            LOGGER.info(
                                "VLM rejected false positive '%s' at %s", c_name, b_coords
                            )
                    rejected = len(candidate_boxes) - len(vlm_boxes)
                    if rejected > 0:
                        LOGGER.info("VLM filtered out %d false-positive candidate boxes", rejected)
                    candidate_boxes = vlm_boxes
                    candidate_labels = vlm_labels
                    candidate_scores = vlm_scores

            if not candidate_boxes:
                self.statusBar().showMessage("No candidate boxes survived VLM verification")
                return

            sam_device = next(self._sam2_model.parameters()).device
            sam_dtype = getattr(self._sam2_model, "dtype", None)
            pixel_boxes = [candidate_boxes]
            sam_inputs = self._sam2_processor(
                images=image, input_boxes=pixel_boxes, return_tensors="pt"
            )
            sam_inputs_device = {}
            for k, v in sam_inputs.items():
                if hasattr(v, "to"):
                    if (
                        sam_dtype is not None
                        and hasattr(v, "dtype")
                        and v.dtype in (torch.float32, torch.float64)
                        and sam_dtype in (torch.float16, torch.bfloat16)
                    ):
                        sam_inputs_device[k] = v.to(device=sam_device, dtype=sam_dtype)
                    else:
                        sam_inputs_device[k] = v.to(sam_device)
                else:
                    sam_inputs_device[k] = v

            with torch.inference_mode():
                sam_outputs = self._sam2_model(**sam_inputs_device, multimask_output=False)
            masks = self._sam2_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"]
            )
            mask_batch = masks[0]

            added = 0
            for idx, (box_coords, class_name, score_val) in enumerate(
                zip(candidate_boxes, candidate_labels, candidate_scores, strict=True)
            ):
                left, top, right, bottom = box_coords
                mask = mask_batch[idx].squeeze()
                rows, columns = torch.where(mask > 0)
                if rows.numel() > 0:
                    refined_box = BoundingBox(
                        float(columns.min()) / image.width,
                        float(rows.min()) / image.height,
                        float(columns.max() + 1) / image.width,
                        float(rows.max() + 1) / image.height,
                    )
                else:
                    refined_box = BoundingBox(
                        left / image.width,
                        top / image.height,
                        right / image.width,
                        bottom / image.height,
                    )

                if any(
                    existing.class_name == class_name
                    and self._box_iou(existing.box, refined_box) >= 0.5
                    for existing in self._document.annotations
                ):
                    continue

                annotation = Annotation(
                    class_name=class_name,
                    box=refined_box,
                    confidence=score_val,
                    source=AnnotationSource.SAM2,
                )
                self._document = self._history.execute(AddAnnotationCommand(annotation))
                added += 1

            self.canvas.set_document(self._document)
            self.statusBar().showMessage(f"DINO + SAM auto-annotated {added} objects")
        except Exception as error:
            LOGGER.exception("DINO + SAM annotation failed")
            self.statusBar().showMessage(f"DINO + SAM annotation failed: {error}")

    def _open_auto_label_dialog(self) -> None:
        """Open the Roboflow-style Auto Label dialog with Grounding DINO, SAM2, and Florence-2 VLM."""
        from app.services.auto_label.engine import AutoLabelEngine
        from app.services.auto_label.models import DEFAULT_AUTO_LABEL_CLASSES, AutoLabelClass
        from app.ui.dialogs.auto_label_dialog import AutoLabelDialog

        image_paths = list(self._project_documents.keys()) if self._project_documents else []
        current_path = self._document.image_path if self._document is not None else None
        if not image_paths and current_path is not None:
            image_paths = [current_path]

        if not image_paths:
            self.statusBar().showMessage("Import a folder or dataset before opening Auto Label")
            return

        initial_classes = []
        for cls_name in sorted(self._enabled_classes):
            matching_default = next(
                (c for c in DEFAULT_AUTO_LABEL_CLASSES if c.name == cls_name), None
            )
            if matching_default is not None:
                initial_classes.append(
                    AutoLabelClass(
                        name=matching_default.name,
                        prompt=matching_default.prompt,
                        color=matching_default.color,
                    )
                )
            else:
                initial_classes.append(AutoLabelClass(name=cls_name, prompt="", color="#29b6f6"))

        if not initial_classes:
            initial_classes = list(DEFAULT_AUTO_LABEL_CLASSES)

        grounding_detector = None
        if self._grounding_model is not None and self._grounding_processor is not None:
            from pipeline_bridge import GroundingDinoDetector

            grounding_detector = GroundingDinoDetector(
                processor=self._grounding_processor,
                model=self._grounding_model,
            )

        sam_segmenter = None
        if self._sam2_model is not None and self._sam2_processor is not None:
            from pipeline_bridge import SamSegmenter

            sam_segmenter = SamSegmenter(
                processor=self._sam2_processor,
                model=self._sam2_model,
            )

        vlm_helper = self._vlm_helper

        engine = AutoLabelEngine(
            grounding_detector=grounding_detector,
            sam_segmenter=sam_segmenter,
            vlm_helper=vlm_helper,
            yolo_detector=self._yolo_model,
            yolo_model_name=str(self._yolo_model_path) if self._yolo_model_path else "yolo11n.pt",
        )

        dialog = AutoLabelDialog(
            image_paths=image_paths,
            current_image_path=current_path,
            engine=engine,
            initial_classes=initial_classes,
            ground_truth=self._project_documents,
            parent=self,
        )
        if hasattr(self, "_loaded_yolo_models") and self._loaded_yolo_models:
            dialog._active_yolo_models = list(self._loaded_yolo_models[:3])
            dialog._update_yolo_button_ui()
        dialog.batch_completed.connect(self._on_auto_label_batch_completed)
        dialog.exec()

    def _on_auto_label_preview_applied(self, result: Any) -> None:
        """Apply previewed Auto Label detections directly to the active document."""
        if self._document is None or self._history is None:
            return
        from app.services.annotation.domain import TARGET_CLASSES

        added = 0
        for det in result.detections:
            if det.class_name not in TARGET_CLASSES:
                continue
            if any(
                existing.class_name == det.class_name
                and self._box_iou(existing.box, det.box) >= 0.5
                for existing in self._document.annotations
            ):
                continue
            source = (
                AnnotationSource.SAM2
                if det.polygon_normalized
                else AnnotationSource.GROUNDING_DINO
            )
            ann = Annotation(
                class_name=det.class_name,
                box=det.box,
                confidence=det.confidence,
                source=source,
            )
            self._document = self._history.execute(AddAnnotationCommand(ann))
            added += 1

        self.canvas.set_document(self._document)
        self._remember_current_document()
        self._refresh_image_browser_order(preserve_current=True)
        self.statusBar().showMessage(
            f"Auto Label applied {added} annotation(s) to {self._document.image_path.name}"
        )

    def _on_auto_label_batch_completed(
        self, updated_documents: dict[Path, AnnotationDocument]
    ) -> None:
        """Handle completed batch Auto Label results."""
        try:
            self._project_documents.update(updated_documents)
            current_path = self._document.image_path if self._document is not None else None
            if current_path is not None and current_path in updated_documents:
                updated = updated_documents[current_path]
                if self._history is not None and self._document is not None:
                    self._document = self._history.execute(
                        ReplaceDocumentCommand(self._document, updated)
                    )
                else:
                    self._document = updated
                    self._history = AnnotationHistory(updated)
                self.canvas.set_document(self._document)
                self._remember_current_document()
            elif current_path is None and updated_documents:
                first_path = next(iter(updated_documents.keys()))
                self._load_image(first_path)
            self._refresh_image_browser_order(preserve_current=True)
            self.statusBar().showMessage(
                f"Auto Label applied to {len(updated_documents)} image(s)"
            )
        except Exception as err:
            LOGGER.exception("Failed to update documents from Auto Label: %s", err)

    @staticmethod
    def _normalize_grounding_prompt(prompt: str) -> str:
        """Convert common class-list formats to Grounding DINO's dot syntax."""
        classes = [part.strip() for part in re.split(r"[,.;\n]+", prompt) if part.strip()]
        return ". ".join(classes) + "."

    def _auto_annotate(self) -> None:
        """Run the loaded Ultralytics YOLO model on the active image."""
        if self._document is None or self._history is None:
            self.statusBar().showMessage("Select an image before auto annotating")
            return
        if self._yolo_model is None:
            self._load_yolo_model()
        if self._yolo_model is None:
            return
        try:
            device = 0 if detect_gpu().device == "cuda" else "cpu"
            result = self._yolo_model(
                str(self._document.image_path),
                device=device,
                conf=self._confidence_threshold,
                verbose=False,
            )[0]
            names = self._yolo_model.names
            class_order = self._enabled_classes
            self._yolo_detections = []
            added = 0
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = str(names[class_id])
                if class_name not in class_order:
                    continue
                left, top, right, bottom = box.xyxy[0].tolist()
                annotation = Annotation(
                    class_name=class_name,
                    box=BoundingBox(
                        left / self._document.image_width,
                        top / self._document.image_height,
                        right / self._document.image_width,
                        bottom / self._document.image_height,
                    ),
                    confidence=float(box.conf[0]),
                    source=AnnotationSource.YOLO,
                )
                self._yolo_detections.append(
                    ModelDetection(
                        class_name=class_name,
                        box=annotation.box,
                        confidence=annotation.confidence or 0.0,
                        source=AnnotationSource.YOLO,
                    )
                )
                if any(
                    existing.class_name == annotation.class_name
                    and self._box_iou(existing.box, annotation.box) >= 0.5
                    for existing in self._document.annotations
                ):
                    continue
                self._document = self._history.execute(AddAnnotationCommand(annotation))
                added += 1
            self.canvas.set_document(self._document)
            self.statusBar().showMessage(f"Auto annotation added {added} boxes")
        except Exception as error:
            LOGGER.exception("automatic annotation failed")
            self.statusBar().showMessage(f"Auto annotation failed: {error}")

    def _run_fusion(self) -> None:
        """Fuse cached Grounding DINO and YOLO detections for the active image."""
        if self._crop_session is not None:
            self.statusBar().showMessage("Commit or cancel Crop Assist before Label Fusion")
            return
        if self._document is None:
            self.statusBar().showMessage("Select an image before running Label Fusion")
            return
        detections = [*self._grounding_detections, *self._yolo_detections]
        if not detections:
            self.statusBar().showMessage("Run Grounding DINO and YOLO before Label Fusion")
            return
        result = FusionEngine(self._fusion_config).fuse(detections)
        annotations = tuple(
            Annotation(
                class_name=item.class_name,
                box=item.bbox,
                confidence=item.confidence,
                source=AnnotationSource.FUSED,
            )
            for item in result.detections
        )
        self._document = AnnotationDocument(
            self._document.image_path,
            self._document.image_width,
            self._document.image_height,
            annotations,
        )
        self._history = AnnotationHistory(self._document)
        self._fusion_result = result
        self._remember_current_document()
        self.canvas.set_document(self._document)
        self.canvas.set_fusion_statuses({
            annotation.annotation_id: detection.status
            for annotation, detection in zip(annotations, result.detections, strict=True)
        })
        stats = result.statistics
        self.statusBar().showMessage(
            f"Fusion: {stats.accepted} accepted, {stats.needs_review} review, "
            f"{stats.conflicts} conflicts, {stats.duplicate_removed} duplicates removed"
        )
        self._score_active_image()

    def _score_active_image(self) -> None:
        """Score the active image in a worker thread and show its review priority."""
        if self._document is None:
            self.statusBar().showMessage("Select an image before scoring review difficulty")
            return
        analysis = ImageAnalysis(
            image_path=self._document.image_path,
            detections=tuple([*self._grounding_detections, *self._yolo_detections]),
            fusion_result=self._fusion_result,
        )
        self.statusBar().showMessage("Calculating review difficulty...")
        task = _ActiveLearningTask(self._active_learning_engine, analysis)
        task.signals.completed.connect(self._active_learning_completed)
        task.signals.failed.connect(self._active_learning_failed)
        self._active_learning_task = task
        QThreadPool.globalInstance().start(task)

    def _active_learning_completed(self, result: DifficultyResult) -> None:
        self._active_learning_result = result
        self.statusBar().showMessage(
            f"Review priority {result.review_priority}/100 | "
            f"{result.difficulty_level.replace('_', ' ').title()} | "
            f"{result.recommended_action.replace('_', ' ').title()}"
        )

    def _active_learning_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"Difficulty calculation failed: {message}")

    def _remove_overlapping(self) -> None:
        """Remove redundant same-class boxes while preserving undo history."""
        if self._document is None or self._history is None:
            self.statusBar().showMessage("Select an image before removing overlaps")
            return
        kept, removed_count = remove_overlapping_annotations(
            self._document.annotations,
            self._fusion_config.overlap_removal_iou_threshold,
            self._fusion_config.overlap_removal_containment_threshold,
            self._fusion_config.overlap_removal_same_class_only,
        )
        if not removed_count:
            self.statusBar().showMessage("No overlapping duplicate boxes found")
            return
        updated = AnnotationDocument(
            self._document.image_path,
            self._document.image_width,
            self._document.image_height,
            kept,
        )
        self._document = self._history.execute(
            ReplaceDocumentCommand(self._document, updated)
        )
        self._remember_current_document()
        self.canvas.set_document(self._document)
        self.statusBar().showMessage(f"Removed {removed_count} overlapping duplicate boxes")

    def _delete_all_annotations(self) -> None:
        """Clear all annotations on the current image only, preserving undo history."""
        if self._document is None or self._history is None:
            self.statusBar().showMessage("Select an image before deleting annotations")
            return
        if not self._document.annotations:
            self.statusBar().showMessage("No annotations to delete on current image")
            return
        count = len(self._document.annotations)
        cleared_document = AnnotationDocument(
            self._document.image_path,
            self._document.image_width,
            self._document.image_height,
            (),
        )
        self._document = self._history.execute(
            ReplaceDocumentCommand(self._document, cleared_document)
        )
        self._selected_annotation_id = None
        self._remember_current_document()
        self.canvas.set_document(self._document)
        self._update_selection_properties()
        self.statusBar().showMessage(f"Deleted all {count} annotations on current image")

    def _delete_picture_from_database(
        self, path: Path | None = None, confirm: bool = True
    ) -> bool:
        """Permanently delete a picture and its annotations from dataset, database, and disk."""
        if self._crop_session is not None:
            self.statusBar().showMessage("Commit or cancel Crop Assist before deleting picture")
            return False

        target_path = path or (self._document.image_path if self._document is not None else None)
        if target_path is None:
            cur_item = self.image_browser.currentItem()
            if cur_item is not None and cur_item.data(256):
                target_path = Path(cur_item.data(256))

        if target_path is None:
            self.statusBar().showMessage("Select an image before deleting from database")
            return False

        if confirm:
            reply = QMessageBox.question(
                self,
                "Delete Picture from Database",
                f"Are you sure you want to delete picture from database?\n\n"
                f"File: {target_path.name}\n"
                f"Path: {target_path}\n\n"
                f"This will remove the picture, all its annotations, and its database records.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False

        # 1. Remove from active project documents
        self._project_documents.pop(target_path, None)

        # 2. Remove from active learning difficulty cache
        if hasattr(self, "_active_learning_engine") and self._active_learning_engine is not None:
            try:
                self._active_learning_engine.remove(target_path)
            except Exception as err:
                LOGGER.debug("Could not remove %s from active learning cache: %s", target_path, err)

        # 3. Remove from DatasetIndex if available
        if hasattr(self, "_dataset_index") and self._dataset_index is not None:
            try:
                self._dataset_index.delete(target_path)
            except Exception as err:
                LOGGER.debug("Could not remove %s from dataset index: %s", target_path, err)

        # 4. Remove associated label files (.txt)
        label_file = target_path.with_suffix(".txt")
        if label_file.is_file():
            try:
                label_file.unlink(missing_ok=True)
            except Exception as err:
                LOGGER.warning("Could not delete label file %s: %s", label_file, err)

        # 5. Remove image file from disk
        if target_path.is_file():
            try:
                target_path.unlink(missing_ok=True)
            except Exception as err:
                LOGGER.warning("Could not delete image file %s: %s", target_path, err)

        # 6. Remove from ImageBrowser UI
        self.image_browser.remove_path(target_path)

        # 7. Switch or clear active document
        if self._document is not None and self._document.image_path == target_path:
            self._selected_annotation_id = None
            if self._project_documents:
                next_path = next(iter(self._project_documents.keys()))
                self._load_image(next_path)
            else:
                self._document = None
                self._history = None
                self.canvas.clear()
                self._update_selection_properties()

        self.statusBar().showMessage(f"Deleted '{target_path.name}' from database")
        return True

    def _remove_database_duplicates(self) -> None:
        """Remove redundant boxes from every imported document."""
        if self._crop_session is not None:
            self.statusBar().showMessage("Commit or cancel Crop Assist first")
            return
        if self._dataset_annotation_task is not None:
            self.statusBar().showMessage("Dataset annotation is already running")
            return
        if not self._project_documents:
            self.statusBar().showMessage("Import a folder or dataset first")
            return

        updated_documents: dict[Path, AnnotationDocument] = {}
        removed_total = 0
        for document in self._project_documents.values():
            kept, removed = remove_overlapping_annotations(
                document.annotations,
                self._fusion_config.overlap_removal_iou_threshold,
                self._fusion_config.overlap_removal_containment_threshold,
                self._fusion_config.overlap_removal_same_class_only,
            )
            updated_documents[document.image_path] = (
                document
                if not removed
                else AnnotationDocument(
                    document.image_path,
                    document.image_width,
                    document.image_height,
                    kept,
                )
            )
            removed_total += removed

        self._project_documents.update(updated_documents)
        if self._document is not None and self._document.image_path in updated_documents:
            updated = updated_documents[self._document.image_path]
            if self._history is not None and updated is not self._document:
                self._document = self._history.execute(
                    ReplaceDocumentCommand(self._document, updated)
                )
            else:
                self._document = updated
            self.canvas.set_document(self._document)
            self._remember_current_document()
        self.statusBar().showMessage(
            f"Database duplicate cleanup complete: removed {removed_total} boxes"
        )

    def _toggle_selected_occluded(self) -> None:
        """Toggle the occlusion flag on the selected annotation."""
        self._toggle_selected_annotation_flag("occluded")

    def _toggle_selected_truncated(self) -> None:
        """Toggle the truncated flag on the selected annotation."""
        self._toggle_selected_annotation_flag("truncated")

    def _toggle_selected_annotation_flag(self, flag: str) -> None:
        if self._document is None or self._history is None or self._selected_annotation_id is None:
            self.statusBar().showMessage("Select an annotation first")
            return
        annotation = next(
            (
                item
                for item in self._document.annotations
                if item.annotation_id == self._selected_annotation_id
            ),
            None,
        )
        if annotation is None:
            self.statusBar().showMessage("Select an annotation first")
            return
        updated = replace(annotation, **{flag: not getattr(annotation, flag)})
        self._document = self._history.execute(UpdateAnnotationCommand(annotation, updated))
        self._remember_current_document()
        self.canvas.set_document(self._document)
        self.statusBar().showMessage(
            f"{flag.title()} set to {getattr(updated, flag)}"
        )
        self._update_selection_properties()

    @staticmethod
    def _box_iou(first: BoundingBox, second: BoundingBox) -> float:
        """Return intersection-over-union for two normalized boxes."""
        intersection_width = max(0.0, min(first.right, second.right) - max(first.left, second.left))
        intersection_height = max(
            0.0, min(first.bottom, second.bottom) - max(first.top, second.top)
        )
        intersection = intersection_width * intersection_height
        union = first.area + second.area - intersection
        return intersection / union if union else 0.0

    def _build_shortcuts(self) -> None:
        undo = QAction("Undo", self)
        undo.setShortcut("Ctrl+Z")
        undo.triggered.connect(self._undo_annotation_edit)
        redo = QAction("Redo", self)
        redo.setShortcuts(["Ctrl+Shift+Z", "Ctrl+Y"])
        redo.triggered.connect(self._redo_annotation_edit)
        self.addAction(undo)
        self.addAction(redo)

    def _undo_annotation_edit(self) -> None:
        """Undo the latest annotation edit and refresh the canvas."""
        if self._history is None or not self._history.can_undo:
            self.statusBar().showMessage("Nothing to undo")
            return
        self._document = self._history.undo()
        self._remember_current_document()
        self.canvas.set_document(self._document)
        self.statusBar().showMessage("Undid annotation edit")
        self._update_selection_properties()

    def _redo_annotation_edit(self) -> None:
        """Redo the latest undone annotation edit and refresh the canvas."""
        if self._history is None or not self._history.can_redo:
            self.statusBar().showMessage("Nothing to redo")
            return
        self._document = self._history.redo()
        self._remember_current_document()
        self.canvas.set_document(self._document)
        self.statusBar().showMessage("Redid annotation edit")
        self._update_selection_properties()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Navigate dataset images with arrows when an editor control is focused."""
        if event.type() != QEvent.Type.KeyPress:
            return super().eventFilter(watched, event)
        if self._crop_session is not None:
            return super().eventFilter(watched, event)
        if isinstance(watched, (QLineEdit, QAbstractSpinBox)):
            return super().eventFilter(watched, event)
        if not isinstance(watched, QWidget):
            return super().eventFilter(watched, event)
        if event.modifiers() & (
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
        ):
            return super().eventFilter(watched, event)
        if not (
            watched is self.canvas
            or self.canvas.isAncestorOf(watched)
            or watched is self.image_browser
            or self.image_browser.isAncestorOf(watched)
        ):
            return super().eventFilter(watched, event)
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self._navigate_image(-1)
            return True
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            self._navigate_image(1)
            return True
        return super().eventFilter(watched, event)

    def _navigate_image(self, offset: int) -> None:
        """Select a neighboring dataset image without leaving the current list."""
        count = self.image_browser.count()
        if count == 0:
            return
        current = self.image_browser.currentRow()
        next_row = max(0, min(count - 1, current + offset))
        if next_row != current:
            self.image_browser.setCurrentRow(next_row)

    def _dock(self, title: str, widget: QWidget) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"{title.lower()}Dock")
        dock.setWidget(widget)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        return dock
