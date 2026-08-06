"""Main application window and initial dock layout."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from threading import Event

from PySide6.QtCore import QEvent, QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QAction, QImage
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QInputDialog,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
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
from app.export.exporters import CocoExporter, YoloExporter
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
from app.ui.canvas.annotation_canvas import AnnotationCanvas
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


class _DatasetAnnotationTask(QRunnable):
    """Run Grounding DINO and YOLO over dataset images off the UI thread."""

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
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.signals = _DatasetAnnotationSignals()
        self._documents = documents
        self._grounding_model = grounding_model
        self._grounding_processor = grounding_processor
        self._yolo_model = yolo_model
        self._prompt = prompt
        self._confidence = confidence
        self._iou_threshold = iou_threshold
        self._containment_threshold = containment_threshold
        self._cancel_requested = Event()

    def cancel(self) -> None:
        """Request cancellation after the current model inference finishes."""
        self._cancel_requested.set()

    def run(self) -> None:
        try:
            results: dict[Path, AnnotationDocument] = {}
            added_total = 0
            removed_total = 0
            for index, document in enumerate(self._documents, start=1):
                if self._cancel_requested.is_set():
                    self.signals.cancelled.emit()
                    return
                predictions = [
                    *self._grounding_detections(document),
                    *self._yolo_detections(document),
                ]
                additions = self._merge_predictions(document, predictions)
                kept, removed = remove_overlapping_annotations(
                    additions,
                    iou_threshold=self._iou_threshold,
                    containment_threshold=self._containment_threshold,
                    same_class_only=True,
                )
                updated = AnnotationDocument(
                    document.image_path,
                    document.image_width,
                    document.image_height,
                    (*document.annotations, *kept),
                )
                results[document.image_path] = updated
                added_total += len(kept)
                removed_total += removed
                self.signals.progress.emit(
                    index,
                    f"{document.image_path.name} | added {len(kept)} | "
                    f"removed {removed}",
                )
            self.signals.completed.emit((results, added_total, removed_total))
        except Exception as error:
            LOGGER.exception("dataset annotation failed")
            self.signals.failed.emit(str(error))

    def _grounding_detections(self, document: AnnotationDocument) -> list[Annotation]:
        import torch
        from PIL import Image

        image = Image.open(document.image_path).convert("RGB")
        inputs = self._grounding_processor(
            images=image,
            text=self._prompt,
            return_tensors="pt",
        )
        device = next(self._grounding_model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            outputs = self._grounding_model(**inputs)
        result = self._grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self._confidence,
            text_threshold=self._confidence,
            target_sizes=[(image.height, image.width)],
        )[0]
        labels = result["text_labels"] if "text_labels" in result else result["labels"]
        detections = []
        for box, score, label in zip(result["boxes"], result["scores"], labels, strict=True):
            class_name = next(
                (
                    name
                    for name in ("motorcycle", "car", "bus", "truck")
                    if name in str(label).lower()
                ),
                None,
            )
            if class_name is None:
                continue
            left, top, right, bottom = box.tolist()
            detections.append(
                self._annotation(
                    class_name,
                    left,
                    top,
                    right,
                    bottom,
                    float(score),
                    AnnotationSource.GROUNDING_DINO,
                    document,
                )
            )
        return [item for item in detections if item is not None]

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
    ) -> list[Annotation]:
        """Preserve manual boxes and reject predictions already covered by them."""
        additions = []
        for prediction in predictions:
            if any(
                existing.class_name == prediction.class_name
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
        self._confidence_threshold = 0.25
        self._grounding_prompt = "motorcycle. car. bus. truck."
        self._grounding_detections: list[ModelDetection] = []
        self._yolo_detections: list[ModelDetection] = []
        self._fusion_result: FusionResult | None = None
        self._fusion_config = fusion_config or FusionConfig()
        self._active_learning_engine = ActiveLearningEngine(active_learning_config)
        self._active_learning_result: DifficultyResult | None = None
        self._active_learning_task: _ActiveLearningTask | None = None
        self._dataset_annotation_task: _DatasetAnnotationTask | None = None
        self._dataset_progress: QProgressDialog | None = None
        self._project_documents: dict[Path, AnnotationDocument] = {}
        self._project_root: Path | None = None
        self._crop_session: CropSession | None = None
        self._crop_original_document: AnnotationDocument | None = None
        self._crop_original_history: AnnotationHistory | None = None
        self._crop_index = 0
        self._crop_directory: Path | None = None
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

    def _build_docks(self) -> None:
        class_list = QTreeWidget()
        class_list.setHeaderLabels(["Classes"])
        for name in ("motorcycle", "car", "bus", "truck"):
            QTreeWidgetItem(class_list, [name])
        class_list.itemSelectionChanged.connect(self._class_changed)
        class_list.setCurrentItem(class_list.topLevelItem(1))
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
        self._properties_layout.setContentsMargins(8, 8, 8, 8)
        self._properties_layout.setSpacing(8)
        self._property_group_layouts: dict[str, QVBoxLayout] = {}
        for title in (
            "Detection Settings",
            "AI Annotation",
            "Dataset Processing",
            "Review & Cleanup",
            "Crop Assist",
            "Project",
        ):
            group = QGroupBox(title)
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(6, 8, 6, 6)
            group_layout.setSpacing(4)
            self._property_group_layouts[title] = group_layout
            self._properties_layout.addWidget(group)
        self._properties_layout.addStretch()
        properties_scroll = QScrollArea()
        properties_scroll.setWidgetResizable(True)
        properties_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        properties_scroll.setWidget(properties)
        properties_dock = self._dock("Properties", properties_scroll)
        properties_dock.setMinimumWidth(320)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            properties_dock,
        )
        self.resizeDocks([classes_dock, properties_dock], [250, 320], Qt.Orientation.Horizontal)
        self._build_shortcuts()
        self.image_browser.image_selected.connect(self._load_image)
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
        refine_sam2 = QAction("Refine Selection (SAM2)", self)
        refine_sam2.setShortcut("Ctrl+Shift+S")
        refine_sam2.triggered.connect(self._refine_with_sam2)
        fuse = QAction("Label Fusion", self)
        fuse.setShortcut("Ctrl+Shift+F")
        fuse.triggered.connect(self._run_fusion)
        cleanup = QAction("Remove Overlapping Duplicates", self)
        cleanup.setShortcut("Ctrl+Shift+D")
        cleanup.triggered.connect(self._remove_overlapping)
        fusion_colors = QAction("Show Fusion Status Colors", self)
        fusion_colors.setCheckable(True)
        fusion_colors.setChecked(True)
        fusion_colors.toggled.connect(self.canvas.set_fusion_colors_enabled)
        active_learning = QAction("Score Review Difficulty", self)
        active_learning.setShortcut("Ctrl+Shift+R")
        active_learning.triggered.connect(self._score_active_image)
        dataset_annotate = QAction("Annotate Entire Dataset", self)
        dataset_annotate.triggered.connect(self._annotate_entire_dataset)
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

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(import_folder)
        file_menu.addAction(import_coco)
        file_menu.addAction(export_dataset)
        file_menu.addAction(save)
        model_menu = self.menuBar().addMenu("Model")
        model_menu.addAction(load_model)
        model_menu.addAction(load_grounding)
        model_menu.addAction(load_sam2)
        annotation_menu = self.menuBar().addMenu("Annotation")
        annotation_menu.addAction(auto_annotate)
        annotation_menu.addAction(grounding_annotate)
        annotation_menu.addAction(refine_sam2)
        annotation_menu.addAction(dataset_annotate)
        annotation_menu.addAction(fuse)
        annotation_menu.addAction(cleanup)
        annotation_menu.addAction(fusion_colors)
        annotation_menu.addAction(active_learning)
        crop_menu = annotation_menu.addMenu("Crop Assist")
        for action in (crop_start, crop_previous, crop_next, crop_commit, crop_cancel):
            crop_menu.addAction(action)
        filter_menu = annotation_menu.addMenu("Fusion Filters")
        for action in self._filter_actions.values():
            filter_menu.addAction(action)

        for group, action in (
            ("AI Annotation", auto_annotate),
            ("AI Annotation", grounding_annotate),
            ("AI Annotation", refine_sam2),
            ("Dataset Processing", dataset_annotate),
            ("Review & Cleanup", fuse),
            ("Review & Cleanup", cleanup),
            ("Review & Cleanup", fusion_colors),
            ("Review & Cleanup", active_learning),
            ("Crop Assist", crop_start),
            ("Crop Assist", crop_previous),
            ("Crop Assist", crop_next),
            ("Crop Assist", crop_commit),
            ("Crop Assist", crop_cancel),
            ("Project", save),
            ("Project", export_dataset),
        ):
            self._add_property_action(group, action)

        toolbar = QToolBar("Annotation tools", self)
        toolbar.setMovable(False)
        toolbar.addAction(import_folder)
        toolbar.addAction(import_coco)
        toolbar.addSeparator()
        prompt = QLineEdit(self._grounding_prompt, self)
        prompt.setMaximumWidth(220)
        prompt.setMinimumWidth(140)
        prompt.setPlaceholderText("car. bus. truck.")
        prompt.setToolTip("Grounding DINO prompt")
        prompt.textChanged.connect(self._set_grounding_prompt)
        confidence = QDoubleSpinBox(self)
        confidence.setRange(0.0, 1.0)
        confidence.setSingleStep(0.05)
        confidence.setDecimals(2)
        confidence.setValue(self._confidence_threshold)
        confidence.setFixedWidth(80)
        confidence.setToolTip("Minimum detection confidence")
        confidence.valueChanged.connect(self._set_confidence_threshold)
        asset_dir = Path(__file__).parent / "assets"
        confidence.setStyleSheet(
            "QDoubleSpinBox::up-arrow {"
            f"image: url(\"{asset_dir / 'spin-up.svg'}\");"
            "}"
            "QDoubleSpinBox::down-arrow {"
            f"image: url(\"{asset_dir / 'spin-down.svg'}\");"
            "}"
        )

        settings_layout = QFormLayout()
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.addRow("Prompt", prompt)
        settings_layout.addRow("Confidence", confidence)
        self._property_group_layouts["Detection Settings"].addLayout(settings_layout)

        self.addToolBar(toolbar)

    def _add_property_action(self, group: str, action: QAction) -> None:
        """Add an action to its grouped tool section in the Properties dock."""
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

    def _class_changed(self) -> None:
        selected = self.sender().currentItem()  # type: ignore[union-attr]
        if selected is not None:
            self._selected_class = selected.text(0)

    def _import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Import image folder")
        if not folder:
            return
        paths = sorted(
            path for path in Path(folder).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        self._project_documents = {}
        for path in paths:
            image = QImage(str(path))
            if not image.isNull():
                self._project_documents[path] = AnnotationDocument(
                    path,
                    image.width(),
                    image.height(),
                )
        self._project_root = None
        self.image_browser.set_paths(paths)
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
        destination = self._new_project_path(Path(parent_name), f"exported-{slug}")
        exporter = CocoExporter() if slug == "coco" else YoloExporter(variant=slug)
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
        self.image_browser.set_paths(sorted(self._project_documents))
        if result.documents:
            self.image_browser.setCurrentRow(0)

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

    def _add_box(self, box) -> None:  # type: ignore[no-untyped-def]
        if self._history is None:
            return
        self._document = self._history.execute(
            AddAnnotationCommand(Annotation(self._selected_class, box))
        )
        self._remember_current_document()
        self.canvas.set_document(self._document)
        self.statusBar().showMessage(f"Added {self._selected_class} box")

    def _resize_box(self, annotation_id, box) -> None:  # type: ignore[no-untyped-def]
        if self._history is None:
            return
        previous = next(
            item
            for item in self._history.document.annotations
            if item.annotation_id == annotation_id
        )
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
        self._document = self._history.execute(RemoveAnnotationCommand(annotation))
        self._remember_current_document()
        self.canvas.set_document(self._document)
        self.statusBar().showMessage("Deleted annotation")

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

    def _load_yolo_model(self) -> None:
        """Load YOLO weights once for reuse across images."""
        model_path, _ = QFileDialog.getOpenFileName(
            self, "Choose YOLO model weights", "", "YOLO weights (*.pt)"
        )
        if not model_path:
            return
        try:
            from ultralytics import YOLO

            self._yolo_model = YOLO(model_path)
            self._yolo_model_path = Path(model_path)
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
            self._grounding_processor = AutoProcessor.from_pretrained(self._grounding_model_id)
            self._grounding_model = GroundingDinoForObjectDetection.from_pretrained(
                self._grounding_model_id
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
            self._sam2_processor = Sam2Processor.from_pretrained(self._sam2_model_id)
            self._sam2_model = Sam2Model.from_pretrained(self._sam2_model_id).to(
                torch.device(device)
            )
            self._sam2_model.eval()
            self.statusBar().showMessage(f"Loaded SAM2: {self._sam2_model_id}")
        except Exception as error:
            self._sam2_processor = None
            self._sam2_model = None
            LOGGER.exception("SAM2 loading failed")
            self.statusBar().showMessage(f"SAM2 loading failed: {error}")

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
        )
        progress = QProgressDialog(
            "Preparing dataset annotation...", "Cancel", 0, len(documents), self
        )
        progress.setWindowTitle("Annotate Entire Dataset")
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.canceled.connect(task.cancel)
        task.signals.progress.connect(lambda value, _message: progress.setValue(value))
        task.signals.progress.connect(
            lambda value, message: progress.setLabelText(
                f"Processing image {value} / {len(documents)}\n{message}"
            )
        )
        task.signals.completed.connect(self._dataset_annotation_completed)
        task.signals.cancelled.connect(self._dataset_annotation_cancelled)
        task.signals.failed.connect(self._dataset_annotation_failed)
        self._dataset_progress = progress
        self._dataset_annotation_task = task
        progress.show()
        self.statusBar().showMessage(f"Annotating dataset: 0/{len(documents)}")
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
                item
                for item in self._document.annotations
                if item.annotation_id == self._selected_annotation_id
            )
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
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.no_grad():
                outputs = self._grounding_model(**inputs)
            results = self._grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self._confidence_threshold,
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
        class_order = {"motorcycle", "car", "bus", "truck"}
        self._grounding_detections = []
        added = 0
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["text_labels"] if "text_labels" in results else results["labels"]
        for index, (box, score) in enumerate(zip(boxes, scores, strict=True)):
            if index >= len(labels):
                continue
            label = labels[index]
            label_text = str(label).lower().strip(" .")
            class_name = next((name for name in class_order if name in label_text), "")
            if not class_name:
                continue
            left, top, right, bottom = box.tolist()
            left = max(0.0, min(float(left), image_width))
            top = max(0.0, min(float(top), image_height))
            right = max(0.0, min(float(right), image_width))
            bottom = max(0.0, min(float(bottom), image_height))
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
            if any(
                existing.class_name == annotation.class_name
                and self._box_iou(existing.box, annotation.box) >= 0.5
                for existing in self._document.annotations
            ):
                continue
            self._document = self._history.execute(AddAnnotationCommand(annotation))
            added += 1
        return added

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
            class_order = {"motorcycle", "car", "bus", "truck"}
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
        kept_ids = {item.annotation_id for item in kept}
        for annotation in self._document.annotations:
            if annotation.annotation_id not in kept_ids:
                self._document = self._history.execute(RemoveAnnotationCommand(annotation))
        self._remember_current_document()
        self.canvas.set_document(self._document)
        self.statusBar().showMessage(f"Removed {removed_count} overlapping duplicate boxes")

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

    def _redo_annotation_edit(self) -> None:
        """Redo the latest undone annotation edit and refresh the canvas."""
        if self._history is None or not self._history.can_redo:
            self.statusBar().showMessage("Nothing to redo")
            return
        self._document = self._history.redo()
        self._remember_current_document()
        self.canvas.set_document(self._document)
        self.statusBar().showMessage("Redid annotation edit")

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
