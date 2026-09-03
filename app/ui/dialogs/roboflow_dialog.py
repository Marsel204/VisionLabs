"""Interactive dialogs for importing from and uploading to Roboflow."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.services.annotation.domain import AnnotationDocument
from app.services.dataset.yolo_importer import YoloImportResult
from app.services.integrations.roboflow_client import (
    RoboflowClient,
    RoboflowError,
    RoboflowProjectRef,
)

LOGGER = logging.getLogger(__name__)


class _RoboflowDownloadWorker(QObject):
    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        workspace: str,
        project: str,
        version: int | str,
        destination: Path,
        api_key: str,
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.project = project
        self.version = version
        self.destination = destination
        self.api_key = api_key

    def run(self) -> None:
        try:
            client = RoboflowClient(self.api_key)
            result = client.download_and_import(
                self.workspace,
                self.project,
                self.version,
                self.destination,
                api_key=self.api_key,
                progress_callback=lambda frac, msg: self.progress.emit(frac, msg),
            )
            self.finished.emit(result)
        except Exception as err:
            LOGGER.exception("Roboflow download failed")
            self.failed.emit(str(err))


class _RoboflowUploadWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(int, int)  # (uploaded_count, total_count)
    failed = Signal(str)

    def __init__(
        self,
        workspace: str,
        project: str,
        documents: Sequence[AnnotationDocument],
        split: str,
        batch_name: str,
        api_key: str,
    ) -> None:
        super().__init__()
        self.workspace = workspace
        self.project = project
        self.documents = list(documents)
        self.split = split
        self.batch_name = batch_name
        self.api_key = api_key
        self._canceled = False

    def cancel(self) -> None:
        self._canceled = True

    def run(self) -> None:
        try:
            client = RoboflowClient(self.api_key)
            total = len(self.documents)
            uploaded = 0
            for idx, doc in enumerate(self.documents, start=1):
                if self._canceled:
                    break
                self.progress.emit(
                    idx,
                    total,
                    f"Uploading ({idx}/{total}): {doc.image_path.name}",
                )
                try:
                    client.upload_document(
                        workspace=self.workspace,
                        project=self.project,
                        document=doc,
                        split=self.split,
                        batch_name=self.batch_name,
                        api_key=self.api_key,
                    )
                    uploaded += 1
                except Exception as err:
                    LOGGER.warning("Failed uploading %s: %s", doc.image_path.name, err)

            self.finished.emit(uploaded, total)
        except Exception as err:
            LOGGER.exception("Roboflow upload failed")
            self.failed.emit(str(err))


class RoboflowImportDialog(QDialog):
    """Dialog to download and import datasets directly from Roboflow Universe or Workspace."""

    dataset_imported = Signal(object)  # Emits YoloImportResult

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import from Roboflow")
        self.resize(560, 360)
        self._thread: QThread | None = None
        self._worker: _RoboflowDownloadWorker | None = None
        self._init_ui()
        self._load_saved_creds()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 1. Credentials & Project Group
        group = QGroupBox("Roboflow Project & API Details", self)
        form = QFormLayout(group)

        self.api_key_edit = QLineEdit(self)
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("Enter your Roboflow Private API Key")

        self.show_key_btn = QPushButton("Show", self)
        self.show_key_btn.setFixedWidth(60)
        self.show_key_btn.clicked.connect(self._toggle_key_visibility)

        key_layout = QHBoxLayout()
        key_layout.addWidget(self.api_key_edit)
        key_layout.addWidget(self.show_key_btn)
        form.addRow("API Key:", key_layout)

        self.url_edit = QLineEdit(self)
        self.url_edit.setPlaceholderText("e.g. universe.roboflow.com/workspace/project or workspace/project")
        self.url_edit.textChanged.connect(self._on_url_changed)
        form.addRow("Project / URL:", self.url_edit)

        self.version_combo = QComboBox(self)
        self.version_combo.setEditable(True)
        self.version_combo.addItems(["1", "2", "3", "latest"])

        self.fetch_btn = QPushButton("Fetch Versions", self)
        self.fetch_btn.clicked.connect(self._fetch_versions)

        ver_layout = QHBoxLayout()
        ver_layout.addWidget(self.version_combo, stretch=1)
        ver_layout.addWidget(self.fetch_btn)
        form.addRow("Version:", ver_layout)

        self.dest_edit = QLineEdit(self)
        self.dest_edit.setPlaceholderText("Destination folder for imported project")
        self.browse_dest_btn = QPushButton("Browse...", self)
        self.browse_dest_btn.clicked.connect(self._browse_destination)

        dest_layout = QHBoxLayout()
        dest_layout.addWidget(self.dest_edit, stretch=1)
        dest_layout.addWidget(self.browse_dest_btn)
        form.addRow("Destination:", dest_layout)

        layout.addWidget(group)

        # 2. Progress & Status
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("", self)
        self.status_label.setStyleSheet("color: #888888;")
        layout.addWidget(self.status_label)

        # 3. Action Buttons
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        self.import_btn = QPushButton("Download & Import", self)
        self.import_btn.setDefault(True)
        self.import_btn.clicked.connect(self._start_download)

        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.import_btn)
        layout.addLayout(btn_layout)

    def _toggle_key_visibility(self) -> None:
        if self.api_key_edit.echoMode() == QLineEdit.EchoMode.Password:
            self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_key_btn.setText("Hide")
        else:
            self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_key_btn.setText("Show")

    def _load_saved_creds(self) -> None:
        creds = RoboflowClient.load_credentials()
        if "api_key" in creds:
            self.api_key_edit.setText(creds["api_key"])
        if "workspace" in creds and "project" in creds:
            ws, proj = creds["workspace"], creds["project"]
            if ws and proj:
                self.url_edit.setText(f"{ws}/{proj}")

    def _on_url_changed(self, text: str) -> None:
        try:
            ref = RoboflowClient.parse_roboflow_url(text)
            if ref.version is not None:
                self.version_combo.setEditText(str(ref.version))
            if not self.dest_edit.text():
                default_dest = Path.home() / "TrafficAnnotator" / "datasets" / f"roboflow-{ref.project}"
                self.dest_edit.setText(str(default_dest))
        except Exception:
            pass

    def _browse_destination(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose destination directory")
        if folder:
            self.dest_edit.setText(folder)

    def _fetch_versions(self) -> None:
        api_key = self.api_key_edit.text().strip()
        url_text = self.url_edit.text().strip()
        if not api_key:
            QMessageBox.warning(self, "API Key Required", "Please enter your Roboflow API key.")
            return
        try:
            ref = RoboflowClient.parse_roboflow_url(url_text)
            client = RoboflowClient(api_key)
            versions = client.get_project_versions(ref.workspace, ref.project)
            if versions:
                self.version_combo.clear()
                for item in sorted(
                    versions,
                    key=lambda v: int(v.get("version", v.get("id", 0))),
                    reverse=True,
                ):
                    ver_num = item.get("version", item.get("id", ""))
                    name = item.get("name", "")
                    label = f"{ver_num} ({name})" if name else str(ver_num)
                    self.version_combo.addItem(label, ver_num)
                self.status_label.setText(f"Found {len(versions)} versions for {ref.project}")
            else:
                self.status_label.setText("No versions returned by project.")
        except Exception as err:
            QMessageBox.warning(self, "Failed to Fetch Versions", str(err))

    def _start_download(self) -> None:
        api_key = self.api_key_edit.text().strip()
        url_text = self.url_edit.text().strip()
        dest_text = self.dest_edit.text().strip()

        if not api_key:
            QMessageBox.warning(self, "API Key Required", "Please enter your Roboflow API key.")
            return
        if not url_text:
            QMessageBox.warning(self, "Project Required", "Please enter a Roboflow project URL or slug.")
            return
        if not dest_text:
            QMessageBox.warning(self, "Destination Required", "Please choose an import destination folder.")
            return

        try:
            ref = RoboflowClient.parse_roboflow_url(url_text)
        except RoboflowError as err:
            QMessageBox.warning(self, "Invalid Roboflow Reference", str(err))
            return

        version = self.version_combo.currentData() or self.version_combo.currentText().split()[0]
        destination = Path(dest_text)

        # Save credentials for future use
        RoboflowClient.save_credentials(api_key, ref.workspace, ref.project)

        self.import_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        self.status_label.setText("Connecting to Roboflow API...")

        self._thread = QThread(self)
        self._worker = _RoboflowDownloadWorker(
            ref.workspace, ref.project, version, destination, api_key
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)

        self._thread.start()

    def _on_progress(self, fraction: float, message: str) -> None:
        self.progress_bar.setValue(int(fraction * 100))
        self.status_label.setText(message)

    def _on_finished(self, result: YoloImportResult) -> None:
        self._clean_thread()
        self.progress_bar.setValue(100)
        self.status_label.setText("Import completed successfully!")
        self.dataset_imported.emit(result)
        self.accept()

    def _on_failed(self, error_message: str) -> None:
        self._clean_thread()
        self.import_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error_message}")
        QMessageBox.critical(self, "Roboflow Import Failed", error_message)

    def closeEvent(self, event: Any) -> None:
        self._clean_thread()
        super().closeEvent(event)

    def reject(self) -> None:
        self._clean_thread()
        super().reject()

    def _clean_thread(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None
            self._worker = None


class RoboflowUploadDialog(QDialog):
    """Dialog to upload current dataset annotations and images directly to Roboflow."""

    def __init__(
        self,
        documents: Sequence[AnnotationDocument],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.documents = list(documents)
        self.setWindowTitle("Upload to Roboflow")
        self.resize(540, 360)
        self._thread: QThread | None = None
        self._worker: _RoboflowUploadWorker | None = None
        self._init_ui()
        self._load_saved_creds()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 1. Credentials & Destination Group
        group = QGroupBox("Roboflow Target Details", self)
        form = QFormLayout(group)

        self.api_key_edit = QLineEdit(self)
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("Enter your Roboflow Private API Key")
        form.addRow("API Key:", self.api_key_edit)

        self.project_edit = QLineEdit(self)
        self.project_edit.setPlaceholderText("Project name or ID (e.g. traffic-detector)")
        form.addRow("Project ID:", self.project_edit)

        self.split_combo = QComboBox(self)
        self.split_combo.addItems(["train", "valid", "test"])
        form.addRow("Dataset Split:", self.split_combo)

        self.batch_edit = QLineEdit("TrafficAnnotator-Batch", self)
        form.addRow("Batch Tag:", self.batch_edit)

        layout.addWidget(group)

        # Summary label
        total_boxes = sum(len(d.annotations) for d in self.documents)
        self.summary_label = QLabel(
            f"Ready to upload <b>{len(self.documents)} images</b> with <b>{total_boxes} annotations</b>.",
            self,
        )
        layout.addWidget(self.summary_label)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, len(self.documents))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("", self)
        self.status_label.setStyleSheet("color: #888888;")
        layout.addWidget(self.status_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.upload_btn = QPushButton("Upload to Roboflow", self)
        self.upload_btn.setDefault(True)
        self.upload_btn.clicked.connect(self._start_upload)

        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.upload_btn)
        layout.addLayout(btn_layout)

    def _load_saved_creds(self) -> None:
        creds = RoboflowClient.load_credentials()
        if "api_key" in creds:
            self.api_key_edit.setText(creds["api_key"])
        if "project" in creds:
            self.project_edit.setText(creds["project"])

    def _start_upload(self) -> None:
        api_key = self.api_key_edit.text().strip()
        project = self.project_edit.text().strip()
        split = self.split_combo.currentText()
        batch_name = self.batch_edit.text().strip() or "TrafficAnnotator"

        if not api_key:
            QMessageBox.warning(self, "API Key Required", "Please enter your Roboflow API key.")
            return
        if not project:
            QMessageBox.warning(self, "Project Required", "Please enter target Roboflow project ID.")
            return

        # Parse project slug if user pasted full URL
        try:
            ref = RoboflowClient.parse_roboflow_url(project)
            project = ref.project
            workspace = ref.workspace
        except Exception:
            workspace = ""

        RoboflowClient.save_credentials(api_key, workspace, project)

        self.upload_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self._thread = QThread(self)
        self._worker = _RoboflowUploadWorker(
            workspace, project, self.documents, split, batch_name, api_key
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)

        self._thread.start()

    def _on_progress(self, current: int, total: int, msg: str) -> None:
        self.progress_bar.setValue(current)
        self.status_label.setText(msg)

    def _on_finished(self, uploaded: int, total: int) -> None:
        self._clean_thread()
        self.progress_bar.setValue(total)
        self.status_label.setText(f"Completed: {uploaded}/{total} images uploaded.")
        QMessageBox.information(
            self,
            "Upload Completed",
            f"Successfully uploaded {uploaded} of {total} images and annotations to Roboflow!",
        )
        self.accept()

    def _on_failed(self, error_message: str) -> None:
        self._clean_thread()
        self.upload_btn.setEnabled(True)
        self.status_label.setText(f"Upload error: {error_message}")
        QMessageBox.critical(self, "Upload Failed", error_message)

    def closeEvent(self, event: Any) -> None:
        self._on_cancel()
        super().closeEvent(event)

    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
        self._clean_thread()
        self.reject()

    def _clean_thread(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None
            self._worker = None
