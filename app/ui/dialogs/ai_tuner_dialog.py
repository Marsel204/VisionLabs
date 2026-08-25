"""Interactive AI Auto-Tuning dialog with live score tracking and prompt diffs."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.services.ai_tuner.models import (
    TunerConfig,
    TunerIteration,
    TunerResult,
)
from app.services.ai_tuner.tuner_engine import AITunerEngine
from app.services.annotation.domain import AnnotationDocument
from app.services.auto_label.engine import AutoLabelEngine
from app.services.auto_label.models import AutoLabelConfig

LOGGER = logging.getLogger(__name__)


class TuningWorkerThread(QThread):
    """Background worker executing the optimization loop without freezing the Qt UI."""

    iteration_completed = Signal(TunerIteration)
    tuning_finished = Signal(TunerResult)
    tuning_failed = Signal(str)

    def __init__(
        self,
        tuner_engine: AITunerEngine,
        sample_images: list[Path],
        ground_truth: dict[Path, AnnotationDocument],
        initial_config: AutoLabelConfig,
        tuner_config: TunerConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.tuner_engine = tuner_engine
        self.sample_images = sample_images
        self.ground_truth = ground_truth
        self.initial_config = initial_config
        self.tuner_config = tuner_config

    def run(self) -> None:
        try:
            result = self.tuner_engine.run_tuning(
                sample_images=self.sample_images,
                ground_truth=self.ground_truth,
                initial_config=self.initial_config,
                tuner_config=self.tuner_config,
                progress_callback=self._on_progress,
            )
            self.tuning_finished.emit(result)
        except Exception as err:
            LOGGER.exception("Tuning thread error")
            self.tuning_failed.emit(str(err))

    def _on_progress(self, iteration: TunerIteration) -> None:
        self.iteration_completed.emit(iteration)


class AITunerDialog(QDialog):
    """Modal dialog for autonomous prompt and hyperparameter optimization."""

    tuning_applied = Signal(AutoLabelConfig)

    def __init__(
        self,
        sample_images: list[Path],
        ground_truth: dict[Path, AnnotationDocument],
        current_config: AutoLabelConfig,
        engine: AutoLabelEngine | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("AITunerDialog")
        self.setWindowTitle("AI Auto-Tuner & Prompt Optimizer")
        self.resize(880, 680)
        self.setMinimumSize(780, 580)

        self.ground_truth = ground_truth or {}

        # Prioritize annotated sample images from dataset
        annotated_samples = [
            p
            for p in sample_images
            if p in self.ground_truth and self.ground_truth[p].annotations
        ]
        if not annotated_samples:
            annotated_in_project = [
                p for p, doc in self.ground_truth.items() if doc and doc.annotations
            ]
            if annotated_in_project:
                self.sample_images = annotated_in_project[:8]
            else:
                self.sample_images = list(sample_images)
        else:
            self.sample_images = list(sample_images)

        self.current_config = current_config
        self.auto_label_engine = engine or AutoLabelEngine()

        self._tuner_result: TunerResult | None = None
        self._worker_thread: TuningWorkerThread | None = None

        self._init_ui()
        self._check_ground_truth_status()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # Header Bar
        header_row = QHBoxLayout()
        title_label = QLabel("✨ AI Auto-Tuner & Prompt Optimizer")
        title_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #f8fafc;")
        badge = QLabel("Agentic In-Loop")
        badge.setStyleSheet(
            "background-color: #312e81; color: #a5b4fc; font-size: 11px; font-weight: 700; "
            "border-radius: 4px; padding: 2px 8px;"
        )
        header_row.addWidget(title_label)
        header_row.addWidget(badge)
        header_row.addStretch(1)
        layout.addLayout(header_row)

        # Ground Truth Status Banner
        self.gt_status_frame = QFrame()
        self.gt_status_frame.setObjectName("GTStatusFrame")
        gt_layout = QHBoxLayout(self.gt_status_frame)
        gt_layout.setContentsMargins(12, 8, 12, 8)
        self.gt_status_icon = QLabel("📌")
        self.gt_status_text = QLabel("Checking sample ground-truth annotations...")
        self.gt_status_text.setStyleSheet("color: #cbd5e1; font-size: 12px; font-weight: 500;")
        gt_layout.addWidget(self.gt_status_icon)
        gt_layout.addWidget(self.gt_status_text, 1)
        layout.addWidget(self.gt_status_frame)

        # Configuration Control Card
        config_card = QFrame()
        config_card.setStyleSheet(
            "background-color: #111827; border: 1px solid #1f293d; "
            "border-radius: 8px; padding: 6px;"
        )
        config_grid = QGridLayout(config_card)
        config_grid.setContentsMargins(10, 8, 10, 8)
        config_grid.setHorizontalSpacing(14)
        config_grid.setVerticalSpacing(8)

        # Target F1 Selector
        config_grid.addWidget(QLabel("Target Match Score:"), 0, 0)
        self.target_combo = QComboBox()
        self.target_combo.addItems(
            ["80% Match (Recommended)", "85% Match", "90% Match", "75% Match"]
        )
        config_grid.addWidget(self.target_combo, 0, 1)

        # Max Iterations
        config_grid.addWidget(QLabel("Max Iterations:"), 0, 2)
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(2, 8)
        self.iter_spin.setValue(4)
        config_grid.addWidget(self.iter_spin, 0, 3)

        # Vision Model Selector
        config_grid.addWidget(QLabel("Vision LLM (OpenRouter):"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItem(
            "Google Gemini 2.5 Flash (Fast & Cheap)", "google/gemini-2.5-flash"
        )
        self.model_combo.addItem("OpenAI GPT-4o Mini", "openai/gpt-4o-mini")
        self.model_combo.addItem("Google Gemini 2.5 Pro", "google/gemini-2.5-pro")
        self.model_combo.addItem("OpenAI GPT-4o", "openai/gpt-4o")
        config_grid.addWidget(self.model_combo, 1, 1, 1, 3)

        # API Key Status Row
        config_grid.addWidget(QLabel("API Key Status:"), 2, 0)
        self.key_status_label = QLabel()
        self._refresh_api_key_status()
        config_grid.addWidget(self.key_status_label, 2, 1, 1, 3)

        layout.addWidget(config_card)

        # Live Progress & Score HUD
        self.hud_frame = QFrame()
        self.hud_frame.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #334155; "
            "border-radius: 8px; padding: 8px;"
        )
        hud_layout = QVBoxLayout(self.hud_frame)
        hud_layout.setContentsMargins(12, 10, 12, 10)
        hud_layout.setSpacing(8)

        score_row = QHBoxLayout()
        self.score_tracker_label = QLabel("Score Progression: Ready to tune")
        self.score_tracker_label.setStyleSheet(
            "font-size: 13px; font-weight: 700; color: #a5b4fc;"
        )
        self.step_badge = QLabel("Idle")
        self.step_badge.setStyleSheet(
            "background-color: #1e293b; color: #94a3b8; font-size: 10px; font-weight: 700; "
            "border-radius: 4px; padding: 2px 6px;"
        )
        score_row.addWidget(self.score_tracker_label, 1)
        score_row.addWidget(self.step_badge, 0)
        hud_layout.addLayout(score_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #1e293b; border: 1px solid #334155; "
            "border-radius: 4px; height: 8px; text-align: center; } "
            "QProgressBar::chunk { background-color: #6366f1; border-radius: 3px; }"
        )
        hud_layout.addWidget(self.progress_bar)

        self.thought_stream = QTextEdit()
        self.thought_stream.setReadOnly(True)
        self.thought_stream.setMaximumHeight(90)
        self.thought_stream.setPlaceholderText("Agent thoughts and diagnostics will appear here...")
        self.thought_stream.setStyleSheet(
            "background-color: #0c1120; border: 1px solid #1e293b; border-radius: 4px; "
            "color: #cbd5e1; font-size: 11px;"
        )
        hud_layout.addWidget(self.thought_stream)

        layout.addWidget(self.hud_frame)

        # Prompt & Parameter Review Table (Diff View)
        diff_label = QLabel("Optimized Prompts & Parameters Diff:")
        diff_label.setStyleSheet("font-size: 12px; font-weight: 700; color: #f8fafc;")
        layout.addWidget(diff_label)

        self.diff_table = QTableWidget(len(self.current_config.classes), 3)
        self.diff_table.setHorizontalHeaderLabels(
            ["Class", "Current Prompt", "Tuned Prompt (Proposed)"]
        )
        self.diff_table.horizontalHeader().setStretchLastSection(True)
        self.diff_table.horizontalHeader().resizeSection(0, 110)
        self.diff_table.horizontalHeader().resizeSection(1, 280)
        self.diff_table.setStyleSheet(
            "QTableWidget { background-color: #0f172a; border: 1px solid #1e293b; "
            "border-radius: 6px; color: #f8fafc; font-size: 11px; } "
            "QHeaderView::section { background-color: #1e293b; color: #94a3b8; "
            "font-weight: 700; border: none; padding: 4px 6px; }"
        )
        for row, cls_item in enumerate(self.current_config.classes):
            self.diff_table.setItem(row, 0, QTableWidgetItem(f"🏷️ {cls_item.name}"))
            self.diff_table.setItem(row, 1, QTableWidgetItem(cls_item.prompt or cls_item.name))
            self.diff_table.setItem(row, 2, QTableWidgetItem("—"))
        layout.addWidget(self.diff_table, 1)

        # Bottom Action Buttons
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 4, 0, 0)

        self.start_btn = QPushButton("🚀 Start Auto-Tuning")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setStyleSheet(
            "background-color: #6366f1; color: #ffffff; font-weight: 700; font-size: 12px; "
            "padding: 8px 18px; border-radius: 6px; border: none;"
        )
        self.start_btn.clicked.connect(self._on_start_tuning)

        self.apply_btn = QPushButton("✔ Apply Tuned Settings")
        self.apply_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.apply_btn.setEnabled(False)
        self.apply_btn.setStyleSheet(
            "background-color: #10b981; color: #ffffff; font-weight: 700; font-size: 12px; "
            "padding: 8px 18px; border-radius: 6px; border: none;"
        )
        self.apply_btn.clicked.connect(self._on_apply_clicked)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.setStyleSheet(
            "background-color: #1e293b; color: #94a3b8; font-weight: 600; font-size: 12px; "
            "padding: 8px 14px; border-radius: 6px; border: 1px solid #334155;"
        )
        cancel_btn.clicked.connect(self.reject)

        bottom_row.addWidget(self.start_btn)
        bottom_row.addWidget(self.apply_btn)
        bottom_row.addStretch(1)
        bottom_row.addWidget(cancel_btn)
        layout.addLayout(bottom_row)

        self._apply_dialog_style()

    def _apply_dialog_style(self) -> None:
        self.setStyleSheet(
            """
            QDialog#AITunerDialog {
                background-color: #0b0f19;
                color: #f8fafc;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }
            QLabel {
                color: #cbd5e1;
                font-size: 11px;
            }
            QComboBox, QSpinBox {
                background-color: #141c2e;
                border: 1px solid #283654;
                border-radius: 4px;
                padding: 4px 8px;
                color: #ffffff;
                font-size: 11px;
            }
            QComboBox:focus, QSpinBox:focus {
                border-color: #6366f1;
            }
            QFrame#GTStatusFrame {
                background-color: #131b2e;
                border: 1px solid #1e293b;
                border-radius: 6px;
            }
            """
        )

    def _refresh_api_key_status(self) -> None:
        tuner_cfg = TunerConfig.from_env()
        if tuner_cfg.api_key:
            k = tuner_cfg.api_key
            masked = f"{k[:6]}...{k[-4:]}" if len(k) > 10 else "***"
            self.key_status_label.setText(f"🟢 OpenRouter Key Loaded from .env ({masked})")
            self.key_status_label.setStyleSheet("color: #34d399; font-weight: 600;")
        else:
            self.key_status_label.setText(
                "🟡 No OPENROUTER_API_KEY in .env (Math solver active; add key to .env for LLM)"
            )
            self.key_status_label.setStyleSheet("color: #fbbf24; font-weight: 500;")

    def _check_ground_truth_status(self) -> None:
        annotated_samples = [
            p
            for p in self.sample_images
            if p in self.ground_truth and self.ground_truth[p].annotations
        ]
        total_boxes = sum(len(self.ground_truth[p].annotations) for p in annotated_samples)

        if annotated_samples:
            self.gt_status_icon.setText("✅")
            names_summary = ", ".join(p.name[:18] for p in annotated_samples[:2])
            if len(annotated_samples) > 2:
                names_summary += f" (+{len(annotated_samples) - 2} more)"
            self.gt_status_text.setText(
                f"Found {len(annotated_samples)} reference image(s) with Ground Truth "
                f"({total_boxes} total objects: {names_summary}). Ready to tune!"
            )
            self.gt_status_frame.setStyleSheet(
                "background-color: #064e3b; border: 1px solid #059669; border-radius: 6px;"
            )
            self.start_btn.setEnabled(True)
        else:
            self.gt_status_icon.setText("⚠️")
            self.gt_status_text.setText(
                "No ground-truth annotations found in dataset. Please draw or verify boxes "
                "on 1-4 images first (or run Preview Mix -> Apply to Samples) to establish "
                "a reference target."
            )
            self.gt_status_frame.setStyleSheet(
                "background-color: #451a03; border: 1px solid #d97706; border-radius: 6px;"
            )
            self.start_btn.setEnabled(False)

    def _on_start_tuning(self) -> None:
        target_map = {0: 0.80, 1: 0.85, 2: 0.90, 3: 0.75}
        target_score = target_map.get(self.target_combo.currentIndex(), 0.80)
        selected_model = self.model_combo.currentData() or "google/gemini-2.0-flash-001"

        tuner_cfg = TunerConfig.from_env(
            target_f1_score=target_score,
            max_iterations=self.iter_spin.value(),
            model_name=selected_model,
        )

        self.start_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.step_badge.setText("Tuning...")
        self.step_badge.setStyleSheet(
            "background-color: #4338ca; color: #ffffff; font-size: 10px; "
            "font-weight: 700; border-radius: 4px; padding: 2px 6px;"
        )
        self.thought_stream.append("🚀 Initializing Agentic AI Tuner loop...")

        engine = AITunerEngine(auto_label_engine=self.auto_label_engine)

        self._worker_thread = TuningWorkerThread(
            tuner_engine=engine,
            sample_images=self.sample_images,
            ground_truth=self.ground_truth,
            initial_config=self.current_config,
            tuner_config=tuner_cfg,
            parent=self,
        )
        self._worker_thread.iteration_completed.connect(self._on_iteration_step)
        self._worker_thread.tuning_finished.connect(self._on_tuning_completed)
        self._worker_thread.tuning_failed.connect(self._on_tuning_failed)
        self._worker_thread.start()

    def _on_iteration_step(self, iteration: TunerIteration) -> None:
        score_pct = int(round(iteration.f1_score * 100))
        prec_pct = int(round(iteration.precision * 100))
        rec_pct = int(round(iteration.recall * 100))
        self.progress_bar.setValue(min(100, score_pct))
        self.score_tracker_label.setText(
            f"Score: {score_pct}% F1 (Prec: {prec_pct}%, Rec: {rec_pct}%)"
        )
        self.step_badge.setText(f"Iter #{iteration.iteration_index}")

        conf_pct = int(round(iteration.confidence_threshold * 100))
        iou_pct = int(round(iteration.iou_threshold * 100))
        log_msg = (
            f"<b>[Iter #{iteration.iteration_index}]</b> F1: {score_pct}% | "
            f"Conf: {conf_pct}% | IoU: {iou_pct}%"
        )
        if iteration.llm_reasoning:
            log_msg += f"<br><i>AI Rationale:</i> {iteration.llm_reasoning}"
        self.thought_stream.append(log_msg)

        # Update diff table proposed column
        for row in range(self.diff_table.rowCount()):
            cls_name_item = self.diff_table.item(row, 0)
            if cls_name_item:
                raw_name = cls_name_item.text().replace("🏷️ ", "").strip()
                if raw_name in iteration.prompt_updates:
                    new_val = iteration.prompt_updates[raw_name]
                    diff_item = QTableWidgetItem(new_val)
                    diff_item.setForeground(QColor("#34d399"))
                    self.diff_table.setItem(row, 2, diff_item)

    def _on_tuning_completed(self, result: TunerResult) -> None:
        self._tuner_result = result
        self.start_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)

        final_pct = int(round(result.final_f1 * 100))
        self.progress_bar.setValue(final_pct)
        if result.target_reached:
            self.step_badge.setText("Target Reached! 🎉")
            self.step_badge.setStyleSheet(
                "background-color: #059669; color: #ffffff; font-weight: 700; "
                "border-radius: 4px; padding: 2px 6px;"
            )
        else:
            self.step_badge.setText("Completed")
            self.step_badge.setStyleSheet(
                "background-color: #1e293b; color: #38bdf8; font-weight: 700; "
                "border-radius: 4px; padding: 2px 6px;"
            )

        self.score_tracker_label.setText(
            f"Final Score: {int(round(result.initial_f1 * 100))}% ➔ {final_pct}% F1"
        )
        self.thought_stream.append(f"<b style='color:#34d399;'>✔ {result.summary}</b>")

    def _on_tuning_failed(self, error_msg: str) -> None:
        self.start_btn.setEnabled(True)
        self.step_badge.setText("Error")
        self.step_badge.setStyleSheet(
            "background-color: #ef4444; color: #ffffff; font-weight: 700; "
            "border-radius: 4px; padding: 2px 6px;"
        )
        self.thought_stream.append(f"<b style='color:#ef4444;'>❌ Tuning failed: {error_msg}</b>")
        QMessageBox.critical(
            self, "Tuning Error", f"Optimization encountered an error:\n{error_msg}"
        )

    def _on_apply_clicked(self) -> None:
        if self._tuner_result is not None:
            self.tuning_applied.emit(self._tuner_result.final_config)
            self.accept()
