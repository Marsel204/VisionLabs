#!/usr/bin/env python3
"""Automated offscreen visual verification and high-resolution screenshot generator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force offscreen rendering if no display is available
if not os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.main import _dark_stylesheet
from app.services.annotation.domain import Annotation, AnnotationDocument, BoundingBox
from app.services.auto_label.models import AutoLabelConfig
from app.ui.canvas.annotation_canvas import CanvasMode
from app.ui.dialogs.ai_tuner_dialog import AITunerDialog
from app.ui.dialogs.auto_label_dialog import AutoLabelDialog
from app.ui.main_window import MainWindow


def generate_visual_verification_screenshots(output_dir: Path | None = None) -> list[Path]:
    """Capture comprehensive high-resolution visual verification screenshots."""
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "docs" / "screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([sys.argv[0], "-platform", "offscreen"])
    app.setStyleSheet(_dark_stylesheet())

    generated: list[Path] = []
    sample_img = _PROJECT_ROOT / "test image" / "000205_jpg.rf.2c4fa3ca1ce0b8650b26a0373bf44a9c.jpg"

    # Setup Main Window
    window = MainWindow()
    window.resize(1440, 900)

    doc = None
    if sample_img.is_file():
        doc = AnnotationDocument(
            image_path=sample_img,
            image_width=640,
            image_height=640,
            annotations=(
                Annotation(
                    annotation_id="1",
                    class_name="motorcycle",
                    box=BoundingBox(0.15, 0.40, 0.45, 0.85),
                    confidence=0.92,
                ),
                Annotation(
                    annotation_id="2",
                    class_name="car",
                    box=BoundingBox(0.55, 0.30, 0.90, 0.75),
                    confidence=0.88,
                ),
                Annotation(
                    annotation_id="3",
                    class_name="truck",
                    box=BoundingBox(0.02, 0.10, 0.35, 0.50),
                    confidence=0.79,
                ),
            ),
        )
        window._project_documents = {sample_img: doc}
        window.image_browser.set_paths([sample_img], {sample_img: 3})
        window._load_image(sample_img)

    # 1. Main Window Overview (No selection)
    window._select_annotation(None)
    window.show()
    app.processEvents()
    p1 = output_dir / "01_main_window_overview.png"
    window.grab().save(str(p1))
    generated.append(p1)
    print(f"✓ Generated: {p1}")

    # 2. Main Window with Selected Bounding Box & Active Properties
    if sample_img.is_file():
        window._select_annotation("1")
        app.processEvents()
        p2 = output_dir / "02_main_window_selected_box.png"
        window.grab().save(str(p2))
        generated.append(p2)
        print(f"✓ Generated: {p2}")

    # 3. Main Window in PAN Tool Mode
    window.canvas.set_pan_mode()
    app.processEvents()
    p3 = output_dir / "03_main_window_pan_mode.png"
    window.grab().save(str(p3))
    generated.append(p3)
    print(f"✓ Generated: {p3}")
    window.canvas.set_draw_mode()

    # 4. Auto Label Dialog
    if sample_img.is_file():
        try:
            dlg = AutoLabelDialog(image_paths=[sample_img], current_image_path=sample_img)
            dlg.resize(1200, 760)
            dlg.show()
            app.processEvents()
            p4 = output_dir / "04_auto_label_workspace.png"
            dlg.grab().save(str(p4))
            generated.append(p4)
            print(f"✓ Generated: {p4}")
            dlg.close()
        except Exception as e:
            print(f"⚠ AutoLabelDialog capture warning: {e}")

    # 5. AI Auto-Tuner Dialog
    if sample_img.is_file() and doc is not None:
        try:
            dlg_tuner = AITunerDialog(
                sample_images=[sample_img],
                ground_truth={sample_img: doc},
                current_config=AutoLabelConfig(),
            )
            dlg_tuner.resize(900, 680)
            dlg_tuner.show()
            app.processEvents()
            p5 = output_dir / "05_ai_tuner_hud.png"
            dlg_tuner.grab().save(str(p5))
            generated.append(p5)
            print(f"✓ Generated: {p5}")
            dlg_tuner.close()
        except Exception as e:
            print(f"⚠ AITunerDialog capture warning: {e}")

    window.close()
    return generated


if __name__ == "__main__":
    out = _PROJECT_ROOT / "docs" / "screenshots"
    print(f"Running visual verification suite, output directory: {out}")
    results = generate_visual_verification_screenshots(out)
    print(f"Visual verification complete: {len(results)} screenshots generated.")
