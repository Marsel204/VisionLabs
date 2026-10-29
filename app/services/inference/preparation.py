"""Prepare selected models without blocking download-status requests."""

from __future__ import annotations

import logging
import threading
from typing import Any

LOGGER = logging.getLogger(__name__)

MODEL_LOADERS = {
    "grounding_dino": ("Grounding DINO", "_get_grounding_detector", "_ensure_loaded"),
    "sam2": ("SAM 2", "_get_sam_segmenter", "_ensure_loaded"),
    "florence2": ("Florence-2", "_get_vlm_helper", "ensure_loaded"),
    "locate_anything": ("Locate Anything 3B", "_get_locate_anything_detector", "_ensure_loaded"),
}


class ModelPreparation:
    def __init__(self, engine: Any, inference_lock: Any) -> None:
        self._engine = engine
        self._inference_lock = inference_lock
        self._lock = threading.Lock()
        self._models: dict[str, dict[str, Any]] = {}
        self._running = False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": "cloud" if self._engine.uses_cloud else "local",
                "models": {key: dict(value) for key, value in self._models.items()},
            }

    def start(self, model_ids: list[str]) -> dict[str, Any]:
        if self._engine.uses_cloud:
            return self.snapshot()
        with self._lock:
            for model_id in model_ids:
                existing = self._models.get(model_id)
                if existing and existing["state"] in {"queued", "preparing", "ready"}:
                    continue
                name = (
                    model_id.removeprefix("yolo:").replace("\\", "/").rsplit("/", 1)[-1]
                    if model_id.startswith("yolo:") else MODEL_LOADERS[model_id][0]
                )
                self._models[model_id] = {"name": name, "state": "queued", "error": None}
            if not self._running and any(m["state"] == "queued" for m in self._models.values()):
                self._running = True
                try:
                    threading.Thread(target=self._run, daemon=True).start()
                except Exception:
                    self._running = False
                    for model in self._models.values():
                        if model["state"] == "queued":
                            model.update(
                                state="error", error="Could not start model preparation. Retry."
                            )
                    raise
        return self.snapshot()

    def _run(self) -> None:
        while True:
            with self._lock:
                model_id = next(
                    (key for key, value in self._models.items() if value["state"] == "queued"),
                    None,
                )
                if model_id is None:
                    self._running = False
                    return
            # Downloads and inference share the same device. Status reads use a separate lock.
            with self._inference_lock:
                with self._lock:
                    self._models[model_id].update(state="preparing", error=None)
                try:
                    if model_id.startswith("yolo:"):
                        self._engine._get_yolo_detector(model_id.removeprefix("yolo:"))
                    else:
                        _, getter, ensure_loaded = MODEL_LOADERS[model_id]
                        model = getattr(self._engine, getter)()
                        getattr(model, ensure_loaded)()
                except Exception as error:
                    LOGGER.exception("Could not prepare model %s", model_id)
                    with self._lock:
                        self._models[model_id].update(state="error", error=str(error))
                else:
                    with self._lock:
                        self._models[model_id].update(state="ready", error=None)
