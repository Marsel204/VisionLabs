"""Prepare selected models without blocking download-status requests."""

from __future__ import annotations

import logging
import threading
import weakref
from typing import Any

from PIL import Image

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
        # Weak references must not keep evicted detector weights alive. Cache failed
        # attempts too: preparation never repeatedly exercises a failing runtime.
        self._yolo_warmups: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._yolo_residents: dict[str, tuple[weakref.ReferenceType, str]] = {}
        self._running = False

    def _discard_stale_yolo_status(self) -> None:
        """Called under the status lock; inspect residency without loading models."""
        for model_id, (reference, device) in list(self._yolo_residents.items()):
            if self._models[model_id]["state"] in {"queued", "preparing"}:
                continue
            cached = self._engine._yolo_detectors.get(model_id.removeprefix("yolo:"))
            resident = reference()
            if resident is not cached or cached is None or device != self._engine._device:
                if resident is not None and resident is not cached:
                    self._yolo_warmups.pop(resident, None)
                # Missing status already means 'prepare this selection' to consumers.
                del self._models[model_id]
                del self._yolo_residents[model_id]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._discard_stale_yolo_status()
            return {
                "backend": "cloud" if self._engine.uses_cloud else "local",
                "models": {key: dict(value) for key, value in self._models.items()},
            }

    def start(self, model_ids: list[str]) -> dict[str, Any]:
        if self._engine.uses_cloud:
            return self.snapshot()
        with self._lock:
            self._discard_stale_yolo_status()
            for model_id, model in list(self._models.items()):
                if model["state"] == "queued" and model_id not in model_ids:
                    del self._models[model_id]
                    self._yolo_residents.pop(model_id, None)
            for model_id in model_ids:
                existing = self._models.get(model_id)
                if existing and existing["state"] in {"queued", "preparing", "ready"}:
                    continue
                name = (
                    model_id.removeprefix("yolo:").replace("\\", "/").rsplit("/", 1)[-1]
                    if model_id.startswith("yolo:") else MODEL_LOADERS[model_id][0]
                )
                self._models[model_id] = {
                    "name": name, "state": "queued", "error": None,
                    "loaded": False,
                    # Other model families currently load weights only. None means
                    # their runtime readiness has not been measured by a forward pass.
                    "inference_ready": False if model_id.startswith("yolo:") else None,
                    "warning": None,
                }
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
                    # Selection may have changed while waiting for active inference.
                    if model_id not in self._models:
                        continue
                    self._models[model_id].update(state="preparing", error=None)
                try:
                    if model_id.startswith("yolo:"):
                        model = self._engine._get_yolo_detector(model_id.removeprefix("yolo:"))
                        with self._lock:
                            self._yolo_residents[model_id] = (
                                weakref.ref(model), self._engine._device,
                            )
                            self._models[model_id]["loaded"] = True
                        device = self._engine._get_yolo_device()
                        attempts = self._yolo_warmups.setdefault(model, {})
                        if device not in attempts:
                            try:
                                # Keep normal predict settings/tensor size. A small
                                # blank PIL input exercises the real Ultralytics path.
                                model(Image.new("RGB", (64, 64)), device=device, verbose=False)
                            except Exception as error:
                                LOGGER.warning("YOLO warmup failed for %s: %s", model_id, error)
                                attempts[device] = (
                                    f"Warmup failed; model loaded but not warmed: {error}"
                                )
                            else:
                                attempts[device] = None
                        with self._lock:
                            self._models[model_id].update(
                                inference_ready=attempts[device] is None,
                                warning=attempts[device],
                            )
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
                        # Preserve 'ready' as usable/loaded for existing desktop clients,
                        # even when optional warmup failed. Expose that distinction above.
                        self._models[model_id].update(state="ready", error=None, loaded=True)
