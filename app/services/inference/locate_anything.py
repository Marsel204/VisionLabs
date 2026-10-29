"""Lazy adapter for NVIDIA's custom LocateAnything-3B grounding runtime."""

from __future__ import annotations

import re
import threading
from typing import Any

from PIL import Image

from app.models.contracts import ModelRuntimeError
from app.services.annotation.domain import BoundingBox

MODEL_ID = "nvidia/LocateAnything-3B"
# The checkpoint includes custom Python code. Keep code and weights on one revision.
MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
_BOX_PATTERN = re.compile(
    r"<box>\s*<(-?\d+)>\s*<(-?\d+)>\s*<(-?\d+)>\s*<(-?\d+)>\s*</box>"
)


class LocateAnythingDetector:
    """Ground each class prompt separately so generated boxes retain their class."""

    def __init__(self, device: str = "auto") -> None:
        self._device = device
        self._dtype: Any = None
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._lock = threading.RLock()

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModel, AutoProcessor, AutoTokenizer

                from app.core.runtime import detect_gpu

                device = detect_gpu().device if self._device == "auto" else self._device
                if device == "gpu":
                    device = "cuda"
                dtype = torch.float32
                if device.startswith("cuda"):
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                options = {"revision": MODEL_REVISION, "trust_remote_code": True}
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **options)
                processor = AutoProcessor.from_pretrained(MODEL_ID, **options)
                model = AutoModel.from_pretrained(
                    MODEL_ID, torch_dtype=dtype, attn_implementation="sdpa", **options
                ).to(device).eval()
            except Exception as error:
                raise ModelRuntimeError(
                    "Locate Anything 3B could not load. Install the model dependencies "
                    "(transformers 4.57.x, torchvision, peft, decord, lmdb), check the "
                    f"Hugging Face download and available memory: {error}"
                ) from error
            self._device, self._dtype = device, dtype
            self._tokenizer, self._processor, self._model = tokenizer, processor, model

    @staticmethod
    def parse_boxes(answer: str) -> list[BoundingBox]:
        """Convert 0–1000 coordinate tokens into clipped, positive-area boxes."""
        boxes = []
        for match in _BOX_PATTERN.finditer(answer):
            x1, y1, x2, y2 = (max(0, min(1000, int(v))) / 1000 for v in match.groups())
            if x1 < x2 and y1 < y2:
                boxes.append(BoundingBox(x1, y1, x2, y2))
        return boxes

    def detect(self, image: Image.Image, prompts: list[str]) -> list[tuple[int, BoundingBox]]:
        """Return prompt indices and normalized boxes; the model emits no scores."""
        if not any(prompt.strip() for prompt in prompts):
            return []
        with self._lock:
            self._ensure_loaded()
            import torch

            detections = []
            try:
                with torch.inference_mode():
                    for index, prompt in enumerate(prompts):
                        if not prompt.strip():
                            continue
                        question = (
                            "Locate all the instances that matches the following description: "
                            f"{prompt.strip().rstrip('.')}."
                        )
                        messages = [{"role": "user", "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question},
                        ]}]
                        text = self._processor.py_apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        images, videos = self._processor.process_vision_info(messages)
                        inputs = self._processor(
                            text=[text], images=images, videos=videos, return_tensors="pt"
                        ).to(self._device)
                        response = self._model.generate(
                            pixel_values=inputs["pixel_values"].to(self._dtype),
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            image_grid_hws=inputs.get("image_grid_hws"),
                            tokenizer=self._tokenizer,
                            max_new_tokens=8192,
                            use_cache=True,
                            generation_mode="hybrid",
                            temperature=0.0,
                            do_sample=False,
                            verbose=False,
                        )
                        answer = response[0] if isinstance(response, tuple) else response
                        if not isinstance(answer, str):
                            raise TypeError("Expected a structured text response")
                        detections.extend((index, box) for box in self.parse_boxes(answer))
            except Exception as error:
                raise ModelRuntimeError(f"Locate Anything 3B inference failed: {error}") from error
            return detections
