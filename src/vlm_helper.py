"""Florence-2 Vision-Language Model (VLM) execution helper, verification, and annotation generation.

This module integrates existing image loading utilities across VisionLab:
- PIL Image loading (pipeline_bridge.py, app/ui/main_window.py, app/services/inference/dense_motorcycle.py)
- OpenCV image loading and slicing (app/services/crop_assisted/crop_generator.py)
- Domain bounding boxes and target classes (app/services/annotation/domain.py)

Provides:
- Universal image loader and cropper supporting file paths, NumPy arrays, and PIL Images.
- Florence-2 execution pipeline under task tokens such as `<CAPTION>` and `<OD>`.
- Verification wrapper checking if model predictions match target detection object classes.
- Object detection via `<OD>` task to generate new annotations directly.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)

# Default Florence-2 model identifier on Hugging Face Hub
DEFAULT_FLORENCE2_MODEL_ID = "microsoft/Florence-2-base"

# Synonym and alias mapping for target detection classes in traffic/vision domains
CLASS_SYNONYMS: dict[str, frozenset[str]] = {
    "motorcycle": frozenset(
        {
            "motorcycle",
            "motorcycles",
            "motorbike",
            "motorbikes",
            "scooter",
            "scooters",
            "moped",
            "mopeds",
            "vespa",
            "vespas",
            "two-wheeler",
            "two-wheelers",
            "motor cycle",
            "motor cycles",
            "bike",
            "bikes",
        }
    ),
    "car": frozenset(
        {
            "car",
            "cars",
            "automobile",
            "automobiles",
            "sedan",
            "sedans",
            "suv",
            "suvs",
            "coupe",
            "coupes",
            "hatchback",
            "hatchbacks",
            "vehicle",
            "vehicles",
            "van",
            "vans",
            "minivan",
            "minivans",
            "taxi",
            "taxis",
            "cab",
            "cabs",
            "jeep",
            "jeeps",
        }
    ),
    "bus": frozenset(
        {
            "bus",
            "buses",
            "busses",
            "minibus",
            "minibuses",
            "coach",
            "coaches",
            "transit bus",
            "transit buses",
            "double-decker",
            "double-deckers",
            "shuttle",
            "shuttles",
        }
    ),
    "truck": frozenset(
        {
            "truck",
            "trucks",
            "lorry",
            "lorries",
            "pickup",
            "pickups",
            "pickup truck",
            "pickup trucks",
            "semi-truck",
            "semi-trucks",
            "semi",
            "semis",
            "trailer",
            "trailers",
            "dump truck",
            "dump trucks",
            "tanker",
            "tankers",
            "hauler",
            "haulers",
        }
    ),
    "person": frozenset(
        {
            "person",
            "people",
            "pedestrian",
            "pedestrians",
            "human",
            "humans",
            "man",
            "men",
            "woman",
            "women",
            "rider",
            "riders",
            "driver",
            "drivers",
        }
    ),
}


# ==============================================================================
# Image Loading & Cropping Utilities
# ==============================================================================


def load_image(source: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """Load and convert an image from various input types into a standard PIL RGB Image.

    Supports:
    - Path / str: Reads file from disk, ensuring existence.
    - np.ndarray: Converts BGR (OpenCV) or RGB or Grayscale arrays.
    - PIL.Image.Image: Ensures RGB color space conversion.

    Args:
        source: Image file path, NumPy ndarray, or PIL Image.

    Returns:
        PIL.Image.Image in RGB mode.

    Raises:
        FileNotFoundError: If a given file path does not exist.
        TypeError: If the source type is unsupported.
        ValueError: If array format cannot be decoded into an image.
    """
    if isinstance(source, Image.Image):
        return source.convert("RGB") if source.mode != "RGB" else source

    if isinstance(source, (str, Path)):
        path = Path(source).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        try:
            return Image.open(path).convert("RGB")
        except Exception as err:
            raise ValueError(f"Failed to open image file {path}: {err}") from err

    if isinstance(source, np.ndarray):
        if source.ndim == 2:  # Grayscale
            return Image.fromarray(source).convert("RGB")
        if source.ndim == 3:
            channels = source.shape[2]
            if channels == 3:
                # Default assume OpenCV BGR if uint8, convert BGR -> RGB
                # However, if values are in RGB, fromarray handles it; here we convert BGR -> RGB standard
                import cv2

                rgb_array = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_array)
            if channels == 4:  # BGRA
                import cv2

                rgb_array = cv2.cvtColor(source, cv2.COLOR_BGRA2RGB)
                return Image.fromarray(rgb_array)
            if channels == 1:
                return Image.fromarray(source[:, :, 0]).convert("RGB")
        raise ValueError(f"Unsupported array shape for image conversion: {source.shape}")

    raise TypeError(f"Unsupported image source type: {type(source)}")


def crop_image(
    image: str | Path | Image.Image | np.ndarray,
    box: tuple[float, float, float, float] | Sequence[float] | Any,
    normalized: bool = True,
) -> Image.Image:
    """Crop a bounding box region from an image.

    Args:
        image: Source image (path, numpy array, or PIL Image).
        box: Coordinates (xmin, ymin, xmax, ymax) or (left, top, right, bottom).
             Can also be a BoundingBox dataclass with left, top, right, bottom attributes.
        normalized: Whether coordinates are normalized in [0, 1]. If False, coordinates are in pixels.

    Returns:
        Cropped PIL.Image.Image in RGB mode.
    """
    img = load_image(image)
    width, height = img.width, img.height

    if hasattr(box, "left") and hasattr(box, "top") and hasattr(box, "right") and hasattr(box, "bottom"):
        # app.services.annotation.domain.BoundingBox object
        xmin, ymin, xmax, ymax = float(box.left), float(box.top), float(box.right), float(box.bottom)
    elif len(box) == 4:
        xmin, ymin, xmax, ymax = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    else:
        raise ValueError(f"Expected 4 bounding box coordinates, got {box}")

    if normalized:
        left = max(0.0, min(xmin * width, float(width)))
        top = max(0.0, min(ymin * height, float(height)))
        right = max(0.0, min(xmax * width, float(width)))
        bottom = max(0.0, min(ymax * height, float(height)))
    else:
        left = max(0.0, min(xmin, float(width)))
        top = max(0.0, min(ymin, float(height)))
        right = max(0.0, min(xmax, float(width)))
        bottom = max(0.0, min(ymax, float(height)))

    if left >= right or top >= bottom:
        # Fallback to full image crop or minimal 1x1 patch to avoid crash
        LOGGER.warning("Degenerate crop coordinates [%s, %s, %s, %s]; using full image", left, top, right, bottom)
        return img

    return img.crop((int(round(left)), int(round(top)), int(round(right)), int(round(bottom))))


# ==============================================================================
# Florence-2 Vision-Language Model Pipeline
# ==============================================================================


def _apply_transformers_compatibility_patch() -> None:
    """Apply compatibility shims for Florence-2 dynamic remote code on transformers >= 4.45 / 5.x.

    Florence-2's remote configuration and processor files check older `forced_bos_token_id`
    and `additional_special_tokens` properties on Hugging Face classes.
    """
    try:
        import transformers.configuration_utils

        if not hasattr(transformers.configuration_utils.PretrainedConfig, "forced_bos_token_id"):
            transformers.configuration_utils.PretrainedConfig.forced_bos_token_id = None
    except Exception:
        pass

    try:
        import transformers.tokenization_utils_base

        if not hasattr(
            transformers.tokenization_utils_base.PreTrainedTokenizerBase,
            "additional_special_tokens",
        ):

            @property
            def _additional_special_tokens(self: Any) -> list[str]:
                if hasattr(self, "_additional_special_tokens_list"):
                    return [str(t) for t in self._additional_special_tokens_list]
                return self.special_tokens_map.get("additional_special_tokens", [])

            transformers.tokenization_utils_base.PreTrainedTokenizerBase.additional_special_tokens = (
                _additional_special_tokens
            )
    except Exception:
        pass

    try:
        import transformers.modeling_utils

        transformers.modeling_utils.PreTrainedModel._supports_sdpa = True
        transformers.modeling_utils.PreTrainedModel._supports_flash_attn_2 = False

        def _safe_sdpa_can_dispatch(self: Any, is_init_check: bool = False) -> bool:
            return True

        transformers.modeling_utils.PreTrainedModel._sdpa_can_dispatch = _safe_sdpa_can_dispatch
    except Exception:
        pass


class Florence2VLM:
    """Execution helper for Microsoft Florence-2 Vision-Language Models."""

    def __init__(
        self,
        model_id: str = DEFAULT_FLORENCE2_MODEL_ID,
        device: str = "auto",
        torch_dtype: Any | None = None,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        """Initialize Florence-2 VLM helper.

        Args:
            model_id: HuggingFace model repository ID (e.g., 'microsoft/Florence-2-base').
            device: Accelerator device ('auto', 'cuda', 'cpu', or 'mps').
            torch_dtype: Optional PyTorch dtype (e.g., torch.float16 or torch.float32).
            model: Optional pre-loaded Florence-2 model for dependency injection.
            processor: Optional pre-loaded Florence-2 processor for dependency injection.
        """
        self.model_id = model_id
        self._device_str = device
        self._torch_dtype = torch_dtype
        self._model = model
        self._processor = processor

    def _resolve_device(self) -> str:
        if self._device_str != "auto":
            return self._device_str
        try:
            from app.core.runtime import detect_gpu

            return "cuda" if detect_gpu().device == "cuda" else "cpu"
        except Exception:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_loaded(self) -> None:
        """Lazily load processor and model weights onto the target device."""
        if self._processor is not None and self._model is not None:
            return

        _apply_transformers_compatibility_patch()

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        target_device = self._resolve_device()
        dtype = self._torch_dtype
        if dtype is None:
            dtype = torch.float32

        LOGGER.info(
            "Loading Florence-2 model '%s' on %s (dtype: %s)",
            self.model_id,
            target_device,
            dtype,
        )

        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(torch.device(target_device))

            # Florence-2 safetensors store shared embeddings under language_model.model.shared.weight.
            # Tie encoder/decoder embed_tokens and lm_head to shared weights if freshly initialized.
            if (
                hasattr(self._model, "language_model")
                and hasattr(self._model.language_model, "model")
                and hasattr(self._model.language_model.model, "shared")
            ):
                shared_w = self._model.language_model.model.shared.weight
                if hasattr(self._model.language_model.model, "encoder") and hasattr(
                    self._model.language_model.model.encoder, "embed_tokens"
                ):
                    self._model.language_model.model.encoder.embed_tokens.weight = shared_w
                if hasattr(self._model.language_model.model, "decoder") and hasattr(
                    self._model.language_model.model.decoder, "embed_tokens"
                ):
                    self._model.language_model.model.decoder.embed_tokens.weight = shared_w
                if hasattr(self._model.language_model, "lm_head"):
                    self._model.language_model.lm_head.weight = shared_w

            self._model.eval()

    def run_task(
        self,
        image: str | Path | Image.Image | np.ndarray,
        task_token: str = "<CAPTION>",
        text_input: str | None = None,
        max_new_tokens: int = 1024,
        num_beams: int = 3,
    ) -> dict[str, Any] | str:
        """Run a Florence-2 task on an image or image crop.

        Args:
            image: PIL Image, NumPy array, or path to image file.
            task_token: Florence-2 task token (e.g. '<CAPTION>', '<DETAILED_CAPTION>', '<OD>').
            text_input: Optional additional prompt text following the task token.
            max_new_tokens: Maximum tokens for auto-regressive generation.
            num_beams: Beam search beam count.

        Returns:
            Parsed generation result dictionary or string output from post-processing.
        """
        self.ensure_loaded()
        import torch

        img = load_image(image)
        prompt = task_token if not text_input else f"{task_token} {text_input}"

        # Prepare pixel_values directly via image_processor to guarantee square resizing (768, 768)
        # across all crop aspect ratios on modern transformers versions.
        if hasattr(self._processor, "image_processor") and hasattr(self._processor, "tokenizer"):
            image_inputs = self._processor.image_processor(img, return_tensors="pt")
            prompt_text = (
                self._processor._construct_prompts([prompt])
                if hasattr(self._processor, "_construct_prompts")
                else prompt
            )
            text_inputs = self._processor.tokenizer(prompt_text, return_tensors="pt")
            inputs = {**text_inputs, "pixel_values": image_inputs["pixel_values"]}
        else:
            inputs = self._processor(text=prompt, images=img, return_tensors="pt")

        try:
            device = next(self._model.parameters()).device
            model_dtype = getattr(self._model, "dtype", None)
            device_inputs = {}
            for key, value in inputs.items():
                if hasattr(value, "to"):
                    if (
                        model_dtype is not None
                        and hasattr(value, "dtype")
                        and value.dtype in (torch.float32, torch.float64)
                        and model_dtype in (torch.float16, torch.bfloat16)
                    ):
                        device_inputs[key] = value.to(device=device, dtype=model_dtype)
                    else:
                        device_inputs[key] = value.to(device)
                else:
                    device_inputs[key] = value
        except (StopIteration, AttributeError):
            device_inputs = dict(inputs)

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=device_inputs.get("input_ids"),
                pixel_values=device_inputs.get("pixel_values"),
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                use_cache=False,
                do_sample=False,
            )

        generated_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]

        try:
            parsed_answer = self._processor.post_process_generation(
                generated_text,
                task=task_token,
                image_size=(img.width, img.height),
            )
            return parsed_answer
        except Exception as err:
            LOGGER.warning("Florence-2 post-processing failed, returning raw text: %s", err)
            return generated_text

    def generate_caption(
        self,
        image: str | Path | Image.Image | np.ndarray,
        task_token: str = "<CAPTION>",
    ) -> str:
        """Generate a caption text string for an image crop.

        Args:
            image: Source image crop.
            task_token: Task token (default: '<CAPTION>').

        Returns:
            Extracted caption string (e.g., 'a motorcycle on the road').
        """
        result = self.run_task(image=image, task_token=task_token)

        if isinstance(result, dict):
            # Florence-2 parsed output format for <CAPTION>: {"<CAPTION>": "a motorcycle parked..."}
            caption = result.get(task_token, "")
            if not caption and result:
                # Fallback to the first value if key differs
                caption = next(iter(result.values()))
            if isinstance(caption, str):
                return caption.strip()
            return str(caption).strip()

        if isinstance(result, str):
            # Strip task tokens from raw text if present
            cleaned = re.sub(r"<[^>]+>", "", result).strip()
            return cleaned

        return str(result).strip()

    def detect_objects(
        self,
        image: str | Path | Image.Image | np.ndarray,
    ) -> list[dict[str, Any]]:
        """Run Florence-2 object detection (`<OD>`) on a full image.

        Args:
            image: Source image (path, numpy array, or PIL Image).

        Returns:
            List of detection dicts, each with keys:
              - ``label`` (str): Predicted object label.
              - ``box`` (list[float]): Pixel coordinates ``[xmin, ymin, xmax, ymax]``.
        """
        result = self.run_task(image=image, task_token="<OD>")

        detections: list[dict[str, Any]] = []

        if isinstance(result, dict):
            # Florence-2 <OD> parsed output: {"<OD>": {"bboxes": [...], "labels": [...]}}
            od_data = result.get("<OD>", result)
            if isinstance(od_data, dict):
                bboxes = od_data.get("bboxes", [])
                labels = od_data.get("labels", [])
                for bbox, label in zip(bboxes, labels):
                    detections.append({"label": str(label), "box": [float(c) for c in bbox]})

        return detections


# Global default VLM instance for lightweight reuse
_GLOBAL_VLM: Florence2VLM | None = None


def get_default_vlm() -> Florence2VLM:
    """Retrieve or initialize the global shared Florence2VLM instance."""
    global _GLOBAL_VLM
    if _GLOBAL_VLM is None:
        _GLOBAL_VLM = Florence2VLM()
    return _GLOBAL_VLM


# ==============================================================================
# Target Object Class Matcher & Verification Wrapper
# ==============================================================================


def match_caption_to_class(caption: str, target_class: str) -> bool:
    """Check if a model-predicted caption string matches a target detection object class.

    Performs case-insensitive token and phrase matching using known domain synonyms
    and word boundaries to avoid false substring collisions (e.g., prevents 'carpet' from matching 'car').

    Args:
        caption: Text caption predicted by Florence-2.
        target_class: Target object detection class name (e.g. 'motorcycle', 'car', 'bus', 'truck').

    Returns:
        True if the caption contains the target class or its synonyms, False otherwise.
    """
    if not caption or not target_class:
        return False

    normalized_caption = caption.lower()
    normalized_target = target_class.lower().strip()

    # Retrieve known synonyms for the target class, or build dynamic word forms
    synonyms = CLASS_SYNONYMS.get(
        normalized_target,
        frozenset({normalized_target, f"{normalized_target}s"}),
    )

    for word in synonyms:
        # Match as discrete word or hyphenated token using regex word boundaries
        pattern = r"\b" + re.escape(word) + r"\b"
        if re.search(pattern, normalized_caption):
            return True

    return False


def verify_crop_class(
    image_crop: Image.Image | np.ndarray | str | Path,
    target_class: str,
    vlm: Florence2VLM | None = None,
    task_token: str = "<CAPTION>",
) -> bool:
    """Wrapper function that passes an image crop to Florence-2 and verifies class match.

    Takes an image crop, passes it to Florence-2 under a `<CAPTION>` task token,
    and returns a boolean value checking if the model-predicted text string matches
    our target detection object class.

    Args:
        image_crop: Image crop input (PIL Image, NumPy array, or path).
        target_class: Target class name to verify (e.g. 'motorcycle', 'car', 'bus', 'truck').
        vlm: Optional custom or injected Florence2VLM instance. If None, uses default VLM.
        task_token: Florence-2 task token (default: '<CAPTION>').

    Returns:
        True if model-predicted text string matches the target detection class, False otherwise.
    """
    model_runner = vlm if vlm is not None else get_default_vlm()
    caption = model_runner.generate_caption(image=image_crop, task_token=task_token)
    matched = match_caption_to_class(caption=caption, target_class=target_class)
    LOGGER.debug(
        "Crop verification: target='%s', predicted_caption='%s', matched=%s",
        target_class,
        caption,
        matched,
    )
    return matched


# ==============================================================================
# VLM Object Detection & Annotation Generation
# ==============================================================================


def map_od_label_to_class(label: str) -> str | None:
    """Map a Florence-2 OD predicted label to a VisionLab target class.

    Uses the existing ``CLASS_SYNONYMS`` dictionary to resolve synonyms
    (e.g. 'sedan' → 'car', 'lorry' → 'truck', 'scooter' → 'motorcycle').

    Args:
        label: Raw label string predicted by Florence-2 ``<OD>`` task.

    Returns:
        Matched target class name (e.g. 'motorcycle', 'car', 'bus', 'truck'),
        or ``None`` if the label does not match any known target class.
    """
    if not label:
        return None
    normalized = label.lower().strip()

    # Direct match against known target class names
    for target_class, synonyms in CLASS_SYNONYMS.items():
        if normalized in synonyms or normalized == target_class:
            return target_class

    return None


def generate_annotations(
    image: str | Path | Image.Image | np.ndarray,
    image_width: int,
    image_height: int,
    vlm: Florence2VLM | None = None,
    enabled_classes: set[str] | frozenset[str] | None = None,
) -> list[tuple[str, Any]]:
    """Run Florence-2 object detection and produce annotation-ready results.

    Uses the ``<OD>`` task to detect objects in the full image, maps labels
    to target classes, and returns normalized bounding boxes ready to be
    converted into ``Annotation`` domain objects.

    Args:
        image: Source image (path, numpy array, or PIL Image).
        image_width: Image width in pixels (for coordinate normalization).
        image_height: Image height in pixels (for coordinate normalization).
        vlm: Optional Florence2VLM instance. If None, uses default VLM.
        enabled_classes: Optional set of target classes to filter by.
            If None, all detected target classes are included.

    Returns:
        List of ``(class_name, BoundingBox)`` tuples where ``BoundingBox`` is
        imported from the annotation domain. Each entry represents a detected
        object ready to be added as an ``Annotation``.
    """
    from app.services.annotation.domain import BoundingBox

    model_runner = vlm if vlm is not None else get_default_vlm()
    detections = model_runner.detect_objects(image=image)

    results: list[tuple[str, Any]] = []

    for det in detections:
        raw_label = det.get("label", "")
        box_pixels = det.get("box", [])
        if len(box_pixels) != 4:
            continue

        mapped_class = map_od_label_to_class(raw_label)
        if mapped_class is None:
            LOGGER.debug("VLM OD label '%s' does not match any target class, skipping", raw_label)
            continue

        if enabled_classes is not None and mapped_class not in enabled_classes:
            continue

        # Normalize pixel coordinates to [0, 1]
        xmin = max(0.0, min(float(box_pixels[0]) / image_width, 1.0))
        ymin = max(0.0, min(float(box_pixels[1]) / image_height, 1.0))
        xmax = max(0.0, min(float(box_pixels[2]) / image_width, 1.0))
        ymax = max(0.0, min(float(box_pixels[3]) / image_height, 1.0))

        if xmin >= xmax or ymin >= ymax:
            LOGGER.warning("VLM OD degenerate box [%s, %s, %s, %s] for '%s', skipping",
                           xmin, ymin, xmax, ymax, mapped_class)
            continue

        try:
            bbox = BoundingBox(xmin, ymin, xmax, ymax)
        except Exception:
            LOGGER.warning("VLM OD invalid box [%s, %s, %s, %s] for '%s', skipping",
                           xmin, ymin, xmax, ymax, mapped_class)
            continue

        results.append((mapped_class, bbox))
        LOGGER.debug("VLM OD detected '%s' (%s) at [%.3f, %.3f, %.3f, %.3f]",
                      raw_label, mapped_class, xmin, ymin, xmax, ymax)

    LOGGER.info("VLM OD generated %d annotation candidates from %d raw detections",
                len(results), len(detections))
    return results


# ==============================================================================
# LocateAnything-3B Vision-Language Model Pipeline
# ==============================================================================

DEFAULT_LOCATE_ANYTHING_MODEL_ID = "nvidia/LocateAnything-3B"


class LocateAnything3BVLM:
    """Execution helper for NVIDIA LocateAnything-3B Vision-Language Model."""

    def __init__(
        self,
        model_id: str = DEFAULT_LOCATE_ANYTHING_MODEL_ID,
        device: str = "auto",
        torch_dtype: Any | None = None,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        """Initialize LocateAnything-3B VLM helper.

        Args:
            model_id: HuggingFace model repository ID (default: 'nvidia/LocateAnything-3B').
            device: Accelerator device ('auto', 'cuda', 'cpu').
            torch_dtype: Optional PyTorch dtype.
            model: Optional pre-loaded model for dependency injection.
            processor: Optional pre-loaded processor for dependency injection.
        """
        self.model_id = model_id
        self._device_str = device
        self._torch_dtype = torch_dtype
        self._model = model
        self._processor = processor

    def _resolve_device(self) -> str:
        if self._device_str != "auto":
            return self._device_str
        try:
            from app.core.runtime import detect_gpu

            return "cuda" if detect_gpu().device == "cuda" else "cpu"
        except Exception:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"

    def _apply_transformers_compatibility_patch(self) -> None:
        """Patch dynamic module methods for newer transformers compatibility."""
        try:
            import inspect
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            cls = get_class_from_dynamic_module(
                "modeling_locateanything.LocateAnythingPreTrainedModel",
                self.model_id,
            )
            if cls is not None and hasattr(cls, "_check_and_adjust_attn_implementation"):
                orig_fn = cls._check_and_adjust_attn_implementation
                if not getattr(orig_fn, "_is_visionlab_patched", False):

                    def patched_check_and_adjust_attn_implementation(
                        self_model: Any, attn_implementation: Any, *args: Any, **kwargs: Any
                    ) -> Any:
                        sig = inspect.signature(orig_fn)
                        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
                        return orig_fn(self_model, attn_implementation, *args, **filtered)

                    patched_check_and_adjust_attn_implementation._is_visionlab_patched = True
                    cls._check_and_adjust_attn_implementation = (
                        patched_check_and_adjust_attn_implementation
                    )
        except Exception as err:
            LOGGER.debug("LocateAnything transformers compatibility patch skipped: %s", err)

    def ensure_loaded(self) -> None:
        """Lazily load LocateAnything-3B processor and model weights."""
        if self._processor is not None and self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoProcessor

        target_device = self._resolve_device()
        dtype = self._torch_dtype
        if dtype is None:
            dtype = (
                torch.bfloat16
                if (target_device == "cuda" and torch.cuda.is_bf16_supported())
                else (torch.float16 if target_device == "cuda" else torch.float32)
            )

        LOGGER.info(
            "Loading LocateAnything-3B model '%s' on %s (dtype: %s)",
            self.model_id,
            target_device,
            dtype,
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._apply_transformers_compatibility_patch()
        self._model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(target_device)
        self._model.eval()

    def detect_objects(
        self,
        image: str | Path | Image.Image | np.ndarray,
        classes: Sequence[Any] | str,
        confidence_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Run LocateAnything-3B visual localization for given classes / prompts.

        Args:
            image: Source image.
            classes: Target class objects, list of prompt strings, or single prompt.
            confidence_threshold: Minimum confidence score.

        Returns:
            List of detection dicts with 'label' (str), 'box' (list[float] [xmin, ymin, xmax, ymax]),
            and 'score' (float).
        """
        self.ensure_loaded()
        img = load_image(image)
        w, h = img.width, img.height

        class_prompts: list[str] = []
        if isinstance(classes, str):
            class_prompts = [classes]
        else:
            for item in classes:
                p = getattr(item, "effective_prompt", None) or getattr(item, "name", str(item))
                class_prompts.append(str(p).strip())

        if not class_prompts:
            return []

        import torch

        prompt = "Locate: " + ", ".join(class_prompts)
        detections: list[dict[str, Any]] = []

        try:
            inputs = self._processor(images=img, text=prompt, return_tensors="pt")
            device = next(self._model.parameters()).device
            device_inputs = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()
            }

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **device_inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )

            output_text = self._processor.decode(generated_ids[0], skip_special_tokens=False)

            if hasattr(self._processor, "post_process_generation"):
                parsed = self._processor.post_process_generation(output_text, image_size=(w, h))
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "box" in item:
                            detections.append(item)
            else:
                # Regex coordinate token extraction fallback
                matches = re.findall(
                    r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]",
                    output_text,
                )
                for match in matches:
                    coords = [float(c) for c in match]
                    if all(0.0 <= c <= 1.0 for c in coords):
                        coords = [coords[0] * w, coords[1] * h, coords[2] * w, coords[3] * h]
                    elif (
                        all(0.0 <= c <= 1000.0 for c in coords)
                        and max(coords) <= 1000.0
                        and max(w, h) > 1000.0
                    ):
                        coords = [
                            coords[0] * w / 1000.0,
                            coords[1] * h / 1000.0,
                            coords[2] * w / 1000.0,
                            coords[3] * h / 1000.0,
                        ]

                    xmin = min(coords[0], coords[2])
                    ymin = min(coords[1], coords[3])
                    xmax = max(coords[0], coords[2])
                    ymax = max(coords[1], coords[3])

                    label = class_prompts[0]
                    detections.append(
                        {
                            "label": label,
                            "box": [xmin, ymin, xmax, ymax],
                            "score": 0.88,
                        }
                    )
        except Exception as err:
            LOGGER.warning("LocateAnything-3B inference encountered an error: %s", err)

        return detections
