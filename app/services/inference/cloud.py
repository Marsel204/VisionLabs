"""Versioned cloud annotation protocol and a client with no model-runtime dependency."""

from __future__ import annotations

import base64
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.services.annotation.domain import BoundingBox
from app.services.auto_label.models import (
    AutoLabelClass,
    AutoLabelConfig,
    AutoLabelDetection,
    AutoLabelPipelineMode,
    AutoLabelResult,
)

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_BODY_BYTES = 28 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
Coordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
Point = tuple[Coordinate, Coordinate]


class CloudInferenceError(RuntimeError):
    """An inference request failed; existing annotations must remain intact."""


@dataclass(frozen=True)
class CloudInferenceSettings:
    url: str
    token: str = field(repr=False)
    timeout_seconds: float = 180

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        local_http = parsed.scheme == "http" and parsed.hostname in {
            "127.0.0.1",
            "localhost",
            "::1",
        }
        if not parsed.hostname or (parsed.scheme != "https" and not local_http):
            raise ValueError(
                "VISIONLAB_CLOUD_URL must use HTTPS (HTTP is allowed on loopback only)"
            )
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "VISIONLAB_CLOUD_URL cannot contain credentials, a query, or a fragment"
            )
        if not self.token or not self.token.isascii() or any(c.isspace() for c in self.token):
            raise ValueError("VISIONLAB_CLOUD_TOKEN must be a nonempty bearer token")
        if not math.isfinite(self.timeout_seconds) or not 1 <= self.timeout_seconds <= 900:
            raise ValueError("VISIONLAB_CLOUD_TIMEOUT must be between 1 and 900 seconds")

    @classmethod
    def from_env(cls) -> CloudInferenceSettings | None:
        backend = os.getenv("VISIONLAB_INFERENCE_BACKEND", "local").strip().lower()
        if backend == "local":
            return None
        if backend != "cloud":
            raise ValueError("VISIONLAB_INFERENCE_BACKEND must be local or cloud")
        url = os.getenv("VISIONLAB_CLOUD_URL", "").strip()
        if not url:
            raise ValueError("VISIONLAB_CLOUD_URL is required for cloud inference")
        return cls(
            url,
            os.getenv("VISIONLAB_CLOUD_TOKEN", ""),
            float(os.getenv("VISIONLAB_CLOUD_TIMEOUT", "180")),
        )


class CloudClass(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=120)
    prompt: str = Field(default="", max_length=1000)
    color: str = Field(default="#29b6f6", pattern=r"^#[0-9a-fA-F]{6}$")


class InferenceOptions(BaseModel):
    """Only inference settings cross the network; paths and devices remain local."""

    model_config = ConfigDict(extra="forbid")
    mode: AutoLabelPipelineMode = AutoLabelPipelineMode.DINO_SAM2_MASKS
    classes: list[CloudClass] = Field(min_length=1, max_length=100)
    confidence_threshold: Coordinate = 0.35
    text_threshold: Coordinate = 0.25
    box_iou_threshold: Coordinate = 0.5
    yolo_models: list[str] = Field(default_factory=lambda: ["yolo11n.pt"], max_length=3)
    enable_grounding_dino: bool = True
    enable_yolo: bool = True
    enable_florence2: bool = False
    enable_locate_anything: bool = False
    enable_florence2_verifier: bool = False
    enable_sam2_masks: bool = True

    @field_validator("yolo_models")
    @classmethod
    def trusted_weights(cls, names: list[str]) -> list[str]:
        # Client filesystem paths and arbitrary URLs cannot select executable model files.
        if any(not re.fullmatch(r"yolo(?:11|26|v8)[nsmxl](?:-seg)?\.pt", n) for n in names):
            raise ValueError("Cloud mode accepts built-in YOLO weights, not local paths or URLs")
        return names

    @classmethod
    def from_config(cls, config: AutoLabelConfig) -> InferenceOptions:
        return cls(
            mode=config.mode,
            classes=[
                CloudClass(name=c.name, prompt=c.prompt, color=c.color)
                for c in config.classes
                if c.enabled and c.name.strip()
            ],
            confidence_threshold=config.confidence_threshold,
            text_threshold=config.text_threshold,
            box_iou_threshold=config.box_iou_threshold,
            yolo_models=(config.yolo_models or [config.yolo_model_name])
            if config.enable_yolo
            else [],
            enable_grounding_dino=config.enable_grounding_dino,
            enable_yolo=config.enable_yolo,
            enable_florence2=config.enable_florence2,
            enable_locate_anything=config.enable_locate_anything,
            enable_florence2_verifier=config.enable_florence2_verifier,
            enable_sam2_masks=config.enable_sam2_masks,
        )

    def to_config(self) -> AutoLabelConfig:
        values = self.model_dump(exclude={"classes"})
        return AutoLabelConfig(
            **values,
            classes=[AutoLabelClass(**c.model_dump()) for c in self.classes],
        )


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    image_base64: str = Field(max_length=((MAX_IMAGE_BYTES + 2) // 3) * 4)
    options: InferenceOptions


class CloudDetection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_name: str = Field(min_length=1, max_length=120)
    confidence: Coordinate
    box: tuple[Coordinate, Coordinate, Coordinate, Coordinate]
    polygon: list[Point] = Field(default_factory=list, max_length=10000)

    @field_validator("box")
    @classmethod
    def positive_box(cls, value):
        if value[0] >= value[2] or value[1] >= value[3]:
            raise ValueError("A bounding box must have positive width and height")
        return value

    @field_validator("polygon")
    @classmethod
    def complete_polygon(cls, value):
        if value and len(value) < 3:
            raise ValueError("A polygon must have at least three vertices")
        return value


class InferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    detections: list[CloudDetection] = Field(max_length=10000)

    @classmethod
    def from_result(cls, result: AutoLabelResult) -> InferenceResponse:
        return cls(
            width=result.image_width,
            height=result.image_height,
            detections=[
                CloudDetection(
                    class_name=d.class_name,
                    confidence=d.confidence,
                    box=(d.box.left, d.box.top, d.box.right, d.box.bottom),
                    polygon=d.polygon_normalized,
                )
                for d in result.detections
            ],
        )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a bearer token to a redirect target.


class CloudInferenceClient:
    def __init__(self, settings: CloudInferenceSettings) -> None:
        self.settings = settings

    def predict(self, image_path: Path, config: AutoLabelConfig) -> AutoLabelResult:
        started = time.perf_counter()
        path = image_path.resolve()
        with Image.open(path) as image:
            width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise CloudInferenceError("Cloud inference supports images up to 40 megapixels")
        with path.open("rb") as source:
            data = source.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise CloudInferenceError("Cloud inference supports image files up to 20 MiB")
        try:
            payload = InferenceRequest(
                image_base64=base64.b64encode(data).decode("ascii"),
                options=InferenceOptions.from_config(config),
            )
        except ValidationError as error:
            raise CloudInferenceError(
                "Invalid cloud options; use built-in YOLO weights and valid classes and thresholds"
            ) from error
        request = Request(
            self.settings.url,
            data=payload.model_dump_json().encode(),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.settings.token}",
            },
        )
        try:
            with build_opener(_NoRedirect()).open(
                request,
                timeout=self.settings.timeout_seconds,
            ) as reply:
                body = reply.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as error:
            if error.code in (401, 403):
                message = "Cloud authentication failed; check VISIONLAB_CLOUD_TOKEN"
            elif error.code == 429:
                message = "Cloud inference rate limit reached; retry later"
            else:
                message = f"Cloud inference returned HTTP {error.code}; check the worker"
            raise CloudInferenceError(message) from None
        except (URLError, TimeoutError, OSError):
            raise CloudInferenceError(
                "Cloud inference could not connect or timed out; check the connection and worker"
            ) from None
        if len(body) > MAX_RESPONSE_BYTES:
            raise CloudInferenceError("Cloud inference response is too large")
        try:
            result = InferenceResponse.model_validate_json(body)
            if (result.width, result.height) != (width, height):
                raise ValueError("Image dimensions do not match")
            classes = {c.name.strip().casefold(): c for c in config.classes if c.enabled}
            detections = []
            for item in result.detections:
                cls = classes[item.class_name.strip().casefold()]
                detections.append(
                    AutoLabelDetection(
                        class_name=cls.name.strip(),
                        confidence=item.confidence,
                        box=BoundingBox(*item.box),
                        color=cls.color,
                        polygon_normalized=[list(p) for p in item.polygon],
                        polygon_pixels=[[p[0] * width, p[1] * height] for p in item.polygon],
                    )
                )
        except (ValidationError, ValueError, KeyError):
            raise CloudInferenceError("Cloud inference returned invalid annotation data") from None
        return AutoLabelResult(path, width, height, detections, time.perf_counter() - started)
