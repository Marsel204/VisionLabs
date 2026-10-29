"""Inference-only worker. Dataset files and the desktop API remain on the laptop."""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import secrets
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.services.auto_label.engine import AutoLabelEngine
from app.services.inference.cloud import (
    MAX_BODY_BYTES,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    InferenceRequest,
    InferenceResponse,
)

LOGGER = logging.getLogger(__name__)


def create_app(*, token: str | None = None, engine: AutoLabelEngine | None = None) -> FastAPI:
    """Create a worker with an explicit shared secret and no recursive cloud routing."""
    expected_token = token if token is not None else os.getenv("VISIONLAB_CLOUD_TOKEN", "")
    if (
        not expected_token
        or not expected_token.isascii()
        or any(c.isspace() for c in expected_token)
    ):
        raise ValueError("Set VISIONLAB_CLOUD_TOKEN before starting the cloud worker")
    pipeline = (
        engine
        if engine is not None
        else AutoLabelEngine(
            allow_cloud=False,
            fail_on_model_error=True,
        )
    )
    lock = threading.Lock()
    app = FastAPI(title="VisionLab Cloud Inference", docs_url=None, redoc_url=None)

    def run(payload: InferenceRequest) -> InferenceResponse:
        try:
            raw = base64.b64decode(payload.image_base64, validate=True)
            if len(raw) > MAX_IMAGE_BYTES:
                raise HTTPException(413, "Image file exceeds 20 MiB")
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(413, "Image exceeds 40 megapixels")
                image.load()
                rgb = image.convert("RGB")
        except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(422, "Invalid image") from None
        with tempfile.TemporaryDirectory(prefix="visionlab-infer-") as directory:
            path = Path(directory) / "image.png"
            rgb.save(path)
            try:
                with lock:
                    result = pipeline.run_preview(path, payload.options.to_config())
                return InferenceResponse.from_result(result)
            except Exception:
                LOGGER.exception("Cloud model inference failed")
                raise HTTPException(503, "Model inference failed; check worker logs") from None

    @app.post("/v1/infer")
    async def infer(request: Request):
        authorization = request.headers.get("Authorization", "")
        if not secrets.compare_digest(authorization.encode(), f"Bearer {expected_token}".encode()):
            raise HTTPException(401, "A valid bearer token is required")
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            raise HTTPException(400, "Invalid Content-Length") from None
        if length > MAX_BODY_BYTES:
            raise HTTPException(413, "Request is too large")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY_BYTES:
                raise HTTPException(413, "Request is too large")
        try:
            payload = InferenceRequest.model_validate_json(body)
        except ValidationError:
            raise HTTPException(422, "Invalid cloud inference request") from None
        return await run_in_threadpool(run, payload)

    return app
