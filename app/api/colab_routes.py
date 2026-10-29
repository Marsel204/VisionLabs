"""Local studio endpoints for browser-authorized Colab annotation jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, SecretStr

from app.services.colab_bundle import build_source_bundle
from app.services.colab_connection import ColabConnection, ColabConnectionError, ColabGpu


class AuthorizationRequest(BaseModel):
    code: SecretStr


class ColabJobRequest(BaseModel):
    dataset_id: str
    classes: list[str] = Field(min_length=1, max_length=100)
    pipeline: Literal[
        "dino-masks", "dino-boxes", "yolo-masks", "yolo-boxes", "locate-masks", "locate-boxes"
    ] = "dino-masks"
    limit: int = Field(default=3, ge=1, le=100)
    confidence: float = Field(default=0.35, ge=0, le=1, allow_inf_nan=False)
    gpu: ColabGpu = "T4"
    only_unannotated: bool = True


def create_router(connection: ColabConnection, selection) -> APIRouter:
    router = APIRouter(prefix="/api/colab")

    def perform(operation):
        try:
            operation()
            return connection.status()
        except (ColabConnectionError, ValueError) as error:
            raise HTTPException(409, str(error)) from None

    @router.get("/status")
    def status():
        return connection.status()

    @router.post("/connect")
    def connect():
        return perform(connection.connect)

    @router.post("/authorize")
    def authorize(body: AuthorizationRequest):
        return perform(lambda: connection.authorize(body.code.get_secret_value()))

    @router.post("/disconnect")
    def disconnect():
        return perform(connection.disconnect)

    @router.post("/switch-account")
    def switch_account():
        return perform(connection.switch_account)

    @router.post("/account")
    def account():
        return perform(connection.refresh_account)

    @router.post("/stop")
    def stop():
        return perform(connection.cancel)

    @router.post("/acknowledge-release")
    def acknowledge_release():
        return perform(connection.acknowledge_release)

    @router.post("/jobs")
    def start(body: ColabJobRequest):
        root, images = selection(body)
        return perform(
            lambda: connection.start_job(
                root,
                images,
                body.classes,
                body.pipeline,
                confidence=body.confidence,
                gpu=body.gpu,
            )
        )

    @router.get("/files/{kind}")
    def file(kind: Literal["notebook", "source"]):
        root = Path(__file__).resolve().parents[2]
        if kind == "source":
            target = connection.profile / "visionlab-colab-source.zip"
            build_source_bundle(target)
            return FileResponse(target, filename=target.name, media_type="application/zip")
        target = root / "notebooks/visionlab_colab_batch.ipynb"
        if not target.is_file():
            target = root / "visionlab_colab_batch.ipynb"
        return FileResponse(
            target, filename="visionlab_colab_batch.ipynb", media_type="application/x-ipynb+json"
        )

    return router
