"""Browser login and bounded Colab jobs with private credentials and owned-runtime cleanup."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import queue
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import webbrowser
import zipfile
from pathlib import Path
from typing import Literal, get_args
from urllib.parse import urlsplit
from uuid import uuid4

from PIL import Image
from pydantic import BaseModel, Field

from app.services.annotation.domain import BoundingBox
from app.services.auto_label.engine import compute_box_iou
from app.services.colab_batch import make_config
from app.services.colab_bundle import build_source_bundle
from app.services.colab_studio_worker import bootstrap_source
from app.services.dataset.index import DatasetIndex
from app.services.dataset.repository import AnnotationRepository, RevisionConflict, atomic_text
from app.services.inference.cloud import InferenceOptions, InferenceResponse

CLI_VERSION = "0.7.5"
MAX_TRANSFER_BYTES = 50 * 1024 * 1024
ColabGpu = Literal["T4", "L4", "A100", "H100", "G4"]
SUPPORTED_COLAB_GPUS = get_args(ColabGpu)


class ColabAccount(BaseModel):
    email: str | None = Field(default=None, max_length=254)
    compute_units: float = Field(allow_inf_nan=False)
    usage_rate_hourly: float = Field(ge=0, allow_inf_nan=False)
    active_runtimes: int = Field(ge=0)


class ColabConnectionError(RuntimeError):
    pass


class ColabConnection:
    def __init__(self, profile: Path) -> None:
        self.profile = profile
        self.token_path = profile / "token.json"
        self.pending_path = profile / "pending-runtime.json"
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._cancel = threading.Event()
        self._auth_state = "disconnected"
        self._auth_url: str | None = None
        self._account: dict | None = None
        self._error: str | None = None
        self._job: dict | None = None
        self._workspace: Path | None = None
        self._python = sys.executable
        if self.pending_path.is_file():
            self._job = {
                "state": "cleanup_required",
                "cleanup_required": True,
                "stage": "A previous Colab runtime needs to be released",
            }

    @property
    def supported(self) -> bool:
        return sys.platform in {"linux", "darwin"} and not getattr(sys, "frozen", False)

    def status(self) -> dict:
        with self._lock:
            auth = self._auth_state
            if auth == "disconnected" and self.token_path.is_file():
                auth = "connected"
            return {
                "supported": self.supported,
                "supported_gpus": list(SUPPORTED_COLAB_GPUS),
                "auth_state": auth,
                "auth_url": self._auth_url,
                "account": dict(self._account) if self._account else None,
                "error": self._error,
                "busy": self._thread is not None and self._thread.is_alive(),
                "job": dict(self._job) if self._job else None,
            }

    def _prepare(self) -> None:
        if not self.supported:
            raise ColabConnectionError(
                "Direct Colab connection supports Linux and macOS; use the notebook on Windows"
            )
        self.profile.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.profile.chmod(0o700)

    def _launch(self, target, *args) -> None:
        with self._lock:
            if self.status()["busy"]:
                raise ColabConnectionError("A Colab operation is already running")
            self._cancel.clear()
            self._error = None
            self._thread = threading.Thread(target=target, args=args, daemon=True)
            self._thread.start()

    def _ensure_cli(self) -> str:
        try:
            if importlib.metadata.version("google-colab-cli") == CLI_VERSION:
                return sys.executable
        except importlib.metadata.PackageNotFoundError:
            pass
        environment = self.profile / "tools"
        python = environment / "bin/python"
        if not python.is_file():
            self._process_command([sys.executable, "-m", "venv", str(environment)], timeout=120)
        self._process_command(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                f"google-colab-cli=={CLI_VERSION}",
            ],
            timeout=600,
        )
        return str(python)

    def _process_command(
        self, command, *, login=False, timeout=1800, cleanup=False, account_info=False
    ) -> None:
        if self._cancel.is_set() and not cleanup:
            raise ColabConnectionError("Operation canceled")
        environment = dict(
            os.environ,
            PYTHONUNBUFFERED="1",
            NO_COLOR="1",
            TERM="dumb",
            VISIONLAB_COLAB_PROFILE=str(self.profile),
            VISIONLAB_COLAB_LOGIN="1" if login else "0",
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
            start_new_session=True,
        )
        lines = queue.Queue()

        def read():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        with self._lock:
            self._process = process
        threading.Thread(target=read, daemon=True).start()
        deadline = time.monotonic() + timeout
        try:
            while True:
                if (self._cancel.is_set() and not cleanup) or time.monotonic() > deadline:
                    self._terminate(process)
                    raise ColabConnectionError("Operation canceled or timed out")
                try:
                    line = lines.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    break
                self._event(line, allow_account=account_info)
            if process.wait(timeout=5) != 0:
                raise ColabConnectionError(
                    "Colab could not finish this step; reconnect or try another GPU"
                )
        finally:
            if process.poll() is None:
                self._terminate(process)
            if process.stdin:
                process.stdin.close()
            with self._lock:
                if self._process is process:
                    self._process = None

    @staticmethod
    def _terminate(process) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    def _execute(self, arguments: list[str], **kwargs) -> None:
        bridge = Path(__file__).with_name("colab_cli_bridge.py")
        command = [
            self._python,
            str(bridge),
            "--auth",
            "oauth2",
            "--config",
            str(self.profile / "sessions.json"),
            "--client-oauth-config",
            str(self.profile / "oauth-client.json"),
            *arguments,
        ]
        self._process_command(command, account_info=arguments[0] == "usage", **kwargs)

    def _event(self, line: str, *, allow_account: bool = False) -> None:
        if line.startswith("VISIONLAB_AUTH "):
            if self._auth_state not in {"connecting", "awaiting_code"}:
                return
            value = json.loads(line[len("VISIONLAB_AUTH ") :])
            parsed = urlsplit(value.get("url", ""))
            if (
                parsed.scheme != "https"
                or parsed.hostname != "accounts.google.com"
                or parsed.username
            ):
                return
            with self._lock:
                self._auth_state = "awaiting_code"
                self._auth_url = value["url"]
            webbrowser.open(value["url"])
        elif line.startswith("VISIONLAB_ACCOUNT ") and allow_account:
            account = ColabAccount.model_validate_json(line[len("VISIONLAB_ACCOUNT ") :])
            with self._lock:
                self._account = account.model_dump()
        elif line.startswith("VISIONLAB_PROGRESS "):
            value = json.loads(line[len("VISIONLAB_PROGRESS ") :])
            with self._lock:
                if self._job:
                    self._job["processed"] = max(
                        0, min(int(value["completed"]), self._job["total"])
                    )

    def connect(self) -> None:
        self._prepare()
        self._launch(self._connect_worker)

    def switch_account(self) -> None:
        self._prepare()
        with self._lock:
            if self.status()["busy"] or self.pending_path.exists():
                raise ColabConnectionError("Release VisionLab's runtime before switching accounts")
            self.token_path.unlink(missing_ok=True)
            (self.profile / "sessions.json").unlink(missing_ok=True)
            self._account = None
            self._auth_state = "disconnected"
            self._auth_url = None
            self._job = None
            self._launch(self._connect_worker)

    def refresh_account(self) -> None:
        with self._lock:
            if self.status()["auth_state"] != "connected" or self.status()["busy"]:
                return
            self._prepare()
            self._launch(self._refresh_account_worker)

    def _refresh_account_worker(self) -> None:
        try:
            self._python = self._ensure_cli()
            self._execute(["usage"], timeout=45)
        except Exception:
            with self._lock:
                self._error = (
                    "Could not refresh account details. Retry or reconnect your Google account."
                )

    def _connect_worker(self) -> None:
        try:
            with self._lock:
                self._auth_state = "connecting"
                self._account = None
            self._python = self._ensure_cli()
            self._execute(["usage"], login=True, timeout=300)
            if not self.token_path.is_file():
                raise ColabConnectionError(
                    "Google authorization was not saved; try connecting again"
                )
            self.token_path.chmod(0o600)
            with self._lock:
                self._auth_state = "connected"
                self._auth_url = None
        except Exception:
            with self._lock:
                self._auth_state = "error"
                self._auth_url = None
                self._error = (
                    "Colab sign-in did not complete. Connect again and finish the Google sign-in."
                )

    def authorize(self, code: str) -> None:
        with self._lock:
            if self._auth_state != "awaiting_code" or not self._process or not self._process.stdin:
                raise ColabConnectionError("There is no pending Google sign-in")
            if not code.strip() or len(code) > 4096 or "\n" in code or "\r" in code:
                raise ColabConnectionError("Paste the single authorization code shown by Google")
            self._process.stdin.write(code.strip() + "\n")
            self._process.stdin.flush()
            self._auth_state = "connecting"
            self._auth_url = None

    def disconnect(self) -> None:
        with self._lock:
            if self.status()["busy"] or self.pending_path.exists():
                raise ColabConnectionError(
                    "Stop and release the Colab runtime before disconnecting"
                )
            self.token_path.unlink(missing_ok=True)
            (self.profile / "sessions.json").unlink(missing_ok=True)
            shutil.rmtree(self.profile / "jobs", ignore_errors=True)
            self._auth_state = "disconnected"
            self._account = None
            self._auth_url = None
            self._error = None

    def cancel(self) -> None:
        self._cancel.set()
        if not self.status()["busy"] and self.pending_path.is_file():
            self._launch(self._release_worker)

    def acknowledge_release(self) -> None:
        """Clear recovery state after the user confirms stopping the runtime in Colab."""
        with self._lock:
            if self.status()["busy"]:
                raise ColabConnectionError("Wait for the current operation to finish")
            if not self.pending_path.exists():
                raise ColabConnectionError("There is no runtime awaiting release")
            self.pending_path.unlink()
            self._job.update(
                state="canceled",
                cleanup_required=False,
                stage="Runtime marked released after your confirmation",
            )
            self._error = None

    def _release_worker(self) -> None:
        try:
            self._python = self._ensure_cli()
            session = json.loads(self.pending_path.read_text())["session"]
            self._execute(["stop", "-s", session], timeout=60, cleanup=True)
            self.pending_path.unlink(missing_ok=True)
            with self._lock:
                self._job.update(
                    state="canceled", cleanup_required=False, stage="Colab GPU released"
                )
        except Exception:
            with self._lock:
                self._error = (
                    "Could not confirm the GPU was released. Stop it in Google Colab or retry."
                )

    def start_job(
        self,
        root: Path,
        images: list[Path],
        classes: list[str],
        pipeline: str,
        *,
        confidence: float = 0.35,
        gpu: str = "T4",
    ) -> None:
        self._prepare()
        with self._lock:
            if self.status()["auth_state"] != "connected":
                raise ColabConnectionError("Connect your Google account first")
            if self.pending_path.exists() or self.status()["busy"]:
                raise ColabConnectionError("Finish or stop the current Colab operation first")
            if not images or len(images) > 100 or gpu not in SUPPORTED_COLAB_GPUS:
                raise ColabConnectionError("Choose between 1 and 100 images and a supported GPU")
            config = make_config(classes, pipeline=pipeline, confidence=confidence)
            options = InferenceOptions.from_config(config).model_dump(mode="json")
            self._launch(self._job_worker, root.resolve(), images, options, gpu)

    def _job_worker(self, root: Path, images: list[Path], options: dict, gpu: str) -> None:
        identifier = uuid4().hex
        session = "visionlab-" + identifier[:12]
        workspace = self.profile / "jobs" / identifier
        self._workspace = workspace
        with self._lock:
            self._job = {
                "id": identifier,
                "dataset_id": str(root),
                "state": "running",
                "stage": "Preparing selected images",
                "total": len(images),
                "processed": 0,
                "saved": 0,
                "failed": 0,
                "conflicts": 0,
                "cleanup_required": False,
            }
        attempted = False
        allocated = False
        try:
            workspace.mkdir(parents=True)
            snapshots = self._pack_inputs(root, images, workspace, options, identifier)
            build_source_bundle(workspace / "source.zip")
            (workspace / "runner.py").write_text(bootstrap_source())
            self._python = self._ensure_cli()
            if self._cancel.is_set():
                raise ColabConnectionError("Job canceled before allocating a GPU")
            self._job["stage"] = "Checking your Colab connection"
            self._execute(["usage"], timeout=60)
            self._job["stage"] = f"Starting Colab {gpu} GPU"
            atomic_text(
                self.pending_path,
                json.dumps(
                    {"session": session, "job_id": identifier, "allocation_state": "requested"}
                ),
            )
            attempted = True
            # Let allocation finish before honoring cancellation so its runtime ID is saved.
            self._execute(["new", "-s", session, "--gpu", gpu], timeout=300, cleanup=True)
            allocated = True
            if self._cancel.is_set():
                raise ColabConnectionError("Job canceled after GPU allocation")
            self._job["stage"] = "Transferring images and preparing annotation models"
            self._execute(
                [
                    "upload",
                    "-s",
                    session,
                    str(workspace / "source.zip"),
                    "content/visionlab-source.zip",
                ]
            )
            self._execute(
                [
                    "upload",
                    "-s",
                    session,
                    str(workspace / "input.zip"),
                    "content/visionlab-input.zip",
                ]
            )
            self._job["stage"] = "Running annotation in Colab"
            self._execute(
                ["exec", "-s", session, "-f", str(workspace / "runner.py"), "--timeout", "1800"],
                timeout=1860,
            )
            self._job["stage"] = "Retrieving and saving annotations"
            self._execute(
                [
                    "download",
                    "-s",
                    session,
                    "content/visionlab-job/result.json",
                    str(workspace / "result.json"),
                ],
                timeout=120,
            )
            self._apply_results(root, snapshots, workspace / "result.json", identifier, options)
            self._job["state"] = "completed"
        except Exception:
            self._job["state"] = "canceled" if self._cancel.is_set() else "failed"
            allocation_error = None
            if attempted:
                with contextlib.suppress(OSError, ValueError):
                    record = json.loads(self.pending_path.read_text())
                    if isinstance(record, dict):
                        allocation_error = record.get("allocation_error")
                        allocated = allocated or record.get("allocation_state") == "confirmed"
            precondition_message = (
                f"Colab rejected the {gpu} request (HTTP 412), but did not confirm the cause. "
                "Check the connected Google account and its active Colab runtimes."
            )
            allocation_messages = {
                "gpu_rejected": (
                    f"Colab rejected the {gpu} request for this account. Compute units do not "
                    "guarantee GPU access. Choose another GPU or check this account in Colab."
                ),
                "capacity": precondition_message,
            }
            self._error = (
                "Annotation canceled"
                if self._cancel.is_set()
                else allocation_messages.get(
                    allocation_error,
                    "Colab annotation failed. Check the connection, GPU availability, "
                    "and selected images.",
                )
            )
        finally:
            if attempted:
                self._job["stage"] = "Releasing Colab GPU"
                try:
                    self._execute(["stop", "-s", session], timeout=60, cleanup=True)
                    self.pending_path.unlink(missing_ok=True)
                except Exception:
                    self._job["cleanup_required"] = True
                    self._error = (
                        "Could not confirm the GPU was released. Stop it in Google Colab or retry."
                    )
            self._job["stage"] = (
                "GPU release needs attention"
                if self._job["cleanup_required"]
                else "Finished; Colab GPU released"
                if allocated
                else "No Colab GPU was allocated"
                if attempted
                else "Job did not start"
            )

    def _pack_inputs(self, root, images, workspace, options, identifier):
        repo = AnnotationRepository(root)
        snapshots = {}
        request = {"version": 1, "job_id": identifier, "options": options, "images": []}
        total = 0
        with zipfile.ZipFile(workspace / "input.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for index, image in enumerate(images):
                path = repo.image_path(image)
                if path.stat().st_size > 20 * 1024 * 1024:
                    raise ColabConnectionError("An image exceeds the 20 MiB transfer limit")
                data = path.read_bytes()
                total += len(data)
                if len(data) > 20 * 1024 * 1024 or total > MAX_TRANSFER_BYTES:
                    raise ColabConnectionError(
                        "Reduce the selection; images exceed the transfer limit"
                    )
                with Image.open(io.BytesIO(data)) as picture:
                    size = picture.size
                key = f"image-{index:04d}{path.suffix.lower()}"
                checksum = hashlib.sha256(data).hexdigest()
                snapshots[key] = {
                    "path": path,
                    "sha256": checksum,
                    "size": size,
                    "data": repo.read(path),
                }
                archive.writestr(key, data)
                request["images"].append({"key": key, "sha256": checksum})
            serialized = json.dumps(request)
            archive.writestr("request.json", serialized)
        (workspace / "request.json").write_text(serialized)
        return snapshots

    def _apply_results(self, root, snapshots, path, identifier, options):
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ColabConnectionError("Colab returned too much annotation data")
        payload = json.loads(path.read_text())
        if payload.get("version") != 1 or payload.get("job_id") != identifier:
            raise ColabConnectionError("Colab result does not match this job")
        rows = payload.get("images", [])
        if len(rows) != len(snapshots) or {r["key"] for r in rows} != set(snapshots):
            raise ColabConnectionError("Colab returned an incomplete or duplicated result")
        validated = []
        requested = {c["name"].casefold(): c["name"] for c in options["classes"]}
        for row in rows:
            snapshot = snapshots[row["key"]]
            if row.get("sha256") != snapshot["sha256"]:
                raise ColabConnectionError("Colab input checksum does not match")
            if "error" in row:
                self._job["failed"] += 1
                continue
            result = InferenceResponse.model_validate(row["result"])
            if (result.width, result.height) != snapshot["size"]:
                raise ColabConnectionError("Colab result dimensions do not match")
            if any(d.class_name.casefold() not in requested for d in result.detections):
                raise ColabConnectionError("Colab returned an unrequested class")
            validated.append((snapshot, result))
        repo = AnnotationRepository(root)
        classes = repo.register_classes(list(requested.values()))
        source = (
            "sam2"
            if options["enable_sam2_masks"]
            else "locate_anything"
            if options.get("enable_locate_anything")
            else "grounding_dino"
            if options["enable_grounding_dino"]
            else "yolo"
        )
        for snapshot, result in validated:
            image = snapshot["path"]
            if hashlib.sha256(image.read_bytes()).hexdigest() != snapshot["sha256"]:
                self._job["conflicts"] += 1
                continue
            boxes = list(snapshot["data"]["boxes"])
            for detection in result.detections:
                name = next(c for c in classes if c.casefold() == detection.class_name.casefold())
                left, top, right, bottom = detection.box
                if any(
                    box["class_name"].casefold() == name.casefold()
                    and compute_box_iou(
                        BoundingBox(
                            box["norm_left"], box["norm_top"], box["norm_right"], box["norm_bottom"]
                        ),
                        BoundingBox(left, top, right, bottom),
                    )
                    >= options["box_iou_threshold"]
                    for box in boxes
                ):
                    continue
                boxes.append(
                    {
                        "id": str(uuid4()),
                        "class_name": name,
                        "class_id": classes.index(name),
                        "confidence": detection.confidence,
                        "source": source,
                        "review_status": "pending",
                        "norm_left": left,
                        "norm_top": top,
                        "norm_right": right,
                        "norm_bottom": bottom,
                        "polygon_normalized": detection.polygon or None,
                    }
                )
            try:
                repo.save(
                    image,
                    boxes,
                    status="ai_labeled",
                    expected_revision=snapshot["data"]["revision"],
                )
                self._job["saved"] += 1
                try:
                    index = DatasetIndex(root / ".dataset_index.sqlite")
                    try:
                        index.set_status_and_count(image, "ai_labeled", len(boxes), 0.0)
                    finally:
                        index.close()
                except (sqlite3.Error, OSError):
                    self._error = (
                        "Annotations were saved. Reopen the dataset to refresh its image queue."
                    )
            except RevisionConflict:
                self._job["conflicts"] += 1
