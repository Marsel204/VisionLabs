"""Roboflow API client for dataset import, export, and annotation synchronization."""

from __future__ import annotations

import base64
import json
import logging
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.export.exporters import CLASS_ORDER
from app.services.annotation.domain import AnnotationDocument
from app.services.dataset.yolo_importer import YoloImporter, YoloImportResult

LOGGER = logging.getLogger(__name__)
ROBOFLOW_API_ROOT = "https://api.roboflow.com"
CREDENTIALS_FILE = Path.home() / ".cache" / "traffic-annotator" / "roboflow_credentials.json"


class RoboflowError(RuntimeError):
    """Raised when a Roboflow API request fails."""


@dataclass(frozen=True, slots=True)
class RoboflowProjectRef:
    """Parsed references for a Roboflow project and version."""

    workspace: str
    project: str
    version: int | None = None


class RoboflowClient:
    """Client for interacting with Roboflow REST API endpoints."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or self.load_credentials().get("api_key", "")

    @staticmethod
    def parse_roboflow_url(input_str: str) -> RoboflowProjectRef:
        """Parse a Roboflow URL, universe link, or workspace/project slug."""
        clean = input_str.strip()
        if not clean:
            raise RoboflowError("Input URL or slug is empty")

        # Match Universe URL e.g. https://universe.roboflow.com/workspace/project/dataset/2 or /model/2
        universe_match = re.search(
            r"universe\.roboflow\.com/([^/]+)/([^/]+)(?:/(?:dataset|model)/(\d+))?",
            clean,
            re.IGNORECASE,
        )
        if universe_match:
            ws, proj, ver = universe_match.groups()
            return RoboflowProjectRef(
                workspace=ws,
                project=proj,
                version=int(ver) if ver else None,
            )

        # Match App URL e.g. https://app.roboflow.com/workspace/project/2 or /browse
        app_match = re.search(
            r"app\.roboflow\.com/([^/]+)/([^/]+)(?:/(\d+))?",
            clean,
            re.IGNORECASE,
        )
        if app_match:
            ws, proj, ver = app_match.groups()
            return RoboflowProjectRef(
                workspace=ws,
                project=proj,
                version=int(ver) if ver and ver.isdigit() else None,
            )

        # Match simple slug: workspace/project or workspace/project/1
        parts = clean.strip("/").split("/")
        if len(parts) >= 2:
            ws = parts[0]
            proj = parts[1]
            ver = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
            return RoboflowProjectRef(workspace=ws, project=proj, version=ver)

        raise RoboflowError(
            f"Could not parse Roboflow reference '{input_str}'. Expected 'workspace/project' or URL."
        )

    def get_project_versions(
        self, workspace: str, project: str, api_key: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch project details and list of available dataset versions."""
        key = api_key or self.api_key
        if not key:
            raise RoboflowError("Roboflow API key is required")

        url = f"{ROBOFLOW_API_ROOT}/{urllib.parse.quote(workspace)}/{urllib.parse.quote(project)}?api_key={key}"
        data = self._http_get_json(url)

        # Check versions list in project payload
        proj_obj = data.get("project", data)
        versions_raw = proj_obj.get("versions", [])
        version_list = []
        if isinstance(versions_raw, list):
            for item in versions_raw:
                if isinstance(item, dict):
                    version_list.append(item)
                elif isinstance(item, (int, str)):
                    version_list.append({"id": str(item), "version": int(item)})
        return version_list

    def download_and_import(
        self,
        workspace: str,
        project: str,
        version: int | str,
        destination: Path,
        api_key: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> YoloImportResult:
        """Download dataset version from Roboflow and import into the destination project."""
        key = api_key or self.api_key
        if not key:
            raise RoboflowError("Roboflow API key is required")

        if progress_callback:
            progress_callback(0.1, f"Fetching export link for {workspace}/{project}/{version}...")

        # Request YOLOv11 export (or YOLOv8 fallback)
        url = (
            f"{ROBOFLOW_API_ROOT}/{urllib.parse.quote(workspace)}/"
            f"{urllib.parse.quote(project)}/{version}/yolov11?api_key={key}"
        )
        try:
            export_data = self._http_get_json(url)
        except RoboflowError:
            # Fallback to yolov8
            url = (
                f"{ROBOFLOW_API_ROOT}/{urllib.parse.quote(workspace)}/"
                f"{urllib.parse.quote(project)}/{version}/yolov8?api_key={key}"
            )
            export_data = self._http_get_json(url)

        # Extract download URL
        download_url = None
        for k in ("yolov11", "yolov8", "export"):
            if isinstance(export_data.get(k), dict) and "link" in export_data[k]:
                download_url = export_data[k]["link"]
                break
        if not download_url and "link" in export_data:
            download_url = export_data["link"]

        if not download_url:
            raise RoboflowError(f"No download link returned by Roboflow for version {version}")

        if progress_callback:
            progress_callback(0.3, "Downloading dataset archive from Roboflow...")

        # Download ZIP to temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_zip = Path(tmp_dir) / "roboflow_dataset.zip"
            self._download_file(download_url, tmp_zip, progress_callback)

            if progress_callback:
                progress_callback(0.7, "Extracting dataset archive...")

            extract_dir = Path(tmp_dir) / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            resolved_extract_dir = extract_dir.resolve()
            with zipfile.ZipFile(tmp_zip, "r") as zf:
                for member in zf.infolist():
                    member_path = (extract_dir / member.filename).resolve()
                    if resolved_extract_dir not in member_path.parents and member_path != resolved_extract_dir:
                        raise RoboflowError(f"Potential zip slip path traversal in archive: {member.filename}")
                zf.extractall(extract_dir)

            # Find data.yaml or dataset.yaml
            yaml_candidates = list(extract_dir.rglob("data.yaml")) + list(
                extract_dir.rglob("dataset.yaml")
            )
            if not yaml_candidates:
                raise RoboflowError("Downloaded archive did not contain a data.yaml or dataset.yaml file")

            yaml_file = yaml_candidates[0]

            if progress_callback:
                progress_callback(0.85, "Importing and cleaning annotations...")

            # Run standard YoloImporter
            importer = YoloImporter()
            result = importer.import_dataset(yaml_file, destination)

            if progress_callback:
                progress_callback(1.0, f"Successfully imported {len(result.documents)} images!")

            return result

    def upload_document(
        self,
        workspace: str,
        project: str,
        document: AnnotationDocument,
        split: str = "train",
        batch_name: str = "TrafficAnnotator-Upload",
        api_key: str | None = None,
    ) -> dict[str, Any]:
        """Upload one image and its annotations to a Roboflow project."""
        key = api_key or self.api_key
        if not key:
            raise RoboflowError("Roboflow API key is required")

        image_bytes = document.image_path.read_bytes()
        image_name = document.image_path.name

        # 1. Upload image
        upload_params = {
            "api_key": key,
            "name": image_name,
            "split": split,
            "batch": batch_name,
        }
        upload_url = f"{ROBOFLOW_API_ROOT}/dataset/{urllib.parse.quote(project)}/upload?{urllib.parse.urlencode(upload_params)}"

        upload_resp = self._http_post(
            upload_url,
            data=image_bytes,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            upload_json = json.loads(upload_resp.decode("utf-8"))
        except Exception:
            upload_json = {}

        image_id = upload_json.get("id")

        # 2. Upload annotations if any and image_id is available
        if image_id and document.annotations:
            lines = []
            for ann in document.annotations:
                if ann.class_name in CLASS_ORDER:
                    cls_idx = CLASS_ORDER.index(ann.class_name)
                    cx, cy, w, h = ann.box.to_yolo()
                    lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if lines:
                label_content = "\n".join(lines) + "\n"
                ann_params = {
                    "api_key": key,
                    "name": f"{document.image_path.stem}.txt",
                }
                ann_url = f"{ROBOFLOW_API_ROOT}/dataset/{urllib.parse.quote(project)}/annotate/{image_id}?{urllib.parse.urlencode(ann_params)}"
                try:
                    self._http_post(
                        ann_url,
                        data=label_content.encode("utf-8"),
                        headers={"Content-Type": "text/plain"},
                    )
                except Exception as err:
                    LOGGER.warning("Could not upload annotations for %s: %s", image_name, err)

        return upload_json

    @staticmethod
    def load_credentials() -> dict[str, str]:
        """Load stored Roboflow credentials from cache."""
        if CREDENTIALS_FILE.is_file():
            try:
                return json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    @staticmethod
    def save_credentials(api_key: str, workspace: str = "", project: str = "") -> None:
        """Persist Roboflow credentials in user cache directory with secure file permissions."""
        import os

        CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "api_key": api_key,
            "workspace": workspace,
            "project": project,
        }
        CREDENTIALS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            os.chmod(CREDENTIALS_FILE, 0o600)
        except OSError:
            pass

    @staticmethod
    def _http_get_json(url: str) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "TrafficAnnotator/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as err:
            err_body = ""
            try:
                err_body = err.read().decode("utf-8")
            except Exception:
                pass
            raise RoboflowError(f"Roboflow API error ({err.code}): {err.reason} - {err_body}") from err
        except Exception as err:
            raise RoboflowError(f"Network error connecting to Roboflow: {err}") from err

    @staticmethod
    def _http_post(url: str, data: bytes, headers: dict[str, str]) -> bytes:
        hdrs = {"User-Agent": "TrafficAnnotator/1.0", **headers}
        req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            err_body = ""
            try:
                err_body = err.read().decode("utf-8")
            except Exception:
                pass
            raise RoboflowError(f"Roboflow API upload error ({err.code}): {err.reason} - {err_body}") from err
        except Exception as err:
            raise RoboflowError(f"Network error during Roboflow upload: {err}") from err

    @staticmethod
    def _download_file(
        url: str,
        destination: Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        req = urllib.request.Request(url, headers={"User-Agent": "TrafficAnnotator/1.0"})
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(req, timeout=120) as resp, open(destination, "wb") as out_file:
            total = resp.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else 0
            downloaded = 0
            chunk_size = 64 * 1024
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_bytes > 0:
                    fraction = 0.3 + (downloaded / total_bytes) * 0.4
                    progress_callback(
                        fraction,
                        f"Downloading: {downloaded / 1024 / 1024:.1f}MB / {total_bytes / 1024 / 1024:.1f}MB",
                    )
