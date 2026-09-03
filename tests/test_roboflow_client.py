import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from app.services.annotation.domain import (
    Annotation,
    AnnotationDocument,
    AnnotationSource,
    BoundingBox,
)
from app.services.integrations.roboflow_client import (
    RoboflowClient,
    RoboflowError,
    RoboflowProjectRef,
)


def test_parse_roboflow_urls() -> None:
    # 1. Universe URL with dataset version
    u1 = "https://universe.roboflow.com/roboflow-100/traffic-vehicles/dataset/3"
    ref1 = RoboflowClient.parse_roboflow_url(u1)
    assert ref1 == RoboflowProjectRef(workspace="roboflow-100", project="traffic-vehicles", version=3)

    # 2. Universe URL with model version
    u2 = "https://universe.roboflow.com/deep-vision/pedestrians/model/5"
    ref2 = RoboflowClient.parse_roboflow_url(u2)
    assert ref2 == RoboflowProjectRef(workspace="deep-vision", project="pedestrians", version=5)

    # 3. Roboflow App URL
    u3 = "https://app.roboflow.com/my-team/my-traffic/2"
    ref3 = RoboflowClient.parse_roboflow_url(u3)
    assert ref3 == RoboflowProjectRef(workspace="my-team", project="my-traffic", version=2)

    # 4. Standard slug
    u4 = "city-traffic/signals"
    ref4 = RoboflowClient.parse_roboflow_url(u4)
    assert ref4 == RoboflowProjectRef(workspace="city-traffic", project="signals", version=None)

    # 5. Slug with version
    u5 = "city-traffic/signals/4"
    ref5 = RoboflowClient.parse_roboflow_url(u5)
    assert ref5 == RoboflowProjectRef(workspace="city-traffic", project="signals", version=4)

    # 6. Invalid URL
    with pytest.raises(RoboflowError):
        RoboflowClient.parse_roboflow_url("invalid-url-string")


def test_credentials_save_and_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_creds_file = tmp_path / "creds.json"
    monkeypatch.setattr(
        "app.services.integrations.roboflow_client.CREDENTIALS_FILE", fake_creds_file
    )

    assert RoboflowClient.load_credentials() == {}
    RoboflowClient.save_credentials("rf_test_key_123", "test-ws", "test-proj")

    loaded = RoboflowClient.load_credentials()
    assert loaded["api_key"] == "rf_test_key_123"
    assert loaded["workspace"] == "test-ws"
    assert loaded["project"] == "test-proj"


def test_get_project_versions_mock() -> None:
    client = RoboflowClient(api_key="rf_dummy")
    mock_payload = {
        "project": {
            "versions": [
                {"id": "1", "version": 1, "name": "v1"},
                {"id": "2", "version": 2, "name": "v2"},
            ]
        }
    }
    with patch.object(client, "_http_get_json", return_value=mock_payload):
        versions = client.get_project_versions("ws", "proj")
        assert len(versions) == 2
        assert versions[0]["version"] == 1
        assert versions[1]["version"] == 2


def test_download_and_import_mock(tmp_path: Path) -> None:
    client = RoboflowClient(api_key="rf_dummy")

    # Create dummy zip with data.yaml and 1 image/label
    zip_bytes_io = io.BytesIO()
    with zipfile.ZipFile(zip_bytes_io, "w") as zf:
        data_yaml_content = yaml.dump({"names": ["car", "bus"], "train": "train/images"})
        zf.writestr("data.yaml", data_yaml_content)
        # Create minimal 1x1 jpg header or dummy bytes
        zf.writestr("train/images/car1.jpg", b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb")
        zf.writestr("train/labels/car1.txt", "0 0.5 0.5 0.2 0.2\n")
    zip_bytes = zip_bytes_io.getvalue()

    def mock_download_file(url, dest_path, progress_cb=None):
        dest_path.write_bytes(zip_bytes)

    with (
        patch.object(
            client,
            "_http_get_json",
            return_value={"yolov11": {"link": "https://api.roboflow.com/download.zip"}},
        ),
        patch.object(client, "_download_file", side_effect=mock_download_file),
    ):
        destination = tmp_path / "roboflow_imported"
        result = client.download_and_import("ws", "proj", 1, destination)
        assert result.project_root == destination
        assert len(result.documents) == 1
        assert result.documents[0].annotations[0].class_name == "car"


def test_upload_document_mock(tmp_path: Path) -> None:
    client = RoboflowClient(api_key="rf_dummy")

    img_path = tmp_path / "img_to_upload.jpg"
    img_path.write_bytes(b"dummy_image_data")

    doc = AnnotationDocument(
        image_path=img_path,
        image_width=100,
        image_height=100,
        annotations=(
            Annotation(
                class_name="motorcycle",
                box=BoundingBox(0.1, 0.1, 0.5, 0.5),
                source=AnnotationSource.HUMAN,
            ),
        ),
    )

    posted_urls = []

    def mock_http_post(url, data, headers):
        posted_urls.append(url)
        if "upload?" in url:
            return json.dumps({"id": "uploaded_img_123", "success": True}).encode("utf-8")
        if "annotate/" in url:
            return json.dumps({"success": True}).encode("utf-8")
        return b"{}"

    with patch.object(client, "_http_post", side_effect=mock_http_post):
        resp = client.upload_document(
            workspace="ws",
            project="proj",
            document=doc,
            split="train",
        )
        assert resp["id"] == "uploaded_img_123"
        assert len(posted_urls) == 2
        assert "upload?" in posted_urls[0]
        assert "annotate/uploaded_img_123" in posted_urls[1]
