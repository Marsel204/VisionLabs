"""Tests for FastAPI backend endpoints."""

from fastapi.testclient import TestClient
from app.api.server import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert "classes" in data
    assert "motorcycle" in data["classes"]
    assert "database_connected" in data
    assert data["database_connected"] is True


def test_images_list():
    res = client.get("/api/images")
    assert res.status_code == 200
    data = res.json()
    assert "images" in data
    assert data["total"] >= 1
    first = data["images"][0]
    assert "filename" in first
    assert "width" in first
    assert "height" in first
    assert "status" in first


def test_prompt_auto_refine():
    res = client.post("/api/prompt/auto-refine", json={"class_name": "motorcycle"})
    assert res.status_code == 200
    data = res.json()
    assert "refined_prompt" in data
    assert "motorcycle" in data["refined_prompt"].lower()


def test_yolo_detection():
    res_imgs = client.get("/api/images")
    img_name = res_imgs.json()["images"][0]["filename"]

    res = client.post("/api/detect/yolo", json={"image_name": img_name, "conf_threshold": 0.25})
    assert res.status_code == 200
    data = res.json()
    assert data["image_name"] == img_name
    assert "boxes" in data
    assert data["count"] > 0


def test_dataset_stats():
    res = client.get("/api/dataset/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_images" in data
    assert "reviewed" in data
    assert "unreviewed" in data
    assert "class_counts" in data
    assert "database_file" in data


def test_save_annotations_updates_sqlite_and_difficulty():
    res_imgs = client.get("/api/images")
    img_name = res_imgs.json()["images"][0]["filename"]

    boxes_payload = [
        {
            "id": "box-1",
            "class_name": "car",
            "class_id": 1,
            "confidence": 0.95,
            "x": 100.0,
            "y": 100.0,
            "width": 200.0,
            "height": 150.0,
            "norm_left": 0.1,
            "norm_top": 0.1,
            "norm_right": 0.3,
            "norm_bottom": 0.25,
            "occluded": False,
            "truncated": False,
            "source": "human",
        }
    ]

    res = client.post(f"/api/annotations/{img_name}", json={"image_name": img_name, "boxes": boxes_payload})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "saved"
    assert data["review_status"] == "reviewed"
    assert "difficulty" in data

    # Verify retrieval
    res_get = client.get(f"/api/annotations/{img_name}")
    assert res_get.status_code == 200
    assert len(res_get.json()["boxes"]) == 1


def test_autolabel_preview():
    res = client.post("/api/autolabel/preview", json={
        "classes": [{"name": "car", "prompt": "passenger car", "color": "#06b6d4", "enabled": True}],
        "confidence_threshold": 0.3,
    })
    assert res.status_code == 200
    data = res.json()
    assert "image_name" in data
    assert "detections" in data
    assert "count" in data


def test_dataset_export():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = client.post("/api/dataset/export", json={
            "format": "yolo",
            "train_ratio": 0.70,
            "val_ratio": 0.20,
            "test_ratio": 0.10,
            "output_dir": tmp_dir,
        })
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["format"] == "yolo"
        assert "splits" in data


def test_active_learning_queue():
    res = client.get("/api/active-learning/queue?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert "queue" in data
    assert "count" in data


def test_model_presets():
    res = client.get("/api/models/presets")
    assert res.status_code == 200
    data = res.json()
    assert "presets" in data
    assert len(data["presets"]) >= 3
    preset_ids = [p["id"] for p in data["presets"]]
    assert "yolo11n.pt" in preset_ids


def test_model_browse_weights():
    res = client.post("/api/models/browse-weights")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data


def test_model_validate_nonexistent():
    res = client.post("/api/models/validate", json={"path": "/nonexistent/model.pt"})
    assert res.status_code == 400


def test_yolo_ensemble_detection():
    res_imgs = client.get("/api/images")
    img_name = res_imgs.json()["images"][0]["filename"]

    res = client.post("/api/detect/yolo", json={
        "image_name": img_name,
        "conf_threshold": 0.25,
        "models": ["yolo11n.pt"],
    })
    assert res.status_code == 200
    data = res.json()
    assert data["image_name"] == img_name
    assert "boxes" in data
    assert data["count"] > 0


def test_autolabel_preview_with_florence_and_yolo_ensemble():
    res = client.post("/api/autolabel/preview", json={
        "classes": [{"name": "car", "prompt": "passenger car", "color": "#06b6d4", "enabled": True}],
        "confidence_threshold": 0.3,
        "enable_grounding_dino": True,
        "enable_yolo": True,
        "yolo_models": ["yolo11n.pt"],
        "enable_florence2": True,
        "enable_florence2_verifier": True,
        "enable_sam2_masks": False,
    })
    assert res.status_code == 200
    data = res.json()
    assert "image_name" in data
    assert "detections" in data

