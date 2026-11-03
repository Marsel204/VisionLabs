import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import server as s
from app.services.dataset.annotation_import import (
    CocoFormatError,
    ImportCounts,
    build_boxes,
    describe_source,
    find_coco_files,
    import_coco,
    read_coco,
)
from app.services.dataset.index import DatasetIndex
from app.services.dataset.repository import AnnotationRepository, RevisionConflict

CATEGORIES = [{"id": 1, "name": "motorcycle"}, {"id": 2, "name": "car"}]


def make_image(path: Path, size=(100, 80)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (90, 120, 160)).save(path)
    return path


def ann(ann_id, image_id, category_id, bbox, **extra):
    return {"id": ann_id, "image_id": image_id, "category_id": category_id, "bbox": bbox, **extra}


def write_coco(path: Path, images, annotations, categories=CATEGORIES) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"images": images, "annotations": annotations, "categories": categories}))
    return path


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "Final"
    make_image(root / "train" / "images" / "a.jpg")
    make_image(root / "train" / "images" / "b.jpg")
    coco = write_coco(
        root / "train" / "annotations.json",
        [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 80},
         {"id": 2, "file_name": "b.jpg", "width": 100, "height": 80},
         {"id": 3, "file_name": "gone.jpg", "width": 100, "height": 80}],
        [ann(10, 1, 1, [10, 8, 20, 16], occluded=True),
         ann(11, 1, 2, [50, 40, 30, 20], truncated=True),
         ann(12, 2, 1, [0, 0, 100, 80]),
         ann(13, 3, 1, [1, 1, 5, 5]),
         ann(14, 99, 1, [1, 1, 5, 5])],
    )
    return root, coco


def test_find_coco_files_ignores_hidden_and_output_folders(dataset):
    root, coco = dataset
    (root / ".visionlab" / "annotations").mkdir(parents=True)
    (root / ".visionlab" / "annotations" / "annotations.json").write_text("{}")
    (root / "exports" / "x").mkdir(parents=True)
    (root / "exports" / "x" / "annotations.json").write_text("{}")
    (root / "train" / "notes.json").write_text("{}")
    assert find_coco_files(root) == [coco.resolve()]


def test_describe_source_counts_images_and_classes(dataset):
    _, coco = dataset
    source = describe_source(coco)
    assert (source.images, source.annotations, source.found_images) == (3, 5, 2)
    assert source.categories == ("motorcycle", "car")
    assert source.image_root == coco.parent / "images"


def test_non_coco_json_is_rejected(tmp_path):
    bad = tmp_path / "annotations.json"
    bad.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(CocoFormatError):
        read_coco(bad)


def test_import_saves_boxes_in_place_as_unreviewed_with_flags(dataset):
    root, coco = dataset
    repo = AnnotationRepository(root)
    counts = import_coco(repo, coco)
    assert (counts.imported, counts.boxes, counts.missing_images, counts.orphan_boxes) == (
        2, 3, 1, 1
    )
    data = repo.read(root / "train" / "images" / "a.jpg")
    assert data["status"] == "unreviewed"
    motorcycle, car = data["boxes"]
    assert (motorcycle["class_name"], motorcycle["occluded"], motorcycle["truncated"]) == (
        "motorcycle", True, False)
    assert (car["class_name"], car["truncated"]) == ("car", True)
    assert motorcycle["norm_left"] == pytest.approx(0.10)
    assert motorcycle["norm_right"] == pytest.approx(0.30)
    assert motorcycle["norm_bottom"] == pytest.approx(0.30)
    assert motorcycle["source"] == "human" and motorcycle["confidence"] == 1.0
    assert not (root / "exports").exists()  # nothing was copied


def test_overlapping_same_class_boxes_are_all_kept(tmp_path):
    root = tmp_path / "d"
    make_image(root / "images" / "x.jpg")
    coco = write_coco(
        root / "annotations.json",
        [{"id": 1, "file_name": "x.jpg", "width": 100, "height": 80}],
        [ann(1, 1, 1, [10, 10, 60, 60]), ann(2, 1, 1, [15, 12, 20, 25]),
         ann(3, 1, 1, [11, 10, 60, 60])],
    )
    repo = AnnotationRepository(root)
    counts = import_coco(repo, coco)
    assert counts.boxes == 3  # an occluded neighbour inside a larger box is a real label
    assert len(repo.read(root / "images" / "x.jpg")["boxes"]) == 3


def test_class_names_match_case_insensitively_by_alias_or_are_registered(tmp_path):
    root = tmp_path / "d"
    make_image(root / "images" / "x.jpg")
    categories = [
        {"id": 1, "name": "Motorbike"}, {"id": 2, "name": "CAR"}, {"id": 3, "name": "Tuktuk"}
    ]
    coco = write_coco(
        root / "annotations.json",
        [{"id": 1, "file_name": "x.jpg", "width": 100, "height": 80}],
        [
            ann(1, 1, 1, [1, 1, 10, 10]), ann(2, 1, 2, [20, 20, 10, 10]),
            ann(3, 1, 3, [40, 40, 10, 10]),
        ],
        categories,
    )
    repo = AnnotationRepository(root)
    counts = import_coco(repo, coco)
    names = [b["class_name"] for b in repo.read(root / "images" / "x.jpg")["boxes"]]
    assert names == ["motorcycle", "car", "Tuktuk"]
    assert counts.classes_added == ["Tuktuk"]
    assert "Tuktuk" in repo.classes()


def test_declared_image_size_rescales_coordinates(tmp_path):
    root = tmp_path / "d"
    make_image(root / "images" / "x.jpg", (100, 80))
    coco = write_coco(
        root / "annotations.json",
        [{"id": 1, "file_name": "x.jpg", "width": 200, "height": 160}],  # exported at 2x
        [ann(1, 1, 1, [20, 16, 40, 32])],
    )
    repo = AnnotationRepository(root)
    import_coco(repo, coco)
    box = repo.read(root / "images" / "x.jpg")["boxes"][0]
    coordinates = (box["norm_left"], box["norm_top"], box["norm_right"], box["norm_bottom"])
    assert coordinates == pytest.approx((0.10, 0.10, 0.30, 0.30))


def test_invalid_and_out_of_bounds_boxes():
    classes = {1: "car"}
    boxes, invalid = build_boxes(
        [ann(1, 1, 1, [10, 10, 0, 5]),            # no width
         ann(2, 1, 1, [-5, -5, 20, 20]),          # spills over the corner: clamped
         ann(3, 1, 9, [1, 1, 5, 5]),              # unknown category
         {"id": 4, "image_id": 1, "category_id": 1}],  # no bbox
        classes, lambda name: name, (100, 80))
    assert invalid == 3 and len(boxes) == 1
    assert (boxes[0]["norm_left"], boxes[0]["norm_top"]) == (0.0, 0.0)


def test_segmentation_polygon_is_kept():
    boxes, _ = build_boxes(
        [ann(1, 1, 1, [10, 10, 20, 20], segmentation=[[10, 10, 30, 10, 30, 30, 10, 30]])],
        {1: "car"}, lambda name: name, (100, 100))
    assert boxes[0]["polygon_normalized"][0] == pytest.approx([0.1, 0.1])


def test_existing_labels_are_kept_unless_overwrite(dataset):
    root, coco = dataset
    repo = AnnotationRepository(root)
    image = root / "train" / "images" / "a.jpg"
    repo.save(image, [], status="reviewed")
    counts = import_coco(repo, coco)
    assert counts.skipped_existing == 1 and counts.imported == 1
    assert repo.read(image)["status"] == "reviewed" and repo.read(image)["boxes"] == []
    counts = import_coco(repo, coco, overwrite=True)
    assert counts.skipped_existing == 0 and counts.imported == 2
    assert len(repo.read(image)["boxes"]) == 2


def test_image_edited_during_import_is_not_overwritten(dataset, monkeypatch):
    root, coco = dataset
    repo = AnnotationRepository(root)
    original = repo.save

    def racing_save(image, boxes, **kwargs):
        # Someone saves the image between the import's check and its write.
        original(image, [], status="reviewed")
        raise RevisionConflict("document changed; reload before saving")

    monkeypatch.setattr(repo, "save", racing_save)
    counts = import_coco(repo, coco)
    assert counts.skipped_conflict == 2 and counts.imported == 0


def test_cancel_stops_the_import(dataset):
    root, coco = dataset
    calls = iter([False, True])
    counts = import_coco(AnnotationRepository(root), coco, cancelled=lambda: next(calls, True))
    assert counts.imported == 1


def test_missing_file_is_found_by_unique_name_elsewhere(tmp_path):
    root = tmp_path / "d"
    make_image(root / "pics" / "deep" / "x.jpg")
    coco = write_coco(
        root / "annotations.json",
        [{"id": 1, "file_name": "x.jpg", "width": 100, "height": 80}],
        [ann(1, 1, 1, [1, 1, 10, 10])],
    )
    found = root / "pics" / "deep" / "x.jpg"
    counts = import_coco(AnnotationRepository(root), coco,
                         locate_fallback=lambda name: found if name == "x.jpg" else None)
    assert counts.imported == 1 and counts.missing_images == 0


# ------------------------------------------------------------------ HTTP endpoints


@pytest.fixture
def client(dataset, monkeypatch):
    root, _ = dataset
    index = DatasetIndex(root / ".dataset_index.sqlite")
    index.scan(root)
    monkeypatch.setattr(s, "ACTIVE_DATASET_DIR", root.resolve())
    monkeypatch.setattr(s, "DATASET_INDEX", index)
    monkeypatch.setattr(s, "persist_active_dataset_dir", lambda _: None)
    monkeypatch.setattr(s, "get_yolo_model", lambda: None)
    yield TestClient(s.app, headers={"X-VisionLab-Token": s.SESSION_TOKEN})
    index.close()


def wait_for_import(client, timeout=20):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/api/dataset/import-annotations/status").json()
        if status.get("completed"):
            return status
        time.sleep(0.05)
    raise AssertionError("import did not finish")


def test_sources_endpoint_lists_the_coco_file_with_counts(client):
    body = client.get("/api/dataset/annotation-sources").json()
    [source] = body["sources"]
    assert source["relative"] == "train/annotations.json"
    assert (source["images"], source["found_images"], source["annotations"]) == (3, 2, 5)
    assert source["image_dir"] == "train/images"


def test_import_endpoint_updates_the_index_so_the_lists_show_the_labels(client):
    started = client.post(
        "/api/dataset/import-annotations", json={"sources": ["train/annotations.json"]}
    )
    assert started.status_code == 200, started.text
    status = wait_for_import(client)
    assert status["error"] is None
    assert (status["imported"], status["boxes"], status["missing_images"]) == (2, 3, 1)
    images = {i["filename"]: i for i in client.get("/api/images").json()["images"]}
    assert images["a.jpg"]["annotation_count"] == 2 and images["a.jpg"]["status"] == "unreviewed"
    assert images["b.jpg"]["annotation_count"] == 1
    stats = client.get("/api/dataset/stats").json()
    assert stats["class_counts"]["motorcycle"] == 2 and stats["class_counts"]["car"] == 1
    assert stats["total_annotations"] == 3 and stats["reviewed"] == 0
    # The review page reads the same boxes back through the normal annotation endpoint.
    shown = client.get("/api/annotations/train/images/a.jpg").json()
    assert [b["class_name"] for b in shown["boxes"]] == ["motorcycle", "car"]


def test_import_rejects_bad_paths_and_non_coco_files(client, dataset):
    root, _ = dataset
    response = client.post("/api/dataset/import-annotations", json={"sources": ["nope.json"]})
    assert response.status_code == 422
    (root / "notes.json").write_text(json.dumps({"x": 1}))
    response = client.post("/api/dataset/import-annotations", json={"sources": ["notes.json"]})
    assert response.status_code == 422
    assert client.post("/api/dataset/import-annotations", json={"sources": []}).status_code == 422


def test_a_second_import_while_one_runs_is_refused(client, monkeypatch):
    monkeypatch.setitem(s.IMPORT_STATUS, "running", True)
    res = client.post(
        "/api/dataset/import-annotations", json={"sources": ["train/annotations.json"]}
    )
    assert res.status_code == 409


def test_batch_auto_label_skips_images_that_already_have_boxes(client):
    client.post("/api/dataset/import-annotations", json={"sources": ["train/annotations.json"]})
    wait_for_import(client)
    repo = AnnotationRepository(s.ACTIVE_DATASET_DIR)
    index = s.get_dataset_index()
    records = {Path(r["path"]).name: r for r in s.all_records(index)}
    assert s._already_annotated(repo, index, records["a.jpg"], Path(records["a.jpg"]["path"]))
    make_image(s.ACTIVE_DATASET_DIR / "train" / "images" / "empty.jpg")
    index.scan(s.ACTIVE_DATASET_DIR)
    fresh = {Path(r["path"]).name: r for r in s.all_records(index)}
    record = fresh["empty.jpg"]
    assert not s._already_annotated(repo, index, record, Path(record["path"]))


def test_counts_dataclass_keeps_only_a_few_failures():
    counts = ImportCounts()
    for n in range(50):
        counts.fail(f"{n}.jpg", "boom")
    assert len(counts.failures) == 20
