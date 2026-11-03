"""Run from repository root: .venv/bin/python artifacts/perf-m5/bench_stats.py before|after."""
import io
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PIL import Image
from app.api import server
from app.services.dataset.index import DatasetIndex
from app.services.dataset.repository import AnnotationRepository

output = {}
with tempfile.TemporaryDirectory(prefix="anotara-m5-") as temporary:
    for kind in ("legacy", "canonical"):
        root = Path(temporary) / kind
        root.mkdir()
        (root / "data.yaml").write_text("names: [car]\n")
        buffer = io.BytesIO()
        Image.new("RGB", (640, 480), (140, 160, 180)).save(buffer, "JPEG")
        for i in range(1000):
            image = root / f"image-{i:04}.jpg"
            image.write_bytes(buffer.getvalue())
            if kind == "legacy":
                image.with_suffix(".txt").write_text("0 .5 .5 .2 .2\n")
            else:
                path = root / ".visionlab/annotations" / (image.name + ".json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"status": "reviewed", "revision": 1, "boxes": [{
                    "id": "one", "class_name": "car", "class_id": 0,
                    "norm_left": .4, "norm_top": .4, "norm_right": .6, "norm_bottom": .6,
                }]}))
        with DatasetIndex(root / ".dataset_index.sqlite") as index:
            index.scan(root)
            server.ACTIVE_DATASET_DIR, server.DATASET_INDEX = root, index
            times, reads, canonical_reads = [], [], []
            original = AnnotationRepository.read
            read_text = Path.read_text
            for _ in range(6):
                counter = [0, 0]
                def counted(repo, *args, **kwargs):
                    counter[0] += 1
                    return original(repo, *args, **kwargs)
                def counted_text(path, *args, **kwargs):
                    if path.suffix == ".json":
                        counter[1] += 1
                    return read_text(path, *args, **kwargs)
                with patch.object(AnnotationRepository, "read", counted), patch.object(Path, "read_text", counted_text):
                    start = time.perf_counter()
                    result = server.get_dataset_stats()
                    times.append((time.perf_counter() - start) * 1000)
                reads.append(counter[0])
                canonical_reads.append(counter[1])
                assert result["class_counts"] == {"car": 1000}
                assert result["total_annotations"] == result["reviewed"] == result["total_images"] == 1000
            output[kind] = dict(cold_ms=times[0], warm_median_ms=statistics.median(times[1:]),
                times_ms=times, repository_reads=reads, json_reads=canonical_reads)
            server.DATASET_INDEX = None
print(json.dumps(output, indent=2))
Path(__file__).with_name(sys.argv[1] + ".json").write_text(json.dumps(output, indent=2) + "\n")
