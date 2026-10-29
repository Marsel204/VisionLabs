"""Deploy from the repository root: modal deploy deploy/modal_worker.py.

Create the Modal secret 'visionlab-cloud' containing VISIONLAB_CLOUD_TOKEN first.
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("visionlab-cloud")
cache = modal.Volume.from_name("visionlab-model-cache", create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(ROOT / "requirements-cloud.txt"))
    .pip_install(
        "torch>=2.4",
        "torchvision",
        "transformers>=4.57.1,<4.58",
        "ultralytics>=8.3",
        "opencv-python-headless>=4.10",
        "einops",
        "timm",
        "peft>=0.17",
        "decord>=0.6.0",
        "lmdb>=1.7.5",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "YOLO_CONFIG_DIR": "/cache/ultralytics",
            "PYTHONPATH": "/root",
        }
    )
    .add_local_dir(str(ROOT / "app"), "/root/app")
    .add_local_dir(str(ROOT / "src"), "/root/src")
    .add_local_file(str(ROOT / "pipeline_bridge.py"), "/root/pipeline_bridge.py")
)


@app.function(
    image=image,
    gpu="L4",
    timeout=600,
    scaledown_window=60,
    max_containers=1,
    secrets=[modal.Secret.from_name("visionlab-cloud")],
    volumes={"/cache": cache},
)
@modal.asgi_app()
def worker():
    from app.api.cloud_worker import create_app

    return create_app()
