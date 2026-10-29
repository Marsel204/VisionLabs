# Annotate without a local GPU

For existing Google Colab credits, use [Colab annotation](colab-annotation.md).
Connect from the studio to save annotations into your open dataset, or use the
notebook fallback to download results for local review.
The rest of this guide covers live annotation using a hosted worker such as Modal.

Cloud mode runs Quick Detect, Auto-Label previews, and batch inference on a separate
GPU worker. Review, manual editing, canonical annotations, class IDs, dataset files,
and exports stay on your computer. Only selected image bytes and inference options
are sent to the worker; image paths and existing annotations are not uploaded.

The worker runs VisionLab's YOLO, Grounding DINO, Florence-2, Locate Anything 3B, and SAM 2
pipeline. It implements a VisionLab-specific API, not a chat/completions API.
For the new detector, see [Locate Anything setup](locate-anything.md) and update
the worker before enabling it in the studio.
Real GPU execution and model accuracy still require validation after deployment.

## Deploy a worker

The included [Modal](https://modal.com/docs/guide) template uses an L4 GPU, limits
the deployment to one container, persists Hugging Face model weights, and lets the
GPU container stop after idle time. GPU execution, container startup/idle time,
and persistent storage can incur charges.

From the repository root, authenticate:

```sh
uvx --from modal modal setup
```

In the [Modal Secrets dashboard](https://modal.com/secrets), create a secret named
`visionlab-cloud` containing a private `VISIONLAB_CLOUD_TOKEN`. Use the same token
in your laptop configuration. Keep it out of frontend code and Git.

Deploy when ready to create cloud resources:

```sh
uvx --from modal modal deploy deploy/modal_worker.py
```

Modal prints the function's HTTPS URL. Append `/v1/infer` for the laptop's
`VISIONLAB_CLOUD_URL`. The first request downloads the selected model weights;
subsequent requests can reuse loaded models and cached weights. A request after an
idle period may need a cold start. Use a longer client timeout for first use.

Alternatively, install the regular project dependencies on a GPU server, supply
`VISIONLAB_CLOUD_TOKEN` through its environment/secret manager, and start:

```sh
uv run uvicorn app.api.cloud_worker:create_app --factory --host 127.0.0.1 --port 9000
```

Place that inference-only worker behind an HTTPS reverse proxy. Keep the existing
desktop `app.api.server` local: it provides folder selection and dataset management.
The cloud worker requires a bearer token for every inference request and has no
dataset browsing or saving endpoints.

## Install the lightweight laptop backend

This install needs no PyTorch, CUDA, Transformers, Ultralytics, or Qt.
On a new checkout, with Python 3.12 or later:

```sh
uv venv --python 3.12
uv pip install -r requirements-cloud.txt
```

Reuse an existing `.venv` if you already have one. Do not run `uv sync` for the
lightweight install: the regular project dependencies include the local AI stack.

Copy `.env.example` to `.env` and configure:

```dotenv
VISIONLAB_INFERENCE_BACKEND=cloud
VISIONLAB_CLOUD_URL=https://YOUR-WORKER-HOST/v1/infer
VISIONLAB_CLOUD_TOKEN=YOUR_PRIVATE_SHARED_TOKEN
VISIONLAB_CLOUD_TIMEOUT=600
```

The URL and token above are placeholders. Use your deployed URL and worker secret.
Do not use a `VITE_` variable for the token; only the Python backend needs it.

Start the local API explicitly so `.env` is loaded. On Linux/macOS:

```sh
.venv/bin/python -m uvicorn app.api.server:app --env-file .env --host 127.0.0.1 --port 8765
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.server:app --env-file .env --host 127.0.0.1 --port 8765
```

Keep the backend running. In a second terminal:

```sh
cd desktop
npm ci
npm run dev
```

Open `http://localhost:1420`. The header shows **Cloud inference / Configured**.
This means configuration is loaded, not that a cloud request has succeeded. Select
a dataset and try one Auto-Label preview before starting a batch. With an existing
Tauri development installation, `npm run tauri dev` can replace the browser studio
after the local backend is running.

## Behavior and limits

- Cloud mode skips local model warm-up and automatic local inference fallback.
  Connection/authentication/model errors are reported; saved annotations remain intact.
- Batch inference uses the same client. Completed images remain saved, and per-image
  failures appear in the existing batch job status.
- A genuine empty cloud response stays empty and does not trigger a local download.
- Results must match the image dimensions, requested classes, normalized box/polygon
  geometry, and finite confidence scores before they can become annotations.
- Image files are limited to 20 MiB and 40 megapixels; responses are limited to 4 MiB.
  Worker image files are temporary and removed after completed requests. Provider
  infrastructure and logging policies also apply.
- Quick Detect uses the dataset's classes unless an explicit class filter is supplied.
  Use Auto-Label for text-prompted detection of other classes.
- Laptop weight paths and arbitrary model URLs cannot select worker models. This
  version accepts built-in YOLO11, YOLOv8, and YOLO26 checkpoint names. Custom weights
  require a separately configured, trusted worker deployment.
- Requests are not automatically retried, because that could repeat billable work.
  Check the worker after a timeout, then explicitly retry if needed.
- To restore local CPU/GPU inference, install the regular project dependencies, set
  `VISIONLAB_INFERENCE_BACKEND=local`, and restart the local backend.

## Verify without a GPU or cloud account

The contract tests use synthetic images and an in-process worker with deterministic
model doubles. They exercise authentication, payloads, invalid geometry, connection
failures, Quick Detect, preview, and batch persistence:

```sh
uv run --no-project --python 3.12 --with pytest --with httpx --with fastapi --with python-multipart --with numpy --with pillow --with PyYAML python -m pytest --noconftest tests/test_cloud_inference.py -q
```

Full project checks also require the regular development/Qt dependencies. These
tests do not validate a paid deployment or real model accuracy.
