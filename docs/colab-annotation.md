# Annotate with existing Google Colab credits

## Connect from VisionLab

On Linux and macOS, open your dataset and click **Colab** in the studio header:

1. Click **Sign in with Google**. VisionLab prepares Google's official Colab CLI
   automatically and opens Google sign-in in your browser.
2. Sign in to the account with your Colab credits. Copy the authorization code Google
   displays, paste it into VisionLab, and click **Finish connection**. Google's current
   CLI flow requires this one-time paste; it does not return directly to the app.
3. Enter the classes to find. Leave **Images this run = 3** for your first run.
4. Click **Start annotation**. VisionLab starts a Colab GPU, transfers those images,
   installs the models, runs annotation, saves the labels in your open dataset, and
   releases the GPU. Open the resulting images in the studio to review their boxes
   and masks.

Signing in does not allocate a GPU. Starting annotation uses your Colab compute
units; GPU availability and cost depend on your Google account. Model setup takes
time on the first run. No Modal account, endpoint, tunnel, or API key is needed.

The dialog shows the signed-in Google email, compute-unit balance, and active
runtime count. Opening it refreshes these details without allocating hardware;
**Refresh account details** checks them again. If the email is your free account
and the credits are on another account, click **Switch account**, select the
credited account in Google's account chooser, and paste the new authorization code.
Switching removes only VisionLab's cached authorization; it keeps local annotations
and transfer files and does not stop notebooks opened outside VisionLab. An unfinished
VisionLab runtime must be released before switching, so cleanup retains its credentials.

The default is Grounding DINO with SAM 2 masks on a T4. Under **More options →
Colab GPU**, choose T4, L4, A100, H100, or G4. These are the GPU request types
supported by the pinned official CLI, not a guarantee of available hardware on
your account. Compute units do not guarantee access to a particular GPU; Colab
also applies account permissions, usage limits, and current capacity. VisionLab
passes your selected GPU to Colab without substituting another type. If Colab
rejects allocation, the app displays the rejection. HTTP 412 does not establish
which limit or precondition failed; check the displayed Google account and active
runtimes before assuming your credited account is out of compute units.

The **Output** menu also offers **Locate Anything 3B boxes** and **Locate Anything
3B boxes and masks**. These use NVIDIA's natural-language grounding model; see
[setup and model terms](locate-anything.md). The notebook and CLI expose the same
choices as `locate-boxes` and `locate-masks`.

Jobs select the next matching
images in dataset order and skip existing labels, including reviewed empty images.
Each run is limited to 100 images, 50 MiB total, and 20 MiB per image. The remote
script has a 30-minute limit. Use smaller batches if setup and inference take longer.
Only the chosen images and inference source are uploaded; Google credentials,
dataset paths, and existing annotations are excluded from the upload.

You can close the dialog while annotation continues. Keep VisionLab running to
receive the results. **Stop annotation** cancels the work and releases its runtime;
normal app shutdown also attempts cleanup. If release cannot be confirmed, VisionLab
shows **Release GPU** and blocks another job until cleanup. It remembers unfinished
cleanup after a restart. You can also stop the runtime in Google Colab, then click
**I've stopped it in Colab** to acknowledge that manual action.

Returned labels are validated against the input image hashes and dimensions before
saving. Changes made to an image or its annotations during inference are preserved
and reported as conflicts. Existing overlapping boxes are retained without duplicates.
The image queue refreshes when a job finishes; annotations still need human review.

Google authorization is kept in a private VisionLab profile under the app cache
directory, separate from any other Colab CLI installation. The one-time code is not
saved or logged. **Disconnect** removes VisionLab's local authorization and transfer
files after its runtime has been released. To revoke Google's grant as well, use
your Google account's connected-app settings.

Direct connection uses the pinned official `google-colab-cli==0.7.5`, whose supported
platforms are Linux and macOS. Frozen Python backends and Windows use the notebook
fallback below; the Colab dialog provides download buttons for both required files.
Browser authorization has been checked interactively. A completed annotation job on
a real Colab GPU, including use of paid compute units, remains unverified. See the
[official Colab CLI](https://github.com/googlecolab/google-colab-cli) and
[Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Notebook fallback

Use [the Colab batch notebook](../notebooks/visionlab_colab_batch.ipynb) to generate
annotations on a Colab GPU, then review them in VisionLab on your laptop. This
fallback works without connecting the studio to Colab. The notebook does not start
or connect to your Colab account by itself.

The default example downloads **exactly three JPEGs** from
[Marsel204/Dataset](https://github.com/Marsel204/Dataset) at revision
`ed87d7c66396c24cab31023b4c76a4e7f964bd62`. It verifies their Git blob hashes and
does not clone the dataset or download its large `annotations.json` file.

## Prepare the files

From this VisionLab checkout:

```sh
python scripts/package_colab.py
```

This creates `artifacts/colab/visionlab-colab-source.zip`, containing inference Python
source, its checksums, and `requirements-colab.txt`. It excludes `.env` files,
datasets, model weights, and desktop binaries. Use this bundle rather than cloning
`main` in Colab: changes in this checkout might not have been published yet.

Download both that source ZIP and `notebooks/visionlab_colab_batch.ipynb` to your
computer. Open [Colab](https://colab.research.google.com/), select **File → Upload
notebook**, and upload the notebook.

## Run a small batch

1. Select **Runtime → Change runtime type → GPU**. Use a Python 3.12 or newer runtime.
2. Run the GPU check, upload the source ZIP when prompted, and install dependencies.
   The notebook retains Colab's existing matching CUDA-enabled PyTorch/torchvision pair.
3. Leave the sample selected and `LIMIT = 3` for the first run. The default pipeline
   is Grounding DINO + SAM 2 masks; YOLO box and mask presets are also available.
4. Set the class names and optional JSON prompt descriptions. Example:
   `{"motorcycle": "motorcycle, scooter, motorbike"}`.
5. Keep **Save to Drive** enabled to retain checkpoints after a disconnect. Google
   asks you to authorize Drive access in the notebook. Give the run a descriptive name.
6. Run inference, inspect the per-image status and errors, then download the results.

For your own images, choose **Upload image ZIP**. The input cell extracts only the
first `LIMIT` supported images, in filename order; it does not unpack the rest of
the dataset or its labels. Increase `LIMIT` deliberately when ready for a larger
batch. Image ZIP entries are limited to 20 MiB each and must have safe relative paths.

## Review in VisionLab

The default **visionlab** download retains images, `data.yaml`, derived YOLO labels,
and canonical `.visionlab` annotations. Extract it on your laptop and select the
extracted folder as the active dataset. Masks, confidence, class IDs, and pending
review information are retained. Do not import just the image files: that omits
the canonical annotations.

Other downloads:

- **yolo:** a `data.yaml`, `images/`, detection `labels/`, and annotation metadata.
  The `.txt` projection contains boxes; polygons remain in metadata.
- **coco:** `annotations.json` with polygon segmentation and `images/`. This checkout's
  COCO importer retains polygon masks. For multipart polygons it uses the largest
  outer contour; the editor does not represent RLE masks or holes.

Colab batch review uses local dataset files, so a live cloud endpoint is unnecessary.
If `.env` was configured for a remote worker earlier, use
`VISIONLAB_INFERENCE_BACKEND=local` and restart VisionLab for local review. A local
GPU is unnecessary for opening, editing, and exporting the annotations. Use the
notebook for further AI annotation when your laptop lacks the local model runtime.

## Resume and recovery

Each completed image is saved before the next image starts. Rerun with the same
input images, run name, and configuration to skip saved images and retry failures.
Existing canonical annotations, including human edits and genuine empty results,
are never overwritten by resume. Errors are kept in `batch-report.json` and the
checkpoint manifest under `.visionlab/colab-batch.json`.

Changing classes, prompts, model selection, confidence, or inference source code
requires a new run/output folder. A changed source image cannot replace an existing
saved image. Do not run two notebooks against the same output folder concurrently.

Drive checkpoints survive runtime deletion. If interrupted while the runtime is
still alive, the final download cell can package the images completed so far.
Without Drive, download results before ending the session; `/content` is temporary.

Stop the GPU runtime using **Runtime → Disconnect and delete runtime** when finished.
Compute availability, idle limits, and session duration depend on your Colab plan
and remaining compute units. See the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Local verification

No cloud credentials or compute credits are needed for the contract tests:

```sh
.venv/bin/python -m pytest tests/test_colab_batch.py tests/test_coco_importer.py tests/test_exporters.py -q
```

The command-line entry point uses the same batch service:

```sh
python -m app.services.colab_batch --images /path/to/images --output /path/to/new-run --classes motorcycle car person --pipeline dino-masks --device cuda --limit 3
```

Use `--device cpu` for a small CPU inference check with the model dependencies
installed. The checked-in notebook starts with a GPU check and is intended to run
on your own Colab account; no Colab GPU session is started by these local changes.
