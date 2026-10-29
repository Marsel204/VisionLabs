# Locate Anything 3B

VisionLab supports NVIDIA's [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)
for object localization from natural-language class descriptions. It can generate
boxes on its own, join the existing detector ensemble, or provide boxes for SAM 2
polygon masks.

## Use in the studio

1. Open **Auto-Label AI** and enable **Locate Anything 3B** under object detectors.
2. Enter your target classes and visual descriptions. Disable Grounding DINO,
   YOLO, and Florence-2 detection to run Locate Anything alone.
3. Enable **SAM 2** for masks, or disable it for boxes only. Florence-2 semantic
   verification is an optional additional model.
4. Run a test preview, then start the batch after inspecting the results.

The Qt interface also offers Locate Anything in its model dropdown and detector
checkboxes. Existing model defaults are unchanged.

The first selected run downloads the weights (approximately 7.7 GB) and the
checkpoint's custom Transformers code from Hugging Face. VisionLab pins both to
revision `c32291ca5e996f5a7a485845b4f57a233936bba0`. Later runs reuse the model
and Hugging Face cache. No separate inference-provider API key is required.

Use the normal launcher or `uv sync` to install the dependencies. This integration
uses Transformers 4.57.x with torchvision, peft, decord, and lmdb. CUDA is
recommended for practical inference speed. Memory usage grows with image
resolution and other loaded models; start with a small image and disable other
detectors when memory is limited. The adapter uses PyTorch SDPA, so FlashAttention
and MagiAttention are optional.

Each class prompt is queried separately to preserve class identity. The model's
structured 0–1000 coordinates are converted to normalized dataset boxes. Invalid
or zero-area boxes and point-only outputs are ignored. Generation uses hybrid
decoding with a limit of 8192 new tokens per class prompt.

Locate Anything does **not** supply per-object confidence probabilities. VisionLab
stores a placeholder score of `1.0` for its valid generated boxes so they fit the
existing annotation format; it is not a calibrated probability or human acceptance.
The confidence threshold still controls the other detectors. Review generated
annotations before using them for training.

If loading or inference fails, the preview displays an error and the batch records
the failure. Locate Anything requests do not silently fall back to YOLO.

## Cloud and Colab

In cloud mode, the studio forwards the same detector toggle to a VisionLab worker.
Update the worker from this checkout before selecting the new model; an older
worker may reject the new option. The Modal worker template includes the model's
dependencies. Running Quick Detect remains the existing YOLO action.

In the **Colab** dialog, select **Locate Anything 3B boxes** or **Locate Anything 3B
boxes and masks**. The notebook and CLI accept `locate-boxes` and `locate-masks`:

```sh
python -m app.services.colab_batch --help
```

The source bundle includes this adapter and its requirements. Start with the
existing three-image sample limit; no remote resources are allocated merely by
selecting a pipeline.

The [NVIDIA model license](https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE)
restricts this checkpoint to non-commercial academic and nonprofit research use.
Check those terms before using the checkpoint in a commercial workflow.
