# Locate Anything 3B integration

Add the pinned `nvidia/LocateAnything-3B` model as an opt-in detector in the
existing auto-label pipeline, with standalone boxes, SAM 2 masks, and ensemble
fusion. Preserve existing defaults and ongoing cloud/Colab changes.

Integration covers the lazy inference adapter, pipeline configuration, preview
and batch API, React and Qt selectors, cloud serialization, Colab pipeline
selection, packaging dependencies, and setup documentation. No AEK enrollment
is present in this repository.

Acceptance checks:

- Parse NVIDIA's structured boxes into valid normalized annotations, ignoring
  malformed, reversed, zero-area, and point-only outputs.
- Keep each natural-language prompt associated with its requested class; exclude
  disabled classes and deduplicate ensemble proposals.
- Load weights only when selected, reuse the loaded model, and report inference
  errors without silently replacing Locate Anything with YOLO.
- Forward selection through preview, batch, cloud, and Colab; preserve annotation
  provenance and existing annotations.
- Support keyboard operation of the new desktop checkbox and existing Qt mode
  synchronization.
- Pass targeted regression tests, the Python suite, desktop tests, and build.

The model does not emit per-object probabilities. A score of 1.0 is a placeholder
for valid generated boxes, not calibrated confidence; document this behavior.
GPU inference cannot be validated on this CPU-only development machine.

Rollback: disable the opt-in toggle or revert only the integration's changes;
existing model selections and stored annotation formats remain usable.

## Verification

- Full combined Python suite: 344 passed, including API, adapter, cloud, Colab,
  and packaging tests on the final PR source.
- Desktop: 20 interaction tests passed; TypeScript and Vite production build passed.
- The real pinned NVIDIA processor tokenized a synthetic image and prompt. The
  real model class imported and constructed on the meta device with SDPA for both
  language and vision attention. No checkpoint weights were downloaded or loaded.
- Dependency lock check, changed Python compilation, new adapter/test Ruff checks,
  notebook JSON, and Git whitespace checks passed. The UI detector found no issues.
- Desktop lint completed with existing React warnings. Checking the existing model
  configuration file also reported two pre-existing long lines in result formatting.
- Self-review covered lazy loading, pinned remote code, prompt/class identity,
  provenance, error propagation, and box-only persistence. The latter now stores
  absent polygons as null rather than invalid empty lists. Locate-only API requests
  also avoid loading YOLO, and neither failures nor empty results trigger a YOLO
  fallback.

Remaining validation: execute actual weights on a suitable GPU and review detection
quality on representative images. A completed Colab GPU annotation round trip and
Modal deployment remain unverified.
