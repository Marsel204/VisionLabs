# Cloud and Colab annotation verification

## Automated checks

The full Python suite passed **344 tests**, covering cloud request validation,
local API integration, account switching, all five supported Colab GPU request
types, runtime ownership and cleanup, bounded batches, preserved human edits,
resume, exports, notebook bootstrap, and Windows/frozen-app fallback resources.
These checks use synthetic images, fake credentials, and subprocess/runtime doubles;
they do not allocate cloud hardware.

The desktop passed **20 regression tests**, including sign-in without allocation,
three-image mask jobs, A100 selection, account switching, status polling, cleanup,
notebook downloads on unsupported platforms, and Locate Anything selection. The TypeScript/Vite production
build passed. Ruff passed on the new Python services, deployment template, packaging
script, and tests. The notebook's code cells compile and contain no saved outputs.
The dependency lock check passed. The complete Windows build remains for CI.

## Three-image model smoke test

An earlier CPU smoke test processed **only three images** from
[Marsel204/Dataset](https://github.com/Marsel204/Dataset), pinned to revision
`ed87d7c66396c24cab31023b4c76a4e7f964bd62`:

- `000001_jpg.rf.b073bea01084e55425513802c3daf67c.jpg`
- `000002_jpg.rf.f5219fa5580c366fef1dd12ca07bdb18.jpg`
- `000003_jpg.rf.3203c0bcf39e472da251bf3e40d38620.jpg`

Each image was checked against its Git blob ID. The dataset was not cloned and its
annotation JSON was not downloaded. Grounding DINO tiny and SAM 2.1 Hiera tiny
produced **83 objects with polygon masks** across the three images, using the
classes motorcycle, car, bus, truck, minivan, and person. This verifies inference
and data handling; it does not measure prediction accuracy against ground truth.

Native project round trips preserved boxes, masks, confidences, class IDs, and
review flags. COCO round trips retained the objects and masks; YOLO detection round
trips retained all objects. Resume skipped all three saved images without loading
models or changing canonical annotation files. Generated archives and model weights
are local artifacts and are excluded from Git.

## Live integration limits

Interactive Google browser authorization and read-only account queries were
confirmed. A completed annotation round trip on a real Colab GPU, including paid
compute units, remains unverified. Locate Anything integration is covered by model
doubles; actual checkpoint inference still needs a suitable GPU. The Modal template has not been
deployed. GPU availability and account permissions still depend on the provider.
