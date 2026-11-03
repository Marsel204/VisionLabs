# M5 quality statistics cache

Branch: `perf/m5-stats`, based on local `perf/m4-upload` (`f4bee61`). No pushes or PRs.

## Implementation

- Upload portability fix (`d5a802e`): prefer atomic hard links, fall back on unsupported/permission/link-limit errors (including simulated Windows error 50) to bounded exclusive-create copying. Collision and partial-write tests retain batch rollback and ownership protection.
- Additive SQLite migration: nullable `images.class_histogram` JSON object, plus a small invalidation-generation table. Existing rows/data remain intact; unknown histograms are rebuilt from annotation documents. `json_each` and SQL `SUM` aggregate valid histograms. Known zero-box rows are skipped, including old caches without histograms.
- Stats resolves the registry once per request. Each missing histogram is read and persisted while coordinating with repository writers; the global dataset lock is released throughout the long scan. Class migrations, rescans, image identity changes, count-only writes, and authoritative saves invalidate affected data. Cache fills use a conditional generation check, preventing stale reads from being published after a rescan. A concurrent invalidation or registry migration produces 409 for retry rather than an incomplete aggregate.
- Repository saves invalidate the cache before the durable canonical write and refresh afterward. A post-save cache-refresh error returns a warning and leaves the row unknown. Annotation reads, API saves, imports, auto-labeling and Colab persist histograms with the status/count update. Per-image locks encompass the read/save and its cache update to prevent a delayed result overwriting a newer histogram.

## Before / after

Run from this worktree with the existing project interpreter:

```
.venv/bin/python artifacts/perf-m5/bench_stats.py before
.venv/bin/python artifacts/perf-m5/bench_stats.py after
```

The `before` result was captured before changing M5 production code, after the upload fix. Each invocation generates fresh separate legacy/canonical datasets: 1,000 flat 640x480 JPEGs, one `car` annotation per image. The cold request starts with a scanned index and unknown annotation counts; warm is the median of the next five requests. Dataset creation, initial scan and module import are excluded. Both runs instrument repository reads and JSON text reads and assert identical annotation/image/review counts. OS page caches were not dropped; cold means application annotation cache, not cold disk. Endpoint function timings exclude HTTP overhead. Runtime: Python 3.14.8, SQLite 3.54.0; existing dependencies only.

| Format | Before cold | After cold | Before warm median | After warm median |
|---|---:|---:|---:|---:|
| Legacy YOLO | 1,534.60 ms | 1,347.25 ms | 1,521.06 ms | 0.913 ms |
| Canonical JSON | 409.76 ms | 382.06 ms | 99.127 ms | 0.903 ms |

Warm repository reads: legacy **1,000 -> 0**, canonical **0 -> 0**. Warm canonical JSON reads: **1,000 -> 0**. Cold reads: 1,000 annotation documents in both versions/formats. Raw samples and counters: `before.json`, `after.json`; generator: `bench_stats.py`.

## Validation

- Pre-fix upload reproduction: EPERM upload returned 409; corrected behavior succeeds with identical response and bytes.
- Upload targeted tests: **27 passed**.
- Pre-M5 response parity fixtures: both legacy and canonical passed before implementation; both warm zero-read requirements failed before implementation, then passed afterward.
- Expanded targeted suite (review regressions, imports, classes, API, services, Colab): **197 passed, 1 skipped**. Two subsequent cache-refresh-failure/count-only checks also passed.
- Final full suite, run in the background and awaited: **476 passed, 3 skipped, 11 warnings in 119.60 s**; exact output: `pytest-full.txt`. The three skips are optional Colab CLI integration checks. The strengthened rescan race guard also passed the focused 18-test stats/cache run.
- `git diff --check`: passed. No `desktop/` changes.

Added regression coverage: **35 parametrized test cases** across the two commits (9 upload, 26 stats/cache/class/import).

## Review points

Review the SQLite migration/SQL aggregation, pre-write invalidation, and per-image lock scope. A rescan intentionally marks annotation counts/histograms unknown even for unchanged images, to discover external label edits (including formerly empty images); it retains existing difficulty for unchanged images. External annotation changes become visible on rescan. Cache deletion and two old schema variants are tested. Failure recovery, class rename/merge/delete, direct and HTTP import overwrite, image size change with preserved mtime, pagination, zero annotation reads, and global-lock release have regression coverage.

Windows/link-unsupported behavior is simulated by injected OS errors; no Windows filesystem run was performed.
