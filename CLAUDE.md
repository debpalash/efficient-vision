# Repo rules

Conventions for this project. These are binding — follow them without being asked.

## Output format

- **Never publish Artifacts.** No hosted pages, no `claude.ai` links. All reports,
  charts, comparisons, and summaries are **local markdown files committed to this
  repo**.
- Tables over charts. A markdown table is diffable, greppable, reviewable in a PR,
  and readable offline. A rendered chart is none of those.
- Put generated reports under `docs/`. Keep raw measurement JSON under `runs/` so a
  report can always be traced back to the run that produced it.
- If a visual really is needed, use a mermaid block in the markdown — it renders on
  GitHub and stays in-repo.

## Dependencies

- **`uv` only.** Never pip, conda, or poetry.
- Pin non-PyPI wheels declaratively in `pyproject.toml` via `[[tool.uv.index]]` +
  `[tool.uv.sources]` — never with a one-off `--index-url` on the command line.
  Torch must come from the CUDA index; the PyPI wheels are CPU-only on Windows and
  silently disable GPU training.

## Measurement discipline

The project's whole claim is "faster than YOLO26n," so measurement rigor is the
product, not overhead.

- **Never quote a published benchmark as if it were ours.** Ultralytics' ~38.9 ms
  YOLO26n figure is an Intel Xeon number and does not transfer. Compare only against
  locally measured baselines.
- **Same machine, same thread count, same runtime, same session** — or it is not a
  comparison.
- **Pin `intra_op_num_threads` explicitly.** ONNX Runtime's default collapses on this
  hybrid CPU (32 threads is slower than 1). See `BASELINE.md`.
- **Never size a GPU run by "it didn't OOM."** CUDA System Memory Fallback is enabled
  here — oversized allocations spill to host RAM and train 5–20x slower instead of
  failing. Compare measured peak against ~85% of VRAM.
- **Fuse `RepConv` before benchmarking or exporting.** Benchmarking the unfused
  training graph badly understates deployed speed.
- Report accuracy and latency together. A change is never accepted on one axis while
  silently regressing the other.

## Honesty about results

- **Never present a projection as a measurement.** Targets, estimates, and
  projections must be labeled as such wherever they appear.
- **Never invent an accuracy number** for a model that has not been trained.
- **Do not compare metrics across datasets.** COCO mAP and "95% on our proprietary
  data" are different metrics on different data; they never share an axis or a table
  column without an explicit caveat.
- If a target is missed, say so plainly with the numbers. Do not reframe a miss as a
  partial success.
