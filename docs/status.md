# Project status

Last updated 2026-07-18. Every number here is **measured** unless explicitly
labelled otherwise. Raw data under `runs/`.

## The one-line version

The student architecture is built and tested end to end, and the YOLO26n latency
baseline is finally measured (**8.70 ms p50 @320**). But the **feasibility gate
failed at 0.507 against a 0.95 target**, so all optimization work is blocked
pending a scoping decision from the user.

## What is measured

### Feasibility gate (task #3) — not passed

Contract metric is per-image string-solve rate, not mAP. Detail in
`docs/feasibility-gate.md`.

| Run | Data | mAP50 | **Solve rate** |
|---|---|---|---|
| yolo26m @160, mosaic on | pooled | 0.826 | 0.431 |
| yolo26m @320, mosaic on | pooled | 0.856 | 0.441 |
| yolo26m @320, mosaic on | 230x60 only | 0.767 | 0.230 |
| yolo26m @320, **mosaic off** | 230x60 only | 0.749 | **0.344** |
| yolo26m @320, **mosaic off** | pooled | 0.836 | **0.507** |

Target is 0.95. Best measured: **0.507**. **The gate fails.** Not a near-miss —
the gap is 44 points, and the teacher is 22M parameters, so capacity is not the
constraint.

### Three hypotheses tested, two refuted, one confirmed

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Label ambiguity caps accuracy | **Refuted** | Case confusions total 29 errors; classification on matched boxes is 96.6% |
| Resolution was too low | **Refuted** | 160→320 gained +0.010 solve rate |
| Mosaic augmentation was harmful | **Confirmed** | Disabling it gained +0.115 solve rate on the hardest source |

### mAP is untrustworthy on this dataset

Three times now, mAP and solve rate have moved differently, twice in opposite
directions:

| Change | mAP50 | Solve rate |
|---|---|---|
| 160 → 320 px | **+0.030** | +0.010 |
| mosaic on → off (230x60) | **−0.018** | **+0.115** |
| mosaic on → off (pooled) | **−0.020** | **+0.066** |

A harness reporting only mAP would have kept the mosaic setting that was costing
11 points of the metric this project actually ships. Everything in
`efficientvision/eval/harness.py` reports solve rate first.

### The dataset is ~10 CAPTCHA generators pooled into one

Solve rate by image size (a generator proxy) ranges **0.164 to 0.841**. The
aggregate 0.44 is an average over subproblems with a 5x difficulty spread and
describes no real task. A dedicated model on the hardest source beats the pooled
model there (0.230 vs 0.164) despite 13x less data — so pooling costs something,
but it is not the ceiling.

Visual inspection explains the spread: the easy sources are bold high-contrast
glyphs; the hard ones are pale, thin, low-contrast glyphs buried under dense
saturated noise lines. It is a **signal-to-noise** problem.

## What is built and tested (95 tests passing)

| Component | Module | Notes |
|---|---|---|
| QARepVGG block | `models/blocks.py` | Exact fusion, structural guard tests |
| Backbone | `models/backbone.py` | P3/P4/P5, width-capped |
| PAN neck | `models/neck.py` | Single narrow width, nearest upsample |
| Detect head | `models/head.py` | Anchor-free, DFL box distributions |
| Loss + assignment | `models/loss.py` | Task-aligned assigner, CIoU, DFL |
| Dataset + aug | `data/dataset.py` | Character-semantic aug lock, no mosaic |
| Training loop | `train/loop.py` | BF16 AMP, full-state resume, VRAM guard |
| Distillation | `train/distill.py` | cls/DFL/feature KD |
| INT8 QAT | `quant/qat.py` | STE, per-channel weights |
| Export | `export/onnx.py` | ONNX + TorchScript, parity checked |
| Eval harness | `eval/harness.py` | Solve rate, CER, confidence sweep |

ONNX parity against PyTorch: boxes within **1e-3 px**, scores within **1e-4**.

## What is NOT done, and what that means

| Gap | Consequence |
|---|---|
| **Feasibility gate FAILED (0.507 vs 0.95)** | The target is not reachable on the pooled dataset by a model far larger than anything this project will ship. Blocks #12 and #16. |
| YOLO26n **accuracy** on this dataset unmeasured | The latency baseline exists; the accuracy side of the head-to-head does not. YOLO26n has never been fine-tuned on project-497. |
| Student never trained | Every accuracy figure above is a **teacher**, not the model this project ships. |
| Distillation never run | Implemented and unit-tested only. |
| QARepVGG INT8 claim unverified locally | Adopted from the published result. The local control measured ratio 1.031 on random weights — **inconclusive**, not confirmation. |
| OpenVINO / NCNN export | Not written; both depend on a trained student. |
| Resolution/width search (#12) | Not started. |

## The measured baseline (task #17)

YOLO26n, ONNX, idle machine, i9-13900HX. Full detail in
`docs/baseline-yolo26n.md`.

| | @320, 4 threads | @320, 8 threads |
|---|---|---|
| p50 latency | 10.92 ms | **8.70 ms** |
| Model size | 9.36 MB | 9.36 MB |
| Params | 2.41 M | 2.41 M |

Thread scaling reverses past 8 threads (16 is slower than 8) — the hybrid P/E-core
effect `BASELINE.md` warns about, and the reason thread count must be pinned.

Ultralytics' published 38.9 ms is a Xeon @640 figure. This machine is 3.6x faster
at a smaller resolution. The two never share a table.

## Honest assessment

**The gate failed and that is the headline.** A 22M-parameter teacher, correctly
configured, reaches 0.507 solve rate against a 0.95 target. Capacity is not the
constraint, resolution is not the constraint, and label ambiguity is not the
constraint — all three were tested and eliminated. What remains is that
project-497 is ten different CAPTCHA generators with a 5x difficulty spread, and
"95% on our dataset" does not name a single task.

The engineering is in good shape: the full student stack is implemented, 95 tests
pass, ONNX parity holds to 1e-3 px, and the baseline is finally a real number. But
**no accuracy claim has been made by the model this project exists to produce**,
and none should be until the target is restated against a scoped dataset.

The one genuine win this stretch was cheap and unglamorous: a default augmentation
setting was silently costing 6.6 points of the contract metric while *improving*
mAP. It was found by looking at the images and questioning a default, not by
adding capacity.
