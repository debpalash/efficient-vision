# Dataset EDA — project-497 (CAPTCHA character detection)

Source: Label Studio export `project-497-at-2026-07-18-17-41-e299d710.zip`, YOLO
format with images. Raw numbers: `runs/eda/summary.json` (regenerate with
`scripts/eda.py`). All figures below are **measured**, not estimated.

## What this dataset actually is

Character-level detection on **CAPTCHA images** — locate and classify each glyph in
a short distorted string. This is the concrete instance of the "proprietary dataset"
the project targets, and it reshapes several assumptions in `ARCHITECTURE.md`.

| Property | Value |
|---|---|
| Images | 6051 (5251 `.png`, 800 `.jpg`) |
| Labeled boxes | 30,697 |
| Label format | YOLO **axis-aligned HBB** — every box is 5 tokens (`cls cx cy w h`) |
| Classes defined | 73 (`! # $ % & + - 0-9 = ? @ A-Z ^ a-z`) |
| Classes with data | 64 |
| Boxes/image | min 3, p50 5, p90 6, max 7 (mean 5.07) |
| Malformed / empty labels | 0 / 0 |
| Out-of-bounds boxes | 591 (1.9%) across 531 images — **clamped**, see below |

## Findings that change the plan

### 1. This is HBB, not OBB
All 30,697 boxes are 5-token axis-aligned. There is **no angle in this data.** The
OBB head, ProbIoU, and angle-encoding A/B (tasks #6, #8) are not exercised by this
dataset. The true local baseline is **YOLO26n (detect)**, not YOLO26n-obb — task #17
re-scoped. OBB stays in the codebase as a capability for the spec, but it is not on
the v1-for-this-data critical path.

### 2. Images are tiny and wide
> **CORRECTED 2026-07-18 — the conclusion below was wrong.** It reasons from
> *image* size, but detection is governed by *object* size. Letterboxing a 300×100
> image to 160px shrinks each character to ~18×24 px and cost 18.7% detection
> recall in the feasibility gate. Resolution must be set by character pixel size,
> not image dimensions. See `docs/feasibility-gate.md`.

Most common sizes: 300×100, 280×54, 200×65, 230×60, 250×100. Nothing near 640².
The resolution lever now points **down**, not at 640/1024. A letterboxed square of
~128–192 px is the search range, not 384–640. Latency will be far below the
`BASELINE.md` YOLO26n@640 numbers — but so must the baseline be re-measured at a
matched small resolution to keep the comparison honest (same-resolution rule).

### 3. Nine classes have zero examples
Absent entirely: `! # $ % & = ? @ ^`. "≥95% balanced across all classes" is
**unachievable** for these nine as-is — no model learns a class it never sees.
Options: (a) drop them from the label space for v1, (b) user supplies examples,
(c) synthesize glyphs. Recommend (a) for v1 → **64-class** problem, revisit later.

### 4. Long tail is mild, not severe
Imbalance ratio 46:1 (max class 1153, min-present 25). Only `l` is under 50
instances. This is well within reach of the planned RFS/IRFS + logit-adjustment
stack — no Seesaw/EQLv2 needed. Far easier than the "hundreds of classes, long tail"
worst case the architecture was hedged against.

### 5. 8.8% of images had out-of-bounds boxes (would be silently dropped)
531 images (591 boxes, 1.9%) have a box edge past the image border — edge
characters whose label extends slightly beyond the frame (max overflow 0.59).
Ultralytics **drops the entire image** on such labels, which would have removed
8.8% of the set non-randomly (biased toward first/last glyphs). `build_split.py`
now clamps box corners to `[0,1]` and recomputes center/size; 590 boxes clamped, 1
degenerate box dropped, 0 images lost. This is why EDA must precede the feasibility
gate — the naive run trained on a biased 91% subset.

### 6. Box geometry is regular
Widths p5–p95 = 0.067–0.185 (normalized); heights 0.29–0.64. Characters are tall,
narrow, uniformly sized, laid out left-to-right — a benign detection problem
compared to COCO. Reinforces the core bet: the 95% target is realistic *because*
the domain is far narrower than COCO.

## Consequences for the task list

- **#3 feasibility gate** is now cheap: small images, 64 classes, regular geometry.
  Train a YOLO26 teacher at a small square resolution; if it clears 95% mAP/accuracy,
  the target holds. This should run before any student work.
- **#6 / #8 OBB paths** are deferred for this dataset (kept for the general spec).
- **#12 resolution search** range corrected to ~128–192 px.
- **#17** re-scoped: baseline YOLO26n **detect** locally at the matched small
  resolution, not obb @1024.
- Decision needed from user: handling of the 9 zero-example classes (recommend drop
  for v1).
