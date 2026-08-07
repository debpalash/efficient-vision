# Feasibility gate (task #3) — results

Can a high-capacity model clear the ≥95% target on this CAPTCHA set? If not, no
compressed student will, and the target needs renegotiating before optimization
work starts. Raw metrics: `runs/feasibility/*/gate_metrics.json`,
`runs/eval/solve_rate.json`, `runs/eval/confusion.json`.

**Contract metric is per-image string-solve rate**, not mAP. A CAPTCHA is solved
only if every character is detected, classified, and ordered correctly. At ~5
characters/image, 95% solve rate requires roughly **99% per-character** accuracy
(0.99^5 ≈ 0.951). mAP50 badly overstates readiness here and is reported only as a
secondary diagnostic.

## Run 1 — yolo26m @ 160px: FAILED

Early-stopped at epoch 87/120 (patience 40; best mAP50-95 at epoch 47).

| Metric | Measured | Target |
|---|---|---|
| **String-solve rate** | **0.431** @ conf 0.30 | ≥0.95 |
| Character accuracy (CER-based) | 0.820 | — |
| Correct character count | 0.593 | — |
| mAP50 | 0.826 | — |
| mAP50-95 | 0.605 | — |

This is a clear miss, not a near-miss. NMS IoU was swept over 0.3/0.5/0.7 with
**identical** results at every setting, ruling out decode mis-tuning.

### Diagnosis: detection recall, not classification

Confusion analysis (`scripts/confusion.py`) on the same checkpoint:

| Signal | Value |
|---|---|
| Classification accuracy **on matched boxes** | **0.966** |
| Missed ground-truth boxes | **851 / 4560 (18.7%)** |
| Spurious boxes | 623 |
| Total misclassifications | 127 |
| ...case-only (`c`→`C`, `z`→`Z`) | 29 (22.8% of confusions) |

Two hypotheses were tested and one was refuted:

1. **Label ambiguity — REFUTED.** The label space is case-sensitive and contains
   shape-identical pairs (`Cc`, `Oo0`, `Ss`, `Xx`, `Zz`, `1lI`), so undecidable
   labels were a plausible hard ceiling. They are not: case confusions account for
   **29 errors in total**. The 64-way character classifier is already at 96.6%.
2. **Resolution too low — SUPPORTED.** 18.7% of characters are never detected.

### Root cause: the 160px choice was wrong

`imgsz=160` was chosen from *image* size ("the images are tiny, so the resolution
lever points down"). The quantity that actually governs detection is *object*
size. A typical 300×100 image letterboxed to 160 becomes 160×53, shrinking each
character to roughly **18×24 px** — under the size where detection is reliable.
The setting shrank precisely what needed resolving.

At 320px the mapping is near 1:1 for the common 300×100 image and characters land
around 33×46 px.

**Correction to `docs/dataset-eda.md` finding #2:** the claim that "the resolution
lever points down, ~128–192" was wrong. It reasoned from image dimensions and
ignored object dimensions. Resolution must be set by character pixel size.

## Run 2 — yolo26m @ 320px: ALSO FAILED (resolution hypothesis refuted)

Stopped at epoch 45; best weights epoch 21.

| Metric | 160px | 320px |
|---|---|---|
| mAP50 | 0.826 | 0.856 |
| **Solve rate** | **0.431** | **0.441** |

Doubling resolution gained **+1 point** of solve rate. mAP rose while the contract
metric did not — the second time mAP has pointed the wrong way on this dataset.
**The resolution hypothesis is refuted**, as is the earlier claim in this document
that 160px was the root cause. Characters at 18px were not the binding constraint.

## Real root cause: this is ~10 different datasets pooled into one

Solve rate broken down by image size, a proxy for CAPTCHA generator
(`scripts/per_source.py`, 320px weights, conf 0.40):

| Source (size) | Train imgs | Val imgs | Solve rate | Char acc |
|---|---|---|---|---|
| 250x100 | 378 | 69 | **0.841** | 0.950 |
| 220x80 | 181 | 41 | 0.732 | 0.897 |
| 225x85 | 367 | 47 | 0.596 | 0.906 |
| 150x50 | 283 | 58 | 0.569 | 0.893 |
| 280x54 | 809 | 161 | 0.509 | 0.838 |
| 300x100 | 1605 | 258 | 0.469 | 0.860 |
| 200x65 | 448 | 85 | 0.235 | 0.703 |
| 180x50 | 158 | 32 | 0.188 | 0.766 |
| 300x75 | 310 | 61 | 0.180 | 0.669 |
| 230x60 | 400 | 61 | 0.164 | 0.738 |

Train has **66 distinct image sizes**; the top 10 cover ~4900 of 5144 images. These
are different CAPTCHA generators — different fonts, distortion, noise lines,
character spacing — pooled into one label space.

**The aggregate 0.44 is an average over subproblems ranging 0.16 to 0.84.** It does
not describe any real task, and no single number does.

Note that difficulty does not track training volume: the 300x100 source has 1605
training images and scores 0.469, while 250x100 has 378 and scores 0.841. So this
is **intrinsic style difficulty**, not just data thinness. More data on the hard
styles will help, but it is not simply a volume problem.

### What this means for the target

"≥95% accuracy on our dataset" is not a well-posed target here, because "our
dataset" is ten tasks with an 5x spread in difficulty. The easiest source already
reaches 0.95 character accuracy with a stock yolo26m — evidence that 95% IS
reachable per-style. Getting one model to 95% across all ten simultaneously is a
very different and much harder proposition, and nothing measured so far suggests
it is close.

Per-character arithmetic to keep in view: current effective per-character rate is
≈0.813 recall × 0.966 classification ≈ **0.786**. Reaching 95% solve needs ≈0.99.
The gap is dominated by recall.

## Ablation A — is pooling itself the problem? Partly.

A dedicated yolo26m@320 trained on **only** the hardest source (230x60, 400 train
/ 61 val), same aug lock, same 64-class id space so the runs differ in exactly one
variable — the data mixture:

| 230x60 val | Solve rate | Char acc |
|---|---|---|
| Pooled model (5144 train imgs) | 0.164 | 0.738 |
| Dedicated model (400 train imgs) | **0.230** | 0.708 |

Pooling costs roughly **6 points** of solve rate on this source — the dedicated
model wins despite seeing 13x less data. So per-style routing is worth something.
But it does not rescue the target: 0.23 is not 0.95. **Pooling is a contributing
factor, not the ceiling.**

## What the hard sources actually look like

Reasoning from aggregate numbers had gone as far as it could, so: look at the
images. The two extremes are not the same problem.

- **250x100 (solve 0.841)** — bold, high-contrast dark glyphs on a light
  background, one thin distractor line.
- **230x60 (solve 0.164)** — pale, thin, low-contrast pastel glyphs buried under
  dense saturated noise lines and colour speckle.

This is a **signal-to-noise** problem, not a capacity or resolution problem. That
reframing is what produced the next ablation.

## Ablation B — mosaic augmentation was actively harmful

Mosaic tiles 4 images into one canvas, so **every character trains at roughly half
its nominal pixel size** for all but the last `close_mosaic` epochs. On faint, thin
glyphs that is not a free regularizer. It also means the earlier 160-vs-320
"resolution test" was confounded: mosaic was on in **both** arms, so effective
per-character resolution never actually doubled.

Same source, same everything, `mosaic=1.0` → `0.0` (and `scale` 0.5 → 0.2):

| 230x60 dedicated | mAP50 | **Solve rate** | Char acc | too_few_boxes |
|---|---|---|---|---|
| mosaic=1.0 | 0.767 | 0.230 | 0.708 | 0.279 |
| **mosaic=0.0** | 0.749 | **0.344** | 0.771 | 0.377 |

**mAP50 went down 1.8 points while the contract metric went up 11.5 points** — the
third time on this dataset that mAP has pointed the opposite way from solve rate.
Any tuning decision made on mAP here is untrustworthy by default.

Consequence: **every gate run before this one was mis-configured.** The pooled
gate is being re-run at `mosaic=0.0` before any conclusion about the 95% target
stands.

## Run 3 — pooled, yolo26m @320, mosaic off: FAILED (0.507 vs 0.95)

Early-stopped at epoch 77/200 (patience 40; best around epoch 37).

| Pooled val | mAP50 | **Solve rate** | Char acc | too_few | too_many |
|---|---|---|---|---|---|
| @320, mosaic on | 0.856 | 0.441 | — | — | — |
| @320, **mosaic off** | 0.836 | **0.507** @ conf 0.40 | 0.841 | 0.255 | 0.095 |

Consistent with ablation B a second time, now on the pooled set: **mAP50 fell
0.020 while solve rate rose 0.066.** That is the fourth instance of mAP moving
independently of the contract metric on this dataset.

**This is still a clear fail.** 0.507 against a 0.95 target is not a near-miss,
and it is the best figure any completed run has produced. Removing the mosaic
defect recovered 6.6 points; the remaining gap is 44 points.

Where the loss now sits, at the best operating point (conf 0.40):

| Failure mode | Rate |
|---|---|
| Too few boxes (missed characters) | **0.255** |
| Too many boxes (spurious characters) | 0.095 |
| Right count, wrong character(s) | ~0.143 |

Detection recall still dominates, as it has since the first run. Per-character
arithmetic: 0.841 character accuracy over ~5 characters gives 0.841^5 ≈ 0.42,
close to the measured 0.507 (the gap is because errors cluster within images
rather than spreading evenly). Reaching 0.95 solve needs ~0.99 per character. The
teacher is at 0.841.

### Incident: a concurrent eval killed the run at epoch 35

The early reading above was taken by running the evaluator with `--device cpu`
while training continued, on the assumption that a CPU eval could not touch GPU
memory. It can: importing torch with a visible CUDA device allocates a CUDA
context regardless of the tensor device, and that pushed the training process into
`CUDA error: out of memory` at epoch 35.

The run was resumed from `last.pt` at epoch 36, costing one epoch. `--resume` was
also fixed in `scripts/feasibility.py` — it previously re-loaded the pretrained
teacher rather than the run's own `last.pt`, which silently restarts from epoch 0
instead of resuming.

**Rule going forward: no concurrent process while a gate run is training, CPU-only
or not.** This project had already learned this once; the "but it's on CPU"
exception is not a real exception.

## Verdict: the gate fails

Three completed runs, best solve rate **0.507** against a **0.95** target.

The gate's stated purpose was: *if a high-capacity teacher cannot clear the
target, no compressed student will, and the target needs renegotiating before
optimization work starts.* A 22M-parameter yolo26m, trained to convergence at a
correct resolution with the augmentation defect removed, reaches 0.507. **The
condition for renegotiation is met.**

What was ruled out along the way, each with measurements rather than argument:

| Explanation | Verdict |
|---|---|
| Label ambiguity (case-identical glyphs) | Refuted — 29 errors total, 96.6% classification on matched boxes |
| Resolution too low | Refuted — 160→320 gained 0.010 |
| Augmentation defect | **Confirmed and fixed** — worth 0.066 pooled, 0.115 on the hardest source |
| Pooling ten generators into one model | Partly true — worth ~0.066 on the hardest source, not the ceiling |
| Model capacity | Not the binding constraint at 22M params |

What remains unexplained is the residual 0.255 rate of missed characters,
concentrated in the low-contrast, heavy-noise sources.

### The target is not well-posed as stated

"≥95% on our dataset" treats project-497 as one task. It is ~10 CAPTCHA
generators with a measured 0.164–0.841 solve-rate spread. The easiest source
already reaches 0.95 character accuracy with a stock model; the hardest reaches
0.344 even with a dedicated model. No single number describes both.

Three coherent ways to make the target well-posed, in order of how much evidence
supports them:

1. **Scope to the generators that matter in production.** If the deployed system
   faces two or three styles, drop the rest and the target may well be reachable.
   This is the only option with positive evidence behind it — some sources already
   perform near target.
2. **One model per style**, routed by image size (trivially detectable, 100%
   reliable here). Costs N models to ship; recovers the ~6 points that pooling
   destroys, which is not by itself enough.
3. **Keep all ten in one model and lower the number.** On current evidence 0.95 is
   not reachable this way, and a defensible target would be set from measurement
   rather than aspiration.

**This is a scoping decision, not a modeling decision, and it is the user's to
make.** No further optimization work should start until it is answered — that is
precisely what this gate exists to prevent.
