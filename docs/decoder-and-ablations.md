# Count-prior decoding, and four things that did not work

All numbers measured on `datasets/project-497/yolo3` (train 4237 / val 907 /
test 907). The **test split was scored exactly once**, at the end, on the single
configuration the validation results had already selected.

## Headline

| Configuration | val | **test** |
|---|---|---|
| Standard threshold decode (best conf) | 0.4972 | 0.4972 |
| **Count-prior decode (gamma 0.25)** | 0.5424 | **0.5436** |
| Delta | +0.045 | **+0.046** |

The gain is **decode-time only** — same weights, same detections, different rule
for deciding which boxes constitute the answer. It costs a search over ~5
candidate counts per image and no retraining.

`gamma` was swept over 6 values on val, so it is a tuned hyperparameter. The test
figure (0.5436) landing within 0.002 of the val figure (0.5424) is what makes the
gain credible rather than selection noise.

## Why it works

At the best operating point the baseline loses **35% of images to count errors
alone** (24% too few characters, 11% too many) against ~14% that get the count
right and a character wrong. Deciding *how many* characters exist is a bigger
problem than recognising them.

The standard decoder makes that decision with one global confidence threshold and
lets the count fall out as a side effect. This decoder searches over plausible
counts instead, scoring each hypothesis by detector confidence, an empirical count
prior measured from training labels, and geometric plausibility.

Measured count prior (from train labels): 3: 0.030, 4: 0.208, **5: 0.417**,
6: 0.342, 7: 0.004.

Effect on the mechanism it targets: **count_match 0.646 → 0.752** on test.

## Two bugs that each made it worse than the baseline

Worth recording, because both produced a *working* decoder that was simply worse,
and either could have been mistaken for "count priors don't help".

**1. The scoring rule made the prior unoverrulable.** Scoring only accepted boxes
by *sum* of log-confidence biases every image toward the shortest hypothesis (each
extra term is negative). Switching to *mean* removes that bias but makes the score
invariant to count when candidates are equally confident — so six clearly-present
characters still decoded as five. Fixed by scoring **rejected** candidates too:
every candidate contributes `log p` if accepted or `log(1-p)` if rejected. The
term count is then constant across hypotheses (no length bias) and discarding a
confident box is expensive (so evidence can beat the prior).

**2. Detector confidence is not P(is-a-character).** The accept/reject likelihood
assumes it is. It is not — confidence is trained against an IoU-weighted target,
so a box at 0.3 is a real glyph far more than 30% of the time. Taking it at face
value made rejection cheap and the decoder under-predicted the count on 46% of
images, scoring **0.399 against a 0.497 baseline**. Fixed with a monotone
calibration `p ** gamma`, which lifts low confidences without changing the
detector's ranking.

| gamma | val solve rate | too_few | too_many |
|---|---|---|---|
| 1.00 (uncalibrated) | 0.3991 | 0.4642 | 0.0088 |
| 0.50 | 0.5094 | 0.2326 | 0.0584 |
| **0.25** | **0.5413** | 0.0992 | 0.1566 |
| 0.15 | 0.5171 | 0.0485 | 0.2337 |

The gain also **replicated on an independent checkpoint** (the 448px model,
+0.031), which is why it is treated as a property of the method rather than of one
set of weights.

---

# Refuted: three plausible ideas that did not work

## Preprocessing for signal-to-noise — REFUTED

Median filter plus CLAHE, applied offline so the two runs differed only in pixel
content. Rationale was sound and visually convincing: distractor lines are 1 px,
speckle is single pixels, glyph strokes are thicker, so a 3x3 median outvotes the
noise selectively. It visibly strips the speckle.

| Metric | Baseline | median_clahe | Delta |
|---|---|---|---|
| **Solve rate** | **0.4123** | 0.4068 | **−0.0055** |
| too_few_boxes | 0.2867 | 0.2712 | −0.0155 |
| too_many_boxes | 0.1654 | 0.1764 | +0.0110 |

It did exactly what was predicted mechanically — detection recall improved — and
bought back an equal number of false positives. Net zero.

Secondary finding worth keeping: **CLAHE alone is worse than no preprocessing.**
It amplifies speckle. Denoising has to come first, so testing only the obvious
contrast fix would have produced a misleading "preprocessing hurts".

## Higher resolution — REFUTED (properly controlled this time)

The original 160-vs-320 comparison was confounded: `mosaic=1.0` was on in *both*
arms, halving effective character resolution in both. That conclusion was void.
Re-run with mosaic off:

| Resolution | Solve rate (count-prior decode) |
|---|---|
| **320** | **0.5413** |
| 448 | 0.5281 |

Higher resolution still does not help. 320 stays, which also protects the CPU
latency budget.

## Two-stage detect → classify — REFUTED

Stage 1 detector for boxes, stage 2 resnet18 reading each box as a 64x64 crop —
roughly 6x the linear detail on the identity decision. Trained on 21,526
ground-truth crops with box-jitter augmentation to cover the GT-box vs
detector-box gap.

| Pipeline | Solve rate (same decoder) |
|---|---|
| **Single-stage** | **0.5424** |
| Two-stage | 0.5325 |

The stage-2 classifier reached only **89.3%** crop-level accuracy, against the
detector head's 96.6% on matched boxes. Extra pixels on the glyph do not
compensate for losing full-image context and joint training with localization.

---

## Where this leaves the target

**Test-set solve rate: 0.5436. Target: 0.90+.**

Per-character accuracy is 0.862; 0.90 solve at 5 characters/image needs ~0.979.
That is a 7x error reduction, and the four levers tried here produced one gain of
4.6 points. Nothing in this batch changes the conclusion of
`docs/feasibility-gate.md`: the remaining gap is a data and task-scoping problem,
not a decoding or architecture problem.
