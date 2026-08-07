# Synthetic CAPTCHA generation (data lever)

The feasibility gate failed at **0.507** solve rate against a **0.95** target, and
the diagnosis was data volume, not model capacity: the hard sources carry 158–400
training images each, while 0.95 solve rate at 5 characters/image needs ~0.99
per-character accuracy. CAPTCHAs are machine-generated, so training data is
synthesizable with perfect labels — the one lever that does not require a better
model.

This document covers a **100-image proof of concept** replicating the hardest
source (230x60, measured solve rate 0.164 pooled / 0.344 dedicated).

Generator: `scripts/synth_captcha.py`. Style measurement: `scripts/analyze_style.py`.

## Parameters were measured, not guessed

`scripts/analyze_style.py` measured 200 real 230x60 images. Output in
`runs/eda/style_230x60.json`:

| Property | Measured |
|---|---|
| Characters per image | **exactly 5** (200/200 images) |
| Glyph box, p50 | 23.3 x 24.3 px |
| Inter-character gap, p50 | +6.2 px |
| Inter-character gap, **p5** | **−24.7 px** — characters overlap heavily |
| Vertical centre (normalized) | 0.51, p5–p95 = 0.31–0.63 |
| Background BGR | (194, 199, 202) |
| Glyph BGR | (79, 103, 105) |
| Glyph-vs-background luma delta | 102.6 |
| Saturation p95 | 196.9 — distractor lines are saturated, not grey |

The overlap figure is the one that would have been missed by eye, and it drives
the layout engine: characters must be allowed to collide.

## Labels are exact by construction

Boxes come from the rendered glyph's alpha mask after rotation, so each label is
the exact axis-aligned hull of the visible ink. This is **more consistent than the
human annotations** on the real data, where 1.9% of boxes extended past the image
border and had to be clamped.

Verified visually: `runs/eda/synth_labels_check.png`.

## Three iterations, and the aggregate statistics were actively misleading

**Attempt 1** matched the measured means and looked wrong: heavy colour blocks,
thick opaque lines, vivid near-black glyphs, a different font per character.

**Attempt 2** fixed those by eye and got *worse in a specific way*. It read the
near-neutral measured background mean (194, 199, 202) as "the paper is white", and
narrowed per-character size variance. Trained synthetic-only, it reached **mAP50
0.235 on real data at epoch 1 and fell to 0.213 by epoch 2** — the run was killed
rather than spend 3.8 hours on data already known to be wrong.

**Attempt 3** came from zooming into the real images at 3x
(`runs/eda/real_zoom.png`) instead of reading summary statistics. Five features
were visible immediately, and none of them appear in the aggregate numbers:

| Feature | Aggregate stat said | Zoom showed |
|---|---|---|
| Background tint | Neutral grey (194, 199, 202) | **Strong** pastel — pink, lavender, mint |
| Per-character size | p50 23x24 px | Varies ~2x *within a single image* |
| Per-character opacity | not measured | Varies ~3x — near-ghost glyphs beside solid ones |
| Line geometry | not measured | Near-**horizontal**, and **dashed**, not continuous |
| Colour structure | saturation p95 197 | Background, lines and speckle share **one hue family per image** |

The background error is the instructive one. The global mean is neutral *because*
each image carries a strong tint and the tints differ per image — averaging
hundreds of them lands on grey. **A parameter that varies per image cannot be read
from a global mean.** Attempt 1 was accidentally closer here than attempt 2.

The opacity finding matters most: near-transparent glyphs are a large part of why
this source scores 0.164, and attempt 2 rendered every glyph fully opaque. It was
synthesizing an *easier* task than the real one — which would have produced a
model that looked fine on synthetic data and failed on real.

Comparison sheet (top row real, bottom row synthetic):
`runs/eda/synth_compare_230x60.png`.

## Result: the replica does NOT help. All three synthetic arms lost.

20,000 images generated from attempt 3, then four arms differing only in training
data. Contract metric is solve rate on **real** 230x60 images.

| Arm | Training data | Val set | **Solve rate** | Char acc | mAP50 |
|---|---|---|---|---|---|
| **A — real only** | 400 real | 61 real | **0.344** | 0.771 | 0.749 |
| B — synthetic only | 20,000 synth | 461 real | 0.228 | 0.742 | 0.302 |
| C — mixed | 20,000 synth + 400 real | 61 real | **0.197** | 0.675 | — |
| D — synth pretrain → real FT | 20,000 synth, then 400 real | 61 real | 0.246 | 0.757 | 0.769 |

**Every synthetic arm underperformed the 400-image real-only baseline.** Adding
20,000 synthetic images made the model worse, in all three ways of adding them.

### mAP misled for the fifth time

Arm D scored **mAP50 0.769** against real-only's 0.749 — better on mAP, 10 points
*worse* on the contract metric. Every time these two have diverged on this
dataset, mAP has been the wrong signal. It is not usable for decisions here.

### What the numbers say happened

* **Arm B (0.228 from zero real images)** shows the replica captures real
  structure — that is far above chance. The generator is not worthless.
* **Arm C (0.197, the worst)** has `too_many_boxes` = **0.492**: it hallucinates
  spurious characters. With 20,000 synthetic to 400 real, real data is 2% of each
  batch, so the model optimizes for the replica and treats the real domain as
  noise it can ignore.
* **Arm D (0.246)** shows the same from the other side: synthetic pretraining set
  features that fine-tuning on 400 real images could not undo.

### Statistical caveat, stated plainly

Arms A, C and D were scored on **61** images, where the standard error near 0.3 is
about ±6 points. A-vs-D (0.344 vs 0.246) is roughly 1.6 SE — suggestive, not
conclusive on its own. The *consistency* of the direction across three independent
arms is what makes the negative result credible, not any single comparison.

### Conclusion

The bottleneck is **replica fidelity**, not the synthetic-data strategy. Real
CAPTCHA libraries apply sine/wave warping and stroke effects this generator does
not reproduce, and a model trained on an approximation learns the approximation.

**Do not scale this generator to 100k images.** Volume amplifies a distribution
mismatch rather than curing it — arm B already had 50x the real data and still
lost.

| Claim | Status |
|---|---|
| Labels are geometrically exact | ✅ Verified |
| Generator captures real structure | ✅ Arm B: 0.228 from zero real images |
| Synthetic data improves real accuracy | ❌ **Refuted — all three arms lost** |
| Replica matches the true generator | ❌ **Refuted** |

### What would change the answer

1. **Identify the actual generator library.** Filenames (`img_*.png`) suggest a
   common Python CAPTCHA package. Using the real library — rather than
   reverse-engineering its output by eye — removes the domain gap entirely and is
   worth more than any further tuning of this replica.
2. Failing that, add wave/sine warping and stroke-width variation, and re-run this
   same four-arm protocol. The protocol is cheap (~3 h) and it is the only thing
   that distinguishes a better replica from a nicer-looking one.

## Known remaining differences from the real images

Visible in the comparison sheet, not yet reproduced:

- Real backgrounds carry pastel tints (pink, green, yellow); synthetic ones are
  more neutral.
- Real speckle is denser and more colour-correlated on some images.
- Real distractor lines include more curvature; synthetic ones are mostly straight
  sweeps.

## Next steps, in order

1. Generate 10,000–50,000 images for this style.
2. Train on synthetic-only, evaluate on the **real** 230x60 val set. If solve rate
   does not beat 0.344, the replica is not good enough and no amount of volume
   fixes that.
3. If it transfers, train on synthetic + real mixed and measure again.
4. Repeat `analyze_style.py` → generator tuning for the other nine sources.

Step 2 is the gate. Skipping it and training on 50k unvalidated synthetic images
would be the same category of error as the mosaic defect: a plausible setting,
never measured, quietly costing accuracy.
