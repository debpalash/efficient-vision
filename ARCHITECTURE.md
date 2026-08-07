# EfficientVision — Architecture Design

> **2026-07-18 research update:** a four-track literature sweep amended several
> decisions below. Deltas and sources: `docs/research-plan.md`. Key corrections:
> QARepVGG BN placement (INT8-critical), YOLO26n-obb is the true baseline (not the
> detect model), NMS-free is kept for export/determinism rather than raw speed,
> INT8 expectation revised to 1.2–1.5x, angle encoding goes to A/B rather than
> being locked.

Working design for a CPU-first vision model targeting YOLO26n-or-better latency
with ≥95% accuracy on narrow-domain proprietary datasets, supporting multi-class
classification and oriented bounding box (OBB) detection.

This document records *why* each choice was made, because several of them run
against common intuition.

---

## The core strategic bet

We are not trying to out-architect Ultralytics on COCO. That is a research-lab
effort against a team that has already harvested the obvious CPU wins (YOLO26
removed DFL and went NMS-free precisely for CPU throughput).

We are exploiting a different asymmetry: **YOLO26n spends most of its capacity on
generality we do not need.** It is trained to handle 80 heterogeneous COCO classes
across arbitrary imagery. A target of ≥95% accuracy implies a far narrower domain
than COCO — where the state of the art sits near 40 mAP. That gap between "COCO-hard"
and "our data" is the budget we spend on being smaller and faster.

Ranked by expected payoff:

| Lever | Why it works | Risk |
|---|---|---|
| Knowledge distillation from a large teacher | Best accuracy-per-FLOP lever known for small models | Low |
| Input resolution tuning | Latency scales ~quadratically; 640→384 is ~2.8x | Low — only if objects stay resolvable |
| INT8 quantization-aware training | 2–4x on x86 with AVX-512 VNNI | Medium — accuracy drift |
| Structural reparameterization | Free inference-time win, no accuracy cost | Low |
| Width/depth pruning to domain | Domain needs far less capacity than COCO | Low |

Note the ordering: **not one of the top levers is a novel architecture block.**
Architecture matters, but it is not where the wins are.

---

## Backbone: plain 3×3 convs, reparameterized

The most common mistake in "fast CPU model" design is reaching for depthwise
separable convolutions because MobileNet-style architectures have excellent FLOP
counts.

**FLOPs are the wrong metric for CPU.** Depthwise convolutions have very low
arithmetic intensity — a handful of FLOPs per byte of memory traffic — so they are
memory-bandwidth-bound, not compute-bound. They cannot saturate AVX2/AVX-512 vector
units and they defeat the cache-blocked GEMM kernels that oneDNN and XNNPACK are
built around. Dense 3×3 convolutions, despite far more FLOPs, are what those kernels
are fastest at, and they often win on wall-clock.

So: **dense 3×3 stacks, RepVGG-style.**

- **Training time:** each block is multi-branch — 3×3 conv + 1×1 conv + identity,
  each with its own BatchNorm. The branches give gradient diversity and train like
  a much deeper residual network.
- **Inference time:** the branches fuse algebraically into a *single* 3×3
  convolution. Identical outputs, one kernel launch, no residual-add memory traffic,
  perfectly cache-friendly.

This is the rare optimization with no accuracy/speed tradeoff — it is exact.

## Activations: ReLU, not SiLU

YOLO uses SiLU. We use ReLU (or ReLU6), for two independent reasons:

1. **Speed.** SiLU requires a sigmoid — an exponential per element. ReLU is a
   single `max` that vectorizes trivially and fuses into the preceding convolution.
2. **Quantization.** SiLU is non-monotonic with an unbounded positive range and a
   small negative lobe, which is genuinely hard to represent in INT8. ReLU6's
   bounded range quantizes cleanly.

The accuracy cost of ReLU is real but small, and distillation recovers most of it.
Given INT8 is a top-three lever, an activation that quantizes badly is disqualifying.

## Detection head: NMS-free, DFL-free

Both match YOLO26's reasoning and both are large CPU wins:

- **NMS-free.** Non-maximum suppression is sequential, data-dependent, and runs on
  CPU regardless of where the network ran. It also makes latency *input-dependent* —
  a crowded image is slower — which breaks real-time guarantees. We use one-to-one
  label assignment during training (dual assignment: one-to-many for training signal,
  one-to-one for inference) so the model emits clean predictions directly.
- **DFL-free.** Distribution Focal Loss predicts a discretized distribution per box
  edge, costing a softmax plus expectation over ~16 bins per edge per anchor. Direct
  regression is cheaper and quantizes better.

## OBB angle: predict (cos 2θ, sin 2θ)

Angle regression has a notorious boundary-discontinuity problem: a box at 179° and
one at 1° are nearly identical, but a naive L1 loss sees a huge error, producing
unstable gradients exactly at the wrap-around.

The standard fixes are Circular Smooth Labels (angle binning) or Gaussian-distribution
losses (KLD/GWD). Both work; both add complexity, and CSL adds a wide classification
output per anchor.

We instead **regress the two-element vector (cos 2θ, sin 2θ)**, recovering
θ = ½·atan2(sin2θ, cos2θ). Because a rectangle has 180° rotational symmetry, doubling
the angle makes the representation exactly periodic — 179° and 1° map to neighbouring
points, and the discontinuity disappears by construction rather than by loss
engineering. It costs two outputs per anchor, needs no special loss, and quantizes
well (both components are bounded to [-1, 1]).

If localization precision proves insufficient, KLD loss is the planned upgrade path.

## Hundreds of classes: decoupled classification

With C classes, a dense per-anchor classification head costs a 1×1 convolution of
shape (C_hidden → C) at every spatial position. At C in the hundreds this head can
dominate both parameters and latency — it scales linearly with class count while the
rest of the network is fixed.

Mitigation, in order of preference:

1. **Shared low-rank projection.** Factor the head into (C_hidden → d) then (d → C)
   with d ≪ C_hidden. Cuts head cost substantially when C is large.
2. **Class-agnostic localization.** Regress boxes and angle without class
   conditioning, so only the classification branch scales with C.
3. **Two-stage for very large C.** Class-agnostic detection, then a tiny classifier
   on crops. Decouples detector cost from class count entirely, and makes adding
   classes a matter of retraining only the small head — which directly serves the
   "easy addition of new classes" requirement.

## Shared backbone, two heads — with a caveat

The spec asks for unified classification + OBB from one architecture. This is
implemented as one backbone with two task heads.

**But it should be a deliberate choice, not a default.** A shared backbone only pays
off when both outputs are needed *on the same image in the same pass*. If
classification and detection run on different inputs in production, two specialized
models will beat the unified one on both speed and accuracy, because multi-task
training forces a capacity compromise neither task asked for.

The code supports both. Which to ship is an empirical question we answer with the
benchmark harness.

---

## Open questions that need real data

These cannot be resolved by design and are the first things to measure:

1. **What resolution do the objects actually need?** This is the single biggest
   latency lever and it is entirely dataset-determined.
2. **How long is the class-frequency tail?** "≥95% balanced across hundreds of
   classes" is the hardest requirement in the spec. Long-tail distributions do not
   yield balanced high accuracy without an explicit data strategy — resampling, loss
   reweighting, or per-class thresholds. This is a data problem, not an architecture
   problem, and no backbone choice fixes it.
3. **Is 95% even reachable?** Measured by training a large, slow, high-capacity model
   first. If a YOLO26x-scale teacher cannot clear 95% on this data, no compressed
   student will, and the target needs renegotiating before weeks are spent on it.

Question 3 should be answered before any optimization work begins.
