# Research: oriented bounding box detection (2023–2026)

Sonnet 5 research agent findings, 2026-07-18. Synthesis in `docs/research-plan.md`.

## Reality checks (change our plan)

1. **Wrong baseline corrected.** 38.9ms/129ms figures are the *axis-aligned detect*
   task at 640px. The real OBB comparison point is **YOLO26n-obb: 97.7±0.9ms
   CPU-ONNX (Xeon), 78.9 mAP50 DOTAv1 @1024px, 2.5M params, 14.0 GFLOPs**
   (arXiv 2509.25164). We must re-baseline locally on the obb model (task added).
2. **No published OBB detector reports CPU latency except YOLO26-obb.** Every
   academic "efficient" OBB paper benchmarks GPU/Jetson only. Our CPU numbers will
   be novel data points.
3. **NMS-free buys little raw CPU latency in sparse scenes** (rotated NMS is
   O(n²) but sub-ms at 5-20 objects). Legit reasons to stay NMS-free: ONNX has
   **no rotated-NMS op** (real export pain), deterministic latency, density
   robustness. Keep the design; drop the "NMS-free = big speed win" claim.
4. **One-to-one assignment has a measured accuracy tax:** ~-0.5 to -0.6 AP even
   with dual assignment (YOLOv10/YOLO26 ablations). Budget for it.
5. **Nobody has published dual assignment adapted to OBB** except Ultralytics'
   shipped code. First-mover territory; YOLO26-obb source is the only prior art.

## Ranked findings

### 1. YOLO26-obb — ADOPT (template + baseline)
Dual o2m/o2o heads, DFL-free (reg_max=1), **Progressive Loss** (o2m weight decays
0.8→0.1 over training), **RotatedTaskAlignedAssigner using ProbIoU** as alignment
metric, STAL small-target assignment. n-scale: 78.9 mAP50, 97.7ms CPU.

### 2. ProbIoU — ADOPT (loss + assignment metric)
**arXiv 2106.06072; IEEE TIP 2024.** Gaussian-box Hellinger distance, bounded
[0,1], no squashing hyperparameters, differentiable everywhere. PP-YOLOE-R
ablation: ProbIoU→KLD drops 78.14→76.03. Cheap closed-form 2×2 ops. Use for both
loss and one-to-one assignment cost (as YOLO26-obb does).

### 3. KFIoU — CONSIDER (ablation candidate)
**arXiv 2201.12558, ICLR 2023.** Kalman-style SkewIoU approximation; tens of lines
of code. Not clearly better than ProbIoU in third-party numbers.

### 4. Angle encoding: keep (cos 2θ, sin 2θ) — ADOPT, with a caveat
**PSC CVPR 2023 (2211.06368); FSC 2026 (2604.20281); CSL/DCL.** FSC shows >2
harmonics hurts (noise amplification) — low-channel trig encodings validated as
near-optimal. CSL costs 180 channels, DCL ~8; both worse for a nano model.
Optionally test a second harmonic (4ch) if near-square objects are common.

### 5. Caveat: YOLO26-obb itself uses direct angle regression — CONSIDER
Verified at source level: single scalar `(sigmoid(x)-0.25)·π`, patched with loss
term `sin²(2Δθ)` weighted by `exp(-(log(w/h))²/9)` to handle near-square
ambiguity. **Run our cos/sin vs direct-scalar ablation on our data before locking
the head** (no controlled ablation exists in the literature).

### 6. GWD/KLD — SKIP as primary
Raw KLD collapses (0.20% AP) without hand-tuned squashing. ProbIoU strictly less
finicky. Borrow KLD's aspect-ratio sensitivity only if thin/elongated objects
dominate the domain.

### 7. FCOSR ellipse assignment — CONSIDER
**arXiv 2111.10780.** Elliptical Gaussian center-sampling (rotation-aware positive
sampling). Cheap, orthogonal to loss/encoding; only sub-4M OBB detector with edge
numbers (Jetson).

### 8. Vertex-distance matching costs — CONSIDER (differentiator)
**RHINO WACV 2025 (2305.07598); RiO-DETR 2026.** Hausdorff/Chamfer distance over
box corners as assignment cost — periodicity-agnostic by construction (+4.2-5.0
AP50 in RHINO from matching alone). Hybrid ProbIoU + vertex-distance cost in a CNN
o2o head is untested territory.

### 9. Rotated-DETR family — SKIP architecture, mine ideas
11 papers, all GPU-only numbers; transformer attention is CPU-hostile. Take the
matching costs (item 8), not the architecture.

### 10. PP-YOLOE-R 91-bin angle DFL — SKIP
Reintroduces channel overhead we removed. Deployability warning worth keeping:
PaddleLite *excludes* all rotated detectors from its CPU-supported list — rotated
CPU export is underserved industry-wide; validate our export path early (task #14).

## Gaps only we can answer (novel measurements)

1. Rotated-NMS cost vs backbone at 5-20 objects/image on CPU (microbenchmark).
2. Dual assignment for OBB beyond Ultralytics' code.
3. Angle-encoding effect on CPU latency (cos/sin vs scalar).
