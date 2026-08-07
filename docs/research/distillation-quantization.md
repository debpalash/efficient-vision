# Research: detection distillation + INT8 quantization (2023–2026)

Sonnet 5 research agent findings, 2026-07-18. Synthesis in `docs/research-plan.md`.

**Headline: naive RepVGG fusion is the root cause of INT8 collapse (20+pp loss),
not quantization itself — and it is fixable architecturally for near-zero cost.
This changes our RepConv block design (task #4).**

## Pipeline that the evidence supports

1. Multi-branch FP32 training with distillation (PKD feature loss + CrossKD-style
   head loss)
2. QARepVGG BN structure in the block from day one (not patched later)
3. Fuse
4. Short QAT (LSQ/LSQ+, per-channel weights) **with the FP32 model still acting as
   teacher** (YOLOv6 recipe) — not QAT against labels alone

## Ranked findings

### 1. QARepVGG — ADOPT (changes task #4)
**arXiv 2212.01593, AAAI 2024.** Multi-branch fusion produces a single kernel with
outlier weight statistics because each branch's BN independently rescales before
summation. Fix: strip BN from identity and 1x1 branches; add one BN *after* branch
summation. RepVGG-A0 INT8: 72.4→52.2% (collapse); QARepVGG-A0: 72.2% INT8 (<1% gap
to FP32). Near-zero inference cost.
**Action: restructure our `RepConv` BN placement now, before any training.**

### 2. RepOptimizer — CONSIDER (parallel prototype)
**arXiv 2205.15242, ICLR 2023.** Encode the multi-branch prior into a custom
optimizer; train a plain single-path net — no fusion, no outlier problem by
construction. Used in production YOLOv6. Higher integration cost than QARepVGG's
structural tweak; prototype later if QAT still struggles.

### 3. YOLOv6+ Regression Normalization — ADOPT
**Springer SIVP 2025.** Box-regression outputs have narrow dynamic range and get
crushed by INT8's 256-level grid. Normalize/rescale regression targets before
quantization. Cheap; apply to our box+angle head regardless of block design.

### 4. YOLOv6 production recipe — ADOPT (template pipeline)
**arXiv 2209.02976 / 2301.05586.** FP32 self-distillation → fuse → selective QAT
(skip most-sensitive layers, e.g. stem and box head) + channel-wise distillation
from the FP32 model during QAT. Full recovery to near-FP32 mAP where naive PTQ
fails. Direct evidence that teacher-in-the-loop QAT beats sequential.

### 5. CrossKD — ADOPT (head distillation)
**arXiv 2306.11369, CVPR 2024.** Route student head features into the teacher's
head; mimic only the teacher-processed predictions. Removes the GT-vs-teacher
target conflict. GFL R50: 40.2→43.7 AP; validated across anchor-based/free/query
heads. Training-only cost.

### 6. PKD — ADOPT (feature distillation)
**arXiv 2207.02039, NeurIPS 2022.** Pearson-correlation (scale-normalized) feature
imitation — right property for a strongly heterogeneous teacher/student pair
(YOLO26x → 2M-param student). Cheaper than MGD (no decoder).

### 7. QAT oscillation damping — CONSIDER
**arXiv 2311.05109, WACV 2024.** STE-based QAT oscillates between quantization
bins; EMA-stabilized updates + 1-epoch correction pass. Essential below INT8;
cheap insurance at INT8 if convergence is noisy.

### 8. LSQ/LSQ+ — ADOPT (QAT baseline machinery)
**ICLR 2020 / CVPRW 2020; still the 2024-2026 baseline.** Learnable quantization
step size; LSQ+ adds learnable zero-point for asymmetric post-ReLU activations.
Per-channel weights + learned per-tensor activation scales. Per-tensor *weight*
quantization is the most common cause of "detection head breaks under INT8."

### 9. Regression-head sensitivity — ADOPT (synthesized)
Box/angle regression absorbs nearly all quantization loss in multiple ablations.
Per-channel quantization mandatory on convs feeding the box head; keep final
regression layer(s) FP16 if INT8 loss concentrates there; RN-normalize targets.

### 10-11. CFWS / RepAPQ — CONSIDER (PTQ fallback only)
**arXiv 2312.10588 / 2402.16121.** PTQ-only fixes for already-fused rep-networks
(CFWS: RepVGG-A1 INT8 at 0.3% loss, no QAT). Pick one as backstop if QAT is too
expensive during development; inferior to fixing the architecture (item 1).

### 12. Runtime reality check — ADOPT (expectation-setting)
**ORT/OpenVINO/Intel/Lenovo benchmarks 2024-2025.** INT8 gains gated by VNNI:
2.3-3x compute-level with it. BUT small already-fast models are overhead/memory
bound — Xeon 6 OpenVINO YOLO detection saw ~1.2x end-to-end. **Budget 1.2-1.5x
for our 2M-param model, not 3x.** ARM INT8 gains unverified — measure on target.

### 13. Assignment-aware distillation — research gap
**LAD arXiv 2108.10520 + DETR-family KD.** Nothing published distills assignment
into a YOLO-style grid-based one-to-one head. Adapt CrossKD's decoupling instead;
treat teacher-guided one-to-one assignment as an experiment, not a recipe.

## Key timing answer

Distill during multi-branch FP32 training (fusion is lossless in FP32 — timing vs
distillation is irrelevant until quantization). Quantize only after fusion. Never
quantize unfused branches.
