# Research: CPU-efficient backbones (2023–2026)

Sonnet 5 research agent findings, 2026-07-18. Ranked by expected impact on this
project (RepVGG-style reparam, ReLU, INT8-QAT, NMS/DFL-free, CPU target).
Synthesis into the project plan lives in `docs/research-plan.md`.

---

### 1. CPUBone — ADOPT (methodology)
**arXiv 2603.26425 (CVPR Findings 2026).** Directly attacks the FLOPs≠latency gap
for low-parallelism CPU devices. Proposes MACs-per-second (MACpS) as the
hardware-aware metric; uses grouped convolutions (g=2) and *reduced* kernel size
(3×3→2×2) where they raise MACpS on CPU. Raspberry Pi 5: B0 24.2ms/77.6% top-1;
beats FasterNet-T1 by 33% latency at +1.4% acc; beats RepViT-M2.3 on all CPU
devices. Shows the same tricks *hurt* GPU throughput 20–25% — GPU-tuned designs
mislead CPU work. Quantization not addressed.
**Action:** replicate the MACpS profiling loop on our i9-13900HX/ORT harness before
locking block shapes; A/B grouped-conv and small-kernel RepConv variants.

### 2. LowFormer — CONSIDER
**arXiv 2409.03460 (WACV 2025).** Same authors; redesigns macro/micro architecture
against measured throughput on GPU/mobile/ARM CPU. Reinforces profile-real-latency
discipline. Slimmed-attention block low priority (we are attention-free by design).

### 3. FasterNet / PConv — ADOPT (trial)
**arXiv 2303.03667 (CVPR 2023).** Partial Convolution: regular conv on a subset of
channels (e.g. 1/4), rest passed through and concatenated — cuts FLOPs *and* memory
access, unlike depthwise which stays memory-bound. FasterNet-T0 is 3.3× faster than
MobileViT-XXS on CPU at +2.9% acc. Pure conv, INT8-compatible.
**Action:** trial PConv as alternative to dense 3×3 in wide middle/late stages where
dense conv becomes memory-bound after reparam.

### 4. RepViT — CONSIDER (macro-design only)
**arXiv 2307.09283 (CVPR 2024).** MobileNetV3 rebuilt with ViT macro lessons, pure
CNN, RepVGG-style reparam depthwise. >80% top-1 at <1ms on iPhone 12 (CoreML).
Uses GELU — bad for INT8; all numbers are iPhone/CoreML, not x86.
**Action:** adopt token/channel-mixer separation and alternating SE placement if
capacity is needed; swap GELU→ReLU; re-validate latency on our harness.

### 5. MobileOne — ADOPT (already our foundation)
**arXiv 2206.04040 (CVPR 2023).** Multi-branch train-time blocks collapse to
branchless conv at inference because branchless minimizes memory access cost —
published validation of our core premise. <1ms iPhone 12 at 75.9% top-1.
**Action:** cross-check their branch-count ablation — diminishing returns beyond a
small number of train-time branches; cap our RepConv branches accordingly.

### 6. FastViT / RepMixer — CONSIDER (one idea)
**arXiv 2303.14189 (ICCV 2023).** Reparameterizes away *skip connections*, not just
conv branches, to cut memory access. GELU caveat again.
**Action:** check whether our neck/head retains skip connections that could be
reparam-collapsed.

### 7. StarNet — CONSIDER (activation finding matters most)
**arXiv 2403.19967 (CVPR 2024).** Element-wise multiply of two branches buys
implicit high-dimensional nonlinearity without widening. Uses **ReLU6**, ablated:
unbounded activations misbehave at narrow width; bounded ranges quantize cleanly.
**Action:** the star op is a cheap capacity lever to test; the ReLU6 evidence feeds
directly into our activation choice.

### 8. RepGhost — CONSIDER (low priority)
**arXiv 2211.06088.** Reparameterized addition replaces Ghost-style concat (concat
has real memory cost). +2.5% top-1 vs GhostNet-0.5x at same latency, ARM-only
validation. Modest gains over MobileOne-style blocks.

### 9. MobileNetV4 — CONSIDER methodology, SKIP block
**arXiv 2404.10518 (ECCV 2024).** Per-hardware latency-table NAS. The UIB block and
Mobile MQA are accelerator-tuned, irrelevant to x86. The latency-LUT-per-target-
hardware search method is worth adopting if we do an architecture search (task #12).

### 10. VanillaNet — CONSIDER (experiment only)
**arXiv 2305.12972 (NeurIPS 2023).** Extreme shallowness (6 conv layers), no skips;
"deep training" prunes activations progressively so deployed net is near-linear.
Plausibly excellent for INT8, no explicit study. Risky for accuracy on a narrow
domain needing capacity. Treat as experiment, not default.

### 11. PP-LCNet — CONSIDER principle, SKIP blocks
**arXiv 2109.15099 (2021).** Intel-x86-specific bag of tricks; dated, but the
principle survives: budget expensive ops (SE, non-ReLU activations) sparingly and
place them late, never throughout.

### 12. Activation choice under INT8 — ADOPT (validates ReLU; test ReLU6)
**Synthesized: ICLR 2024 (OpenReview 9ydLP7como); arXiv 2402.01169; arXiv
2209.06383; StarNet ablation.** ReLU folds into the preceding conv and costs a
comparator; GELU cannot fold and needs erf/tanh approximation or LUT under INT8.
Multiple independent works confirm GELU→ReLU substitution specifically to speed up
quantized inference. Bounded ReLU6 additionally gives clean clipping ranges for
INT8 calibration, especially at narrow widths.
**Action:** keep ReLU; run ReLU6 as a near-free A/B during QAT.

### 13. Mobile linear attention — SKIP for now
**ICML 2024 (Yao et al.).** Most credible CPU-fast attention found (36% latency
reduction vs standard attention, Pixel 6 XNNPACK CPU). Not INT8-validated. Only
revisit if a pure-conv backbone plateaus on accuracy and cheap global context is
needed — then this is the first thing to prototype, not a transformer block.

### 14. UniRepLKNet — SKIP (useful negative finding)
**arXiv 2311.15599 (CVPR 2024).** Large-kernel accuracy gains are GPU-reported;
large depthwise kernels are memory-bound on CPU and contradict CPUBone's
small-kernel finding. Keep "wide not deep" in mind; skip the mechanism.

---

## Meta-finding

**arXiv 2607.01984 (2026):** controlled multi-generation study showing many claimed
lightweight-CNN gains don't survive matched training budgets. Almost none of the
papers above report x86 desktop latency — several are iPhone/CoreML-only.
**Standing rule: re-benchmark every candidate technique on our own harness before
adopting. Cross-paper mobile numbers are directional at best.**

## Priority actions

1. MACpS-style profiling loop on our harness before locking block shapes (CPUBone).
2. A/B PConv vs dense 3×3 in wide/late stages (FasterNet).
3. ReLU→ReLU6 A/B during QAT (multiple independent sources).
4. Audit neck/head for reparam-collapsible skip connections (FastViT).
5. Cap RepConv train-time branch count per MobileOne's ablation.
