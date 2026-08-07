# v1 Roadmap

Path to a shippable v1: a CPU model beating the measured 73.15 ms YOLO26n
baseline while holding ≥95% on the proprietary datasets, doing both multi-class
classification and OBB detection.

Tracked as tasks #1–#16. Phases are ordered by *risk retired per unit of work*,
not by how the final system is assembled.

---

## Phase 0 — Feasibility (tasks #1–#3) — **BLOCKED ON DATASET**

The most important phase, and the one most often skipped.

1. **Ingest dataset + converter** (#1)
2. **EDA: class histogram, object sizes, angles** (#2)
3. **Train a large teacher — the feasibility gate** (#3)

Phase 0 exists to answer one question: **is 95% reachable on this data at all?**

If a high-capacity teacher cannot clear 95%, then no compressed student will, and
the target is wrong — better to learn that in week one than week eight. The teacher
is also the distillation source, so passing the gate costs nothing extra.

**Do not begin optimization work until this gate passes.**

The EDA is not busywork either. Two numbers from it drive the entire design:
- **Object size distribution** → whether 640px can drop to 384/512. Latency scales
  roughly quadratically with resolution, making this the cheapest large win available.
- **Class frequency tail** → whether "≥95% balanced across hundreds of classes" is
  attainable. Long-tail distributions do not yield balanced accuracy without an
  explicit data strategy, and no architecture choice substitutes for one.

## Phase 1 — Model core (tasks #4–#8)

4. **RepVGG-style backbone** (#4) — builds on the already-tested `RepConv`
5. **Lightweight FPN/PAN neck** (#5)
6. **OBB head with (cos 2θ, sin 2θ)** (#6)
7. **Classification head, low-rank projected** (#7)
8. **Losses + one-to-one assignment** (#8)

Runs in parallel with Phase 0 where it does not depend on data. Note the reference
YOLO26n emits `{one2many, one2one}` outputs — confirmation that dual assignment is
the right target for NMS-free inference.

## Phase 2 — Training (tasks #9–#11)

9. **OBB-correct augmentation** (#9)
10. **BF16 training loop with resume** (#10)
11. **Knowledge distillation** (#11)

Task #9 carries more risk than its size suggests. Mosaic and perspective transforms
on *oriented* boxes are a well-known source of silent label corruption — the labels
end up subtly wrong, accuracy degrades, and nothing raises an error. It gets visual
verification tests, not just unit tests.

Task #11 is the highest-leverage accuracy work in the project.

## Phase 3 — Compression (tasks #12–#13)

12. **Resolution + width/depth search** (#12) — plot the accuracy/latency Pareto frontier
13. **INT8 QAT** (#13) — QAT, not post-training quantization; PTQ typically costs
    more accuracy than a 95% target can absorb

## Phase 4 — Ship (tasks #14–#16)

14. **Exports: ONNX, OpenVINO, NCNN, TorchScript** (#14) — with numerical parity tests
15. **Unified eval harness** (#15) — every metric in the spec, plus per-class breakdown
16. **Final head-to-head validation** (#16)

---

## Hardware constraints

**Training — RTX 4070 Laptop, 8 GB.** Teacher size is capped by activation memory,
not parameter count. Measured peak training memory at 640px with BF16
(`scripts/vram_probe.py`), against a usable budget of ~6.8 GB (85% of VRAM):

| Teacher | batch 4 | batch 8 | batch 16 | Largest usable batch |
|---|---|---|---|---|
| yolo26s | 1.2 GB | 2.3 GB | 4.5 GB | 16 |
| yolo26m | 2.3 GB | 4.4 GB | 8.6 GB ✗ | 8 |
| yolo26l | 2.8 GB | 5.4 GB | 10.6 GB ✗ | 8 |
| yolo26x | 4.3 GB | 8.1 GB ✗ | 15.8 GB ✗ | 4 |

**Recommended teacher: yolo26x at batch 4, with gradient accumulation ×4** for an
effective batch of 16. This is the highest-capacity teacher that genuinely fits,
and teacher capacity bounds the student's achievable accuracy.

> **Critical caveat — CUDA System Memory Fallback is enabled on this machine.**
> Verified directly: a 10 GB allocation on an 8 GB card succeeded without raising
> `OutOfMemoryError`. Oversized configurations silently spill into host RAM over
> PCIe and train 5–20x slower rather than failing. Never size a run by "it didn't
> OOM" here — compare measured peak against the budget. Consider disabling the
> fallback (NVIDIA Control Panel → Manage 3D Settings → CUDA Sysmem Fallback
> Policy) before long runs so misconfiguration fails loudly.

If the teacher proves too weak at these limits: gradient checkpointing (~40% less
activation memory, ~30% slower), lower training resolution (memory scales
~quadratically), or rent a cloud GPU for the teacher alone — a one-off cost, since
distillation only needs the teacher once.

**Deployment — pin your thread count.** The measured baseline collapses past 12
threads on this hybrid CPU; 32 threads is slower than 1. See BASELINE.md.

**Multi-GPU (DDP) is in the spec but cannot be tested here** — only one GPU is
available. Those code paths will be written but must be marked untested.

---

## Known risks

| Risk | Severity | Mitigation |
|---|---|---|
| 95% unreachable on this data | **Critical** | Phase 0 gate answers it in week one |
| Long-tail classes drag balanced accuracy down | **High** | EDA first; resampling + loss reweighting |
| OBB augmentation corrupts labels silently | **High** | Visual verification tests in #9 |
| INT8 costs more accuracy than budget allows | Medium | QAT over PTQ; keep FP16 fallback |
| 8 GB caps teacher quality | Medium | Checkpointing/accumulation, or cloud for teacher only |
| Unified model loses to two specialists | Low | Benchmark both; the harness makes this cheap |
