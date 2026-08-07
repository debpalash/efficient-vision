# YOLO26 family vs EfficientVision — CPU comparison

Measured 2026-07-18 on this machine. Raw data: `runs/family/family_640_t4.json`.

Reproduce:

```sh
uv run --no-sync python scripts/family_bench.py --imgsz 640 --threads 4
```

## Read this first

**EfficientVision does not exist yet.** No trained weights, no accuracy number.
Every EfficientVision figure below is a **design target**, marked `(target)`. Nothing
here is a measurement of a model we have built.

**Accuracy is deliberately absent from the comparison.** The YOLO26 family's numbers
are COCO mAP; the project requirement is ≥95% on proprietary data. Different dataset,
different metric — putting them in one column would manufacture a comparison that does
not exist. Accuracy becomes comparable only after the feasibility gate (task #3) trains
a teacher on the real data.

**YOLO26x is the teacher, not the competitor.** It is a GPU-only model whose job is
supplying distillation targets. The model EfficientVision must beat on speed is
YOLO26**n**.

## Measured CPU latency

640×640, batch 1, 4 pinned threads, ONNX Runtime CPUExecutionProvider.
p50 of a timed run after warmup. No run was flagged for thermal drift.

| Model | p50 latency | FPS | ONNX size | Params | COCO mAP¹ | Role |
|---|---:|---:|---:|---:|---:|---|
| **YOLO26n** | **129.1 ms** | 7.8 | 9.5 MB | 2.4 M | 39.8 | **the model to beat** |
| YOLO26s | 407.2 ms | 2.5 | 36.5 MB | 9.5 M | 47.2 | |
| YOLO26m | 1.21 s | 0.8 | 78.2 MB | 20.4 M | 51.5 | |
| YOLO26l | 1.53 s | 0.7 | 95.0 MB | 24.8 M | 53.0 | |
| YOLO26x | 3.30 s | 0.3 | 212.9 MB | 56.9 M | 54.9 | distillation teacher — GPU only |
| **EfficientVision** | **≤ 64.5 ms** *(target)* | ≥ 15.5 *(target)* | — | — | not yet measurable | the goal |

¹ Ultralytics' published COCO figures, for context on model capability only. **Not
measured here, and not comparable to the ≥95% proprietary-data target.**

Relative to the YOLO26n baseline:

| Model | vs YOLO26n |
|---|---:|
| YOLO26s | 3.2× slower |
| YOLO26m | 9.4× slower |
| YOLO26l | 11.9× slower |
| YOLO26x | **25.6× slower** |
| EfficientVision *(target)* | 2.0× faster |

## The finding that matters

**Scaling up costs far more on CPU than parameter counts suggest.**

From n to x: **23.7× the parameters, 25.6× the latency — for +15.1 COCO mAP.**
Latency grows slightly faster than parameter count, because larger activations stop
fitting in cache and the model turns memory-bandwidth-bound rather than compute-bound.

Two consequences for this project:

1. **YOLO26x was never a CPU deployment candidate.** At 3.3 s/image it is 0.3 FPS.
   Its only role here is as a distillation teacher on the GPU. Comparing EfficientVision
   against it would be comparing the product against the training infrastructure.
2. **The accuracy tail is expensive.** The last +1.9 mAP (l → x) costs 1.77 s of CPU
   latency. This is exactly the region where distillation earns its place: buy the
   teacher's accuracy without paying the teacher's inference cost.

## Training memory (RTX 4070 Laptop, 8 GB)

Peak allocated during forward + backward at 640px with BF16. Usable budget 6.8 GB
(85% of VRAM). Raw data via `scripts/vram_probe.py`.

| Teacher | batch 4 | batch 8 | batch 16 | Largest usable batch |
|---|---:|---:|---:|---:|
| yolo26s | 1.2 GB | 2.3 GB | 4.5 GB | 16 |
| yolo26m | 2.3 GB | 4.4 GB | 8.6 GB ✗ | 8 |
| yolo26l | 2.8 GB | 5.4 GB | 10.6 GB ✗ | 8 |
| yolo26x | 4.3 GB | 8.1 GB ✗ | 15.8 GB ✗ | 4 |

✗ = exceeds budget. **These do not raise `OutOfMemoryError` on this machine** — CUDA
System Memory Fallback spills them into host RAM over PCIe, and they train 5–20×
slower instead of failing. See `CLAUDE.md`.

**Recommended teacher: yolo26x at batch 4 with gradient accumulation ×4** (effective
batch 16). It is the highest-capacity teacher that genuinely fits, and teacher capacity
bounds the student's achievable accuracy.

## Caveats

- **4 threads, not the optimum.** YOLO26n reaches 73.2 ms at 12 threads, but runs above
  6 threads get flagged for thermal drift on this laptop. The 4-thread numbers are the
  trustworthy ones, so the comparison is built on them. Both are valid; they are not
  interchangeable.
- **Laptop under thermal constraint.** Final numbers for a production decision should
  come from the actual deployment hardware.
- **640px is a default, not a decision.** Input resolution is the largest single latency
  lever (~quadratic) and is dataset-determined. See task #12.
- The EfficientVision target of 2× faster than YOLO26n is a **design goal chosen by this
  project**, not a result. It is unvalidated until task #16.
