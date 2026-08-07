# YOLO26n CPU baseline (task #17) — measured

The number this project must beat. Every figure below was measured on this
machine, in one session, on an idle system with no GPU run in flight. Raw data:
`runs/baseline/baseline_320.json`.

## Host

| | |
|---|---|
| CPU | 13th Gen Intel Core i9-13900HX |
| OS | Windows 11 (10.0.26200) |
| Runtime | ONNX Runtime 1.27.0, `CPUExecutionProvider` |
| Model | YOLO26n, fused, exported to ONNX opset 20, onnxslim'd, 9.36 MB |
| Params | 2,408,932 / 5.4 GFLOPs @320 |

## Latency at 320px

320px is the resolution the feasibility gate settled on, so the baseline is
measured there rather than at 640 — a comparison at a different resolution is not
a comparison.

| Threads | p50 (ms) | p90 (ms) | FPS |
|---|---|---|---|
| 1 | 27.55 | 28.21 | 36.3 |
| 2 | 16.44 | 17.34 | 60.8 |
| 4 | 10.92 | 11.48 | 91.6 |
| **8** | **8.70** | 11.61 | **114.9** |
| 16 | 10.35 | 12.26 | 96.6 ⚠️ |

⚠️ The 16-thread run was flagged unreliable by the benchmark's own drift check.

**Scaling is sublinear and then reverses.** 1→8 threads gives 3.2x on 8x the
threads, and 16 threads is *slower* than 8. This is the hybrid P-core/E-core
behaviour `BASELINE.md` warns about: past the P-core count the scheduler spreads
work onto E-cores and the slowest thread gates the batch. It is exactly why
`intra_op_num_threads` must be pinned rather than left to the runtime's default.

## The published figure does not transfer

Ultralytics quotes ~38.9 ms for YOLO26n at 640px on an Intel Xeon @ 2.00 GHz.
This machine is a mobile i9-13900HX and measures **8.70 ms at 320px / 8 threads**.
The two numbers share neither resolution nor hardware and must never appear in the
same comparison. Per `CLAUDE.md`, only the locally measured value is a target.

## What the project has to beat

| Metric | YOLO26n (measured) |
|---|---|
| p50 latency @320, 8 threads | **8.70 ms** |
| p50 latency @320, 4 threads | **10.92 ms** |
| Model size (ONNX) | **9.36 MB** |
| Parameters | **2.41 M** |

The student must be compared at the **same resolution, same thread count, same
runtime, same session**. Any student latency quoted without those four matching is
not a result.

## What this does NOT establish

This is a **latency-only** baseline. YOLO26n's *accuracy* on project-497 has not
been measured — the COCO-pretrained weights have never seen this label space, so
its out-of-the-box mAP here is meaningless and its fine-tuned accuracy is a
separate run that has not been done. A complete head-to-head (task #16) needs
YOLO26n fine-tuned on this dataset at 320px with the same augmentation lock, and
that has not happened yet.
