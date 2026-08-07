# Measured Baseline — YOLO26n on this hardware

The target we have to beat, measured locally rather than taken from published
figures. Reproduce with:

```
uv run --no-sync python scripts/baseline.py --imgsz 640 --threads 12
uv run --no-sync python scripts/thread_sweep.py runs/baseline/yolo26n_640.onnx
```

## Host

| | |
|---|---|
| CPU | 13th Gen Intel Core i9-13900HX (8 P-cores + 16 E-cores, 32 threads) |
| OS | Windows 11 (10.0.26200) |
| Runtime | ONNX Runtime 1.27.0, CPUExecutionProvider |
| GPU (training only) | RTX 4070 Laptop, 8 GB, sm_89, BF16 supported |

## YOLO26n @ 640px, batch 1

| Metric | Value |
|---|---|
| Parameters | 2,408,932 |
| GFLOPs | 5.4 |
| ONNX size | 9.48 MB |
| **Best latency** | **73.15 ms (p50) @ 12 threads** |
| Latency @ 4 threads | 145.55 ms |
| Latency @ 1 thread | 504.32 ms |
| Output shape | (1, 300, 6) — NMS-free, no post-processing cost |

## Thread scaling — the important finding

| Threads | p50 (ms) | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 504.32 | 1.00x | 100% |
| 2 | 272.67 | 1.85x | 92% |
| 4 | 145.55 | 3.46x | 87% |
| 6 | 107.41 | 4.70x | 78% |
| 8 | 88.12 | 5.72x | 72% |
| **12** | **73.15** | **6.89x** | **57%** |
| 16 | 123.65 | 4.08x | 25% |
| 24 | 411.45 | 1.23x | 5% |
| 32 | 698.64 | 0.72x | 2% |

**Performance collapses past 12 threads** — 32 threads is *slower than a single
thread*. This is the i9-13900HX hybrid topology: past ~12 threads the scheduler
places work on E-cores and hyperthread siblings, and the resulting core-to-core
latency imbalance means every layer's parallel barrier waits on the slowest,
weakest core.

Two consequences:

1. **Never leave thread count to the runtime default on hybrid CPUs.** ORT's
   default would have picked a catastrophic value here. Production deployment must
   pin threads explicitly.
2. **Allocate ~8–12 threads per inference stream, not the whole machine.**
   Efficiency is already 57% at 12 threads. On a server, running multiple
   concurrent 4-thread streams will deliver far more total throughput than one
   32-thread stream.

## On the published 38.9 ms figure

Ultralytics reports ~38.9 ms for YOLO26n at 640px on an Intel Xeon @ 2.00 GHz.
We measure 73.15 ms on a faster consumer CPU. The gap is not a contradiction —
published benchmarks typically run on many-core server parts with a tuned
threading configuration, and this is a thermally constrained laptop.

**This is exactly why the local baseline exists.** Any claim of "faster than
YOLO26n" must be measured on the same machine, same thread count, same runtime,
in the same session. Cross-machine comparisons are meaningless.

## Caveats on these numbers

- Runs at 8, 16, 24, and 32 threads were **flagged** by the harness for thermal
  drift or high variance. The trustworthy region on this laptop is ≤6 threads.
- This is a laptop under thermal constraint. Final numbers for a production
  decision should come from the actual deployment hardware.
- 640px is the default, not necessarily the right resolution. Input size is the
  largest single latency lever available and is dataset-determined — see
  ARCHITECTURE.md.
