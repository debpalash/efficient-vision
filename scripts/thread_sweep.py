"""Measure how a model's CPU latency scales with intra-op thread count.

Two reasons this matters:

1.  Fair comparison. A latency number is meaningless without its thread count.
    Comparing our model at 8 threads against YOLO26n at 4 would be a fake win.
2.  Deployment sizing. Convolutional networks stop scaling well past a handful of
    threads because layers become memory-bandwidth-bound. Finding the knee tells
    us how many cores to actually allocate per inference stream -- allocating
    beyond it wastes cores that could serve concurrent requests instead.

Run:
    uv run --no-sync python scripts/thread_sweep.py runs/baseline/yolo26n_640.onnx
"""

from __future__ import annotations

import argparse
import os

from efficientvision.bench.latency import benchmark_onnx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="path to .onnx model")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=15)
    args = ap.parse_args()

    cores = os.cpu_count() or 8
    candidates = [t for t in (1, 2, 4, 6, 8, 12, 16, 24, 32) if t <= cores]
    if cores not in candidates:
        candidates.append(cores)

    print(f"logical cores: {cores}")
    print(f"model: {args.model} @ {args.imgsz}px, batch 1\n")
    print(f"{'threads':>8} {'p50 (ms)':>10} {'FPS':>8} {'speedup':>8} {'efficiency':>11}")
    print("-" * 50)

    single: float | None = None
    best: tuple[int, float] | None = None

    for t in candidates:
        r = benchmark_onnx(
            args.model,
            input_shape=(1, 3, args.imgsz, args.imgsz),
            iterations=args.iterations,
            warmup=args.warmup,
            threads=t,
            name=f"t{t}",
        )
        if single is None:
            single = r.p50
        speedup = single / r.p50
        # Parallel efficiency: how much of the ideal linear speedup we captured.
        # Falls off sharply once the model turns memory-bound.
        eff = speedup / t
        flag = "  [!]" if r.warnings else ""
        print(f"{t:>8} {r.p50:>10.2f} {r.fps:>8.1f} {speedup:>7.2f}x {eff:>10.0%}{flag}")

        if best is None or r.p50 < best[1]:
            best = (t, r.p50)

    if best:
        print(f"\nfastest: {best[0]} threads at {best[1]:.2f} ms")
    print("\n[!] marks runs flagged for thermal drift or high variance.")


if __name__ == "__main__":
    main()
