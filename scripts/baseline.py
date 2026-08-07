"""Establish the YOLO26n CPU latency baseline on this machine.

Ultralytics publishes ~38.9 ms for YOLO26n at 640px on an Intel Xeon @ 2.00 GHz.
That number does not transfer to other hardware, so every speed claim this
project makes is measured against a locally reproduced baseline instead.

Run:
    python scripts/baseline.py
    python scripts/baseline.py --imgsz 384 --threads 4

Requires the `baseline` extra:  uv pip install -e ".[baseline]"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from efficientvision.bench.latency import benchmark_onnx, host_info

BASELINE_DIR = Path("runs/baseline")
# Published Ultralytics figure, for context only. Our comparisons use the
# locally measured value.
PUBLISHED_MS = 38.9
PUBLISHED_HOST = "Intel Xeon @ 2.00 GHz, ONNX, 640px"


def export_yolo26n(imgsz: int, out_dir: Path) -> Path:
    """Export YOLO26n to ONNX, reusing the file if already present."""
    target = out_dir / f"yolo26n_{imgsz}.onnx"
    if target.exists():
        print(f"reusing existing export: {target}")
        return target

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "ultralytics is required to build the baseline.\n"
            '  uv pip install -e ".[baseline]"'
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"exporting yolo26n at {imgsz}px (downloads weights on first run)...")
    model = YOLO("yolo26n.pt")
    # simplify=True folds the training-time graph into the shape ONNX Runtime
    # actually executes; benchmarking an unsimplified graph overstates latency.
    exported = model.export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False)

    Path(exported).replace(target)
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imgsz", type=int, default=640, help="input resolution (default: 640)")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="intra-op threads; pin this when comparing models (default: ORT picks)",
    )
    args = ap.parse_args()

    info = host_info()
    print("host:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()

    onnx_path = export_yolo26n(args.imgsz, BASELINE_DIR)

    result = benchmark_onnx(
        onnx_path,
        input_shape=(1, 3, args.imgsz, args.imgsz),
        iterations=args.iterations,
        warmup=args.warmup,
        threads=args.threads,
        name=f"yolo26n@{args.imgsz}",
    )

    print(result.summary())
    print()
    if args.imgsz == 640:
        ratio = result.p50 / PUBLISHED_MS
        print(f"published reference: {PUBLISHED_MS} ms ({PUBLISHED_HOST})")
        print(f"this machine is {ratio:.2f}x that figure -- expected to differ; "
              f"the local number is what we target.")

    record = {
        "host": info,
        "imgsz": args.imgsz,
        "result": {
            "name": result.name,
            "p50_ms": result.p50,
            "p90_ms": result.p90,
            "p99_ms": result.p99,
            "mean_ms": result.mean,
            "stdev_ms": result.stdev,
            "fps": result.fps,
            "model_mb": result.model_mb,
            "threads": result.threads,
            "iterations": result.iterations,
            "throttle_drift": result.throttle_drift,
            "warnings": result.warnings,
            "trustworthy": result.trustworthy,
        },
    }
    out = BASELINE_DIR / f"baseline_{args.imgsz}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")

    if not result.trustworthy:
        raise SystemExit(
            "\nbaseline flagged as unreliable -- re-run on an idle, cool machine "
            "before using it as a comparison target."
        )


if __name__ == "__main__":
    main()
