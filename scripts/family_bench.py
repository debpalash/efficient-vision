"""Benchmark the whole YOLO26 family on CPU to map the speed/accuracy frontier.

This produces the reference curve that EfficientVision must beat. Having the full
family rather than just the nano variant matters because it shows the *shape* of
the tradeoff -- how much latency each accuracy point costs on this hardware --
which is what tells us whether a target is ambitious or fantasy.

COCO mAP figures are Ultralytics' published values, used only to place each model
on the accuracy axis. Latency is measured locally; published latency does not
transfer across machines.

Run:
    uv run --no-sync python scripts/family_bench.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from efficientvision.bench.latency import benchmark_onnx, host_info

OUT_DIR = Path("runs/family")

# Published COCO mAP50-95 for each variant, and parameter counts. These are
# reference values for the accuracy axis, NOT measured here.
PUBLISHED = {
    "yolo26n": {"map": 39.8, "params_m": 2.4},
    "yolo26s": {"map": 47.2, "params_m": 9.5},
    "yolo26m": {"map": 51.5, "params_m": 20.4},
    "yolo26l": {"map": 53.0, "params_m": 24.8},
    "yolo26x": {"map": 54.9, "params_m": 56.9},
}

# Larger variants are slow enough on CPU that a full 200-iteration run wastes
# minutes without improving the estimate.
ITERATIONS = {"yolo26n": 120, "yolo26s": 80, "yolo26m": 50, "yolo26l": 40, "yolo26x": 30}


def export(model_name: str, imgsz: int) -> Path:
    """Export one variant to ONNX, reusing the file when present."""
    target = OUT_DIR / f"{model_name}_{imgsz}.onnx"
    if target.exists():
        return target

    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  exporting {model_name}...", flush=True)
    exported = YOLO(f"{model_name}.pt").export(
        format="onnx", imgsz=imgsz, simplify=True, dynamic=False
    )
    Path(exported).replace(target)
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--threads",
        type=int,
        default=4,
        help="pinned intra-op threads. 4 sits inside this laptop's thermally "
        "trustworthy region; the 12-thread optimum gets flagged for drift.",
    )
    args = ap.parse_args()

    info = host_info()
    print(f"host: {info['processor']}")
    print(f"imgsz {args.imgsz}, batch 1, {args.threads} threads\n")

    rows = []
    for name, ref in PUBLISHED.items():
        print(f"{name}:", flush=True)
        path = export(name, args.imgsz)
        r = benchmark_onnx(
            path,
            input_shape=(1, 3, args.imgsz, args.imgsz),
            iterations=ITERATIONS.get(name, 50),
            warmup=10,
            threads=args.threads,
            name=name,
        )
        rows.append(
            {
                "model": name,
                "p50_ms": round(r.p50, 2),
                "fps": round(r.fps, 2),
                "onnx_mb": round(r.model_mb, 2),
                "params_m": ref["params_m"],
                "coco_map": ref["map"],
                "flagged": bool(r.warnings),
            }
        )
        flag = "  [!]" if r.warnings else ""
        print(f"  {r.p50:.1f} ms | {r.fps:.1f} FPS | {r.model_mb:.1f} MB{flag}\n", flush=True)

    print(f"{'model':>9} {'p50 ms':>9} {'FPS':>7} {'MB':>7} {'params M':>9} {'COCO mAP':>9}")
    print("-" * 56)
    for row in rows:
        print(
            f"{row['model']:>9} {row['p50_ms']:>9.1f} {row['fps']:>7.1f} "
            f"{row['onnx_mb']:>7.1f} {row['params_m']:>9.1f} {row['coco_map']:>9.1f}"
        )

    out = OUT_DIR / f"family_{args.imgsz}_t{args.threads}.json"
    out.write_text(json.dumps({"host": info, "threads": args.threads,
                               "imgsz": args.imgsz, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
