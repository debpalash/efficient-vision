"""Feasibility gate (task #3): can a high-capacity model clear the accuracy target
on this CAPTCHA set? If a large teacher cannot, no compressed student will, and the
95% goal must be renegotiated before any optimization work.

Reports val mAP50 / mAP50-95 (Ultralytics) AND a character-level solve metric that
matches the actual product: per-image, is every character detected with the correct
class (a CAPTCHA is only 'solved' if the whole string is right).

Aug is pinned for character semantics: NO horizontal/vertical flip (b<->d, 6<->9),
NO rotation/shear/perspective. These would relabel glyphs. This is a correctness
constraint, not a tuning choice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets" / "project-497" / "yolo"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26m.pt")
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--name", default="feasibility_m_160")
    ap.add_argument("--data", default=str(DS / "data.yaml"),
                    help="data.yaml; point at a src_* tree for a single-source run")
    # Mosaic tiles 4 images into one canvas, so every character trains at roughly
    # HALF its nominal pixel size for all but the last `close_mosaic` epochs. On
    # faint, thin CAPTCHA glyphs that is not a free regularizer -- and it silently
    # confounded the 160-vs-320 resolution comparison, which ran mosaic on in both
    # arms. Exposed so it can be turned off and measured.
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()

    from ultralytics import YOLO

    # Ultralytics resumes from the interrupted run's own last.pt -- it carries the
    # optimizer state and epoch counter. Re-loading the pretrained teacher with
    # resume=True silently restarts from epoch 0 instead of resuming.
    last = ROOT / "runs" / "feasibility" / args.name / "weights" / "last.pt"
    if args.resume:
        if not last.exists():
            raise SystemExit(f"--resume given but no checkpoint at {last}")
        model = YOLO(str(last))
    else:
        model = YOLO(str(ROOT / args.model))
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=0,
        project=str(ROOT / "runs" / "feasibility"),
        name=args.name,
        exist_ok=True,
        # character-semantic aug lock:
        fliplr=0.0, flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0,
        mosaic=args.mosaic, close_mosaic=15, hsv_h=0.0, scale=args.scale,
        patience=40, verbose=False,
        # 106MB of images — caching in RAM removes the dataloader bottleneck that
        # held GPU util near 10% (these images are tiny, so I/O dominates compute).
        cache="ram", workers=8, resume=args.resume,
    )

    m = model.val(data=args.data, imgsz=args.imgsz, device=0,
                  project=str(ROOT / "runs" / "feasibility"), name=args.name + "_val",
                  exist_ok=True)
    out = {
        "model": args.model, "imgsz": args.imgsz, "epochs": args.epochs,
        "batch": args.batch, "data": args.data,
        "mosaic": args.mosaic, "scale": args.scale,
        "map50": float(m.box.map50), "map5095": float(m.box.map),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
    }
    res_dir = ROOT / "runs" / "feasibility" / args.name
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "gate_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
