"""Train the custom student detector (task #10 driver).

    python scripts/train_student.py --data datasets/project-497/yolo --nc 64

Deliberately mirrors the feasibility-gate settings so the student and the teacher
are comparable: same split tree, same 320px input, same character-semantic
augmentation lock, and mosaic absent (measured harmful -- see
`docs/feasibility-gate.md`).

Reports solve rate, not just loss. Loss curves have already proven a poor guide on
this dataset; only the contract metric decides whether a change is an improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets" / "project-497" / "yolo"))
    ap.add_argument("--nc", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--width-mult", type=float, default=1.0)
    ap.add_argument("--neck-ch", type=int, default=96)
    ap.add_argument("--tower-ch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--name", default="student_320")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from efficientvision.data.dataset import CharDetectionDataset, collate
    from efficientvision.eval.harness import sweep_confidence
    from efficientvision.train.loop import Trainer, TrainConfig

    out_dir = ROOT / "runs" / "student" / args.name
    cfg = TrainConfig(
        data_root=args.data, nc=args.nc, imgsz=args.imgsz, epochs=args.epochs,
        batch=args.batch, lr=args.lr, workers=args.workers, device=args.device,
        out_dir=str(out_dir), base_ch=args.base_ch, width_mult=args.width_mult,
        neck_ch=args.neck_ch, tower_ch=args.tower_ch,
    )

    trainer = Trainer(cfg)
    if args.resume:
        trainer.resume(args.resume)
    result = trainer.fit()

    # Class names come from the split's data.yaml so the decoded strings match the
    # ids the labels were remapped to -- never from a hardcoded alphabet.
    import yaml

    meta = yaml.safe_load((Path(args.data) / "data.yaml").read_text())
    names = [meta["names"][i] for i in range(len(meta["names"]))]

    ckpt = torch.load(out_dir / "best.pt", map_location=cfg.device, weights_only=False)
    trainer.model.load_state_dict(ckpt["model"])

    val_dl = DataLoader(
        CharDetectionDataset(args.data, "val", imgsz=args.imgsz, augment=False),
        batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate,
    )
    best, all_results = sweep_confidence(
        trainer.model, val_dl, names, device=cfg.device
    )

    report = {
        "config": {k: v for k, v in vars(args).items()},
        "best_val_loss": result["best_val_loss"],
        "contract_metric": "solve_rate (per-image string match)",
        "best_operating_point": best.as_dict(),
        "confidence_sweep": [r.as_dict() for r in all_results],
    }
    (out_dir / "student_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["best_operating_point"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
