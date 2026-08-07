"""Search input resolution and model width (task #12).

Reports accuracy and latency **together** for every configuration. A width that
gains 2 points of solve rate while doubling CPU latency is not an improvement for
this project, and a table showing only one column would hide that (CLAUDE.md).

Every configuration is measured on the same machine, in the same session, with
`intra_op_num_threads` pinned to the same value -- ONNX Runtime's default thread
count collapses on this hybrid CPU (see `BASELINE.md`), and an unpinned session is
not comparable to anything, including itself.

Models are fused before benchmarking. Benchmarking the unfused training graph
measures three convolutions where one ships.

Writes `runs/search/<name>.json` and `docs/resolution-width-search.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (imgsz, width_mult, base_ch, neck_ch). Resolution is set by CHARACTER pixel
# size, not image size -- getting that backwards cost 18.7% recall in the
# feasibility gate. At 320px a typical 300x100 image maps near 1:1 and characters
# land around 33x46 px, so 320 anchors the range and 256/384 bracket it.
GRID = [
    (256, 0.5, 32, 64),
    (256, 1.0, 32, 96),
    (320, 0.5, 32, 64),
    (320, 1.0, 32, 96),
    (320, 1.5, 32, 128),
    (384, 1.0, 32, 96),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets" / "project-497" / "yolo"))
    ap.add_argument("--nc", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=4,
                    help="ONNX Runtime intra-op threads; pinned for comparability")
    ap.add_argument("--name", default="grid_v1")
    args = ap.parse_args()

    import torch
    import yaml
    from torch.utils.data import DataLoader

    from efficientvision.bench.latency import benchmark_onnx, host_info
    from efficientvision.data.dataset import CharDetectionDataset, collate
    from efficientvision.eval.harness import sweep_confidence
    from efficientvision.export.onnx import export_onnx
    from efficientvision.train.loop import Trainer, TrainConfig

    meta = yaml.safe_load((Path(args.data) / "data.yaml").read_text())
    names = [meta["names"][i] for i in range(len(meta["names"]))]

    out_root = ROOT / "runs" / "search"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for imgsz, wm, base_ch, neck_ch in GRID:
        tag = f"i{imgsz}_w{wm}_n{neck_ch}"
        print(f"\n=== {tag} ===")
        cfg = TrainConfig(
            data_root=args.data, nc=args.nc, imgsz=imgsz, epochs=args.epochs,
            batch=args.batch, device=args.device, base_ch=base_ch,
            width_mult=wm, neck_ch=neck_ch,
            out_dir=str(out_root / tag),
        )
        trainer = Trainer(cfg)
        trainer.fit()

        ckpt = torch.load(out_root / tag / "best.pt", map_location=cfg.device,
                          weights_only=False)
        trainer.model.load_state_dict(ckpt["model"])

        val_dl = DataLoader(
            CharDetectionDataset(args.data, "val", imgsz=imgsz, augment=False),
            batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate,
        )
        best, _ = sweep_confidence(trainer.model, val_dl, names, device=cfg.device)

        onnx_path = export_onnx(
            trainer.model.cpu(), out_root / tag / "model.onnx", imgsz=imgsz
        )
        lat = benchmark_onnx(
            onnx_path, input_shape=(1, 3, imgsz, imgsz), iterations=200, warmup=30,
            threads=args.threads, name=tag,
        )
        params = sum(p.numel() for p in trainer.model.parameters())

        rows.append({
            "tag": tag, "imgsz": imgsz, "width_mult": wm, "neck_ch": neck_ch,
            "params_m": round(params / 1e6, 2),
            "solve_rate": round(best.solve_rate, 4),
            "char_accuracy": round(best.char_accuracy, 4),
            "conf": best.conf,
            "p50_ms": round(lat.p50, 3),
            "threads": args.threads,
            "trustworthy": lat.trustworthy,
        })
        print(json.dumps(rows[-1], indent=2))

    record = {"host": host_info(), "threads": args.threads, "results": rows}
    (out_root / f"{args.name}.json").write_text(json.dumps(record, indent=2))

    lines = [
        "# Resolution / width search (task #12)",
        "",
        f"All rows measured on one machine, one session, `intra_op_num_threads="
        f"{args.threads}`, models fused before export. Raw data: "
        f"`runs/search/{args.name}.json`.",
        "",
        "Accuracy and latency are reported together on purpose: a configuration "
        "that gains solve rate while doubling CPU latency is not an improvement "
        "for this project.",
        "",
        "| Config | imgsz | width | Params (M) | **Solve rate** | Char acc | p50 (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: -r["solve_rate"]):
        flag = "" if r["trustworthy"] else " ⚠️"
        lines.append(
            f"| {r['tag']} | {r['imgsz']} | {r['width_mult']} | {r['params_m']} | "
            f"**{r['solve_rate']}** | {r['char_accuracy']} | {r['p50_ms']}{flag} |"
        )
    lines += ["", "⚠️ marks a latency measurement the benchmark flagged as unreliable "
              "(thermal drift or a busy machine); re-run those before quoting them."]
    (ROOT / "docs" / "resolution-width-search.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"\nwrote docs/resolution-width-search.md ({len(rows)} configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
