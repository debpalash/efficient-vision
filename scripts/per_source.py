"""Break solve rate down by image size (a proxy for CAPTCHA source/generator).

Both capacity hypotheses failed: label ambiguity was refuted (96.6% classification
on matched boxes) and resolution was refuted (320px gained +1 point of solve rate
over 160px). That points at the dataset rather than the model.

The filenames suggest several different origins mixed together (img_*.png,
captcha_*.jpg, image_*.png, bare hashes) and EDA found a dozen distinct image
sizes. If this is really N different CAPTCHA generators pooled into one label
space, a single 6k-image set may simply be too thin per style -- and the aggregate
solve rate would be hiding wide per-source variation. If instead every source is
uniformly mediocre, the problem is the task, not the mixture.

Output: runs/eval/per_source.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets" / "project-497" / "yolo"


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs" / "feasibility" /
                                             "feasibility_m_320" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = model.names
    img_dir, lbl_dir = DS / "images" / "val", DS / "labels" / "val"
    images = sorted(p for p in img_dir.iterdir() if p.is_file())

    groups = defaultdict(lambda: {"n": 0, "solved": 0, "edits": 0, "chars": 0})
    preds = model.predict(source=[str(p) for p in images], imgsz=args.imgsz,
                          conf=args.conf, device=args.device, verbose=False, stream=True)
    for p, r in zip(images, preds):
        with Image.open(p) as im:
            key = f"{im.size[0]}x{im.size[1]}"
        items = []
        for ln in (lbl_dir / (p.stem + ".txt")).read_text().splitlines():
            if ln.strip():
                q = ln.split()
                items.append((float(q[1]), names[int(q[0])]))
        items.sort(key=lambda t: t[0])
        gt = "".join(c for _, c in items)

        b = r.boxes
        pi = sorted(zip(b.xywh[:, 0].tolist(), [names[int(c)] for c in b.cls.tolist()]),
                    key=lambda t: t[0])
        pred = "".join(c for _, c in pi)

        g = groups[key]
        g["n"] += 1
        g["solved"] += pred == gt
        g["edits"] += levenshtein(pred, gt)
        g["chars"] += len(gt)

    out = {}
    for k, g in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
        if g["n"] < 5:
            continue
        out[k] = {
            "n_val": g["n"],
            "solve_rate": round(g["solved"] / g["n"], 3),
            "char_accuracy": round(1 - g["edits"] / g["chars"], 3) if g["chars"] else None,
        }

    (ROOT / "runs" / "eval" / "per_source.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"{'size':>12} {'n':>5} {'solve':>7} {'char_acc':>9}")
    for k, v in out.items():
        print(f"{k:>12} {v['n_val']:>5} {v['solve_rate']:>7.3f} {v['char_accuracy']:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
