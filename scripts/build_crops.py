"""Extract per-character crops for the two-stage classifier (task #23).

Stage 1 (detector) localizes glyphs; stage 2 classifies each one from a crop
upscaled to a fixed size. The point is resolution: at 320px input a character
occupies roughly 24x28 px and the classifier head sees a feature cell covering
several glyph pixels. Cropping and upscaling to 64x64 gives the classifier ~6x
the linear detail on the thing it actually has to identify.

Crops are taken with **context padding** around the box. A tight crop discards
the local background that distinguishes a pale stroke from a distractor line, and
it also makes the classifier brittle to the detector's box being a few pixels
off -- which it routinely is.

Training crops come from ground-truth boxes; the classifier is later applied to
*detector* boxes, which are noisier. Jitter augmentation during training covers
that gap (see scripts/train_classifier.py).

Output: <out>/{train,val}/<class_name>/<stem>_<i>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Characters that are illegal in Windows filenames; the label space includes '+'
# and '-' which are fine, but this guards against future additions.
_SAFE = {"+": "_plus", "-": "_minus"}


def safe_dirname(name: str) -> str:
    """Case-sensitive-safe directory name.

    Windows filesystems are case-insensitive, so 'a' and 'A' would collide into
    one directory and silently merge two classes -- which would look like a
    classifier that cannot tell case apart. Upper case gets an explicit suffix.
    """
    if name in _SAFE:
        return _SAFE[name]
    return f"{name}_upper" if name.isupper() else name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "datasets" / "project-497" / "yolo3"))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "crops"))
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--pad", type=float, default=0.25,
                    help="context padding as a fraction of box size")
    args = ap.parse_args()

    import cv2
    import yaml

    src = Path(args.src)
    out = Path(args.out)
    meta = yaml.safe_load((src / "data.yaml").read_text())
    names = [meta["names"][i] for i in range(len(meta["names"]))]

    counts = {}
    for split in ("train", "val", "test"):
        img_dir = src / "images" / split
        if not img_dir.is_dir():
            continue
        n = 0
        for p in sorted(img_dir.iterdir()):
            if not p.is_file():
                continue
            lbl = src / "labels" / split / (p.stem + ".txt")
            if not lbl.exists():
                continue
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            for i, ln in enumerate(lbl.read_text().splitlines()):
                if not ln.strip():
                    continue
                c, cx, cy, bw, bh = ln.split()
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                px, py = bw * args.pad, bh * args.pad
                x1 = max(0, int(cx - bw / 2 - px))
                y1 = max(0, int(cy - bh / 2 - py))
                x2 = min(w, int(cx + bw / 2 + px))
                y2 = min(h, int(cy + bh / 2 + py))
                if x2 - x1 < 3 or y2 - y1 < 3:
                    continue
                crop = cv2.resize(img[y1:y2, x1:x2], (args.size, args.size),
                                  interpolation=cv2.INTER_CUBIC)
                d = out / split / safe_dirname(names[int(c)])
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{p.stem}_{i}.png"), crop)
                n += 1
        counts[split] = n

    info = {"source": str(src), "size": args.size, "pad": args.pad,
            "classes": len(names), "class_dirs": [safe_dirname(n) for n in names],
            "names": names, **counts}
    out.mkdir(parents=True, exist_ok=True)
    (out / "crops_meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items()
                      if k not in ("class_dirs", "names")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
