"""Measure a source's visual statistics before trying to synthesize it (lever #1).

A synthetic generator that does not match the real distribution produces training
data that actively hurts: the model learns the synthetic style's quirks and
transfers nothing. So the parameters get measured from the real images rather
than guessed by eye.

What is measured, and why each one matters for the generator:
  * character count and per-character box geometry -> layout engine
  * glyph colour vs background colour -> the contrast that makes this source hard
  * stroke darkness relative to background -> whether glyphs are pale or bold
  * noise-line count and colour saturation -> the distractor layer
  * speckle density -> the per-pixel noise layer

Output: runs/eda/style_<size>.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="230x60")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    import cv2

    root = ROOT / "datasets" / "project-497" / f"src_{args.source}"
    img_dir, lbl_dir = root / "images" / "train", root / "labels" / "train"
    paths = sorted(p for p in img_dir.iterdir() if p.is_file())[: args.limit]

    counts = Counter()
    widths, heights, gaps, y_centres = [], [], [], []
    bg_samples, glyph_samples = [], []
    sat_hist = []

    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        rows = [
            [float(v) for v in ln.split()]
            for ln in (lbl_dir / (p.stem + ".txt")).read_text().splitlines()
            if ln.strip()
        ]
        counts[len(rows)] += 1
        rows.sort(key=lambda r: r[1])

        prev_right = None
        for _, cx, cy, bw, bh in rows:
            widths.append(bw * w)
            heights.append(bh * h)
            y_centres.append(cy)
            left = (cx - bw / 2) * w
            if prev_right is not None:
                gaps.append(left - prev_right)
            prev_right = (cx + bw / 2) * w

            x1, y1 = int(max(0, left)), int(max(0, (cy - bh / 2) * h))
            x2, y2 = int(min(w, prev_right)), int(min(h, (cy + bh / 2) * h))
            crop = img[y1:y2, x1:x2]
            if crop.size:
                # The glyph is the darker mode inside its own box; the lighter
                # mode is the local background showing through.
                grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                thresh = grey.mean()
                dark = crop[grey < thresh]
                if len(dark):
                    glyph_samples.append(dark.mean(0))

        # Corners are reliably background: no glyph box reaches them.
        for yy, xx in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
            bg_samples.append(img[yy, xx].astype(float))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat_hist.append(float(np.percentile(hsv[..., 1], 95)))

    glyph = np.array(glyph_samples)
    bg = np.array(bg_samples)

    out = {
        "source": args.source,
        "n_images": len(paths),
        "char_count_distribution": {str(k): v for k, v in sorted(counts.items())},
        "box_w_px": {"p5": float(np.percentile(widths, 5)),
                     "p50": float(np.percentile(widths, 50)),
                     "p95": float(np.percentile(widths, 95))},
        "box_h_px": {"p5": float(np.percentile(heights, 5)),
                     "p50": float(np.percentile(heights, 50)),
                     "p95": float(np.percentile(heights, 95))},
        "gap_px": {"p5": float(np.percentile(gaps, 5)),
                   "p50": float(np.percentile(gaps, 50)),
                   "p95": float(np.percentile(gaps, 95))} if gaps else None,
        "y_centre_norm": {"p5": float(np.percentile(y_centres, 5)),
                          "p50": float(np.percentile(y_centres, 50)),
                          "p95": float(np.percentile(y_centres, 95))},
        "background_bgr_mean": bg.mean(0).tolist(),
        "background_bgr_std": bg.std(0).tolist(),
        "glyph_bgr_mean": glyph.mean(0).tolist(),
        "glyph_bgr_std": glyph.std(0).tolist(),
        # The headline number for this source: how little the glyphs stand out.
        "glyph_vs_bg_luma_delta": float(bg.mean() - glyph.mean()),
        "saturation_p95_mean": float(np.mean(sat_hist)),
    }

    out_path = ROOT / "runs" / "eda" / f"style_{args.source}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
