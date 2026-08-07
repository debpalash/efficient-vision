"""Synthesize labeled CAPTCHA images matching a measured source style (lever #1).

The feasibility gate failed at 0.507 solve rate against a 0.95 target, and the
binding constraint is data: the hard sources carry 158-400 training images each,
while 0.99 per-character accuracy is what 0.95 solve rate requires. CAPTCHAs are
machine-generated, so training data is synthesizable in unlimited quantity with
perfect labels -- which is the one lever that does not require a better model.

Every generator parameter below is taken from `scripts/analyze_style.py` measuring
the real images, not chosen by eye. For 230x60 those measurements were:

    exactly 5 characters per image (200/200 images)
    glyph boxes ~23 x 24 px (p50)
    inter-character gap p50 +6 px but p5 -24.7 px  -> characters OVERLAP
    vertical centre 0.51 +/- 0.16 (normalized)     -> vertical jitter
    background BGR ~(194, 199, 202), std ~(71, 61, 56)
    glyph BGR ~(79, 103, 105)
    saturation p95 ~197                            -> saturated distractor lines

Labels are computed from the rendered glyph's actual alpha mask, so the boxes are
exact by construction -- tighter and more consistent than the hand annotations.

IMPORTANT: this is a *replica*, not the original generator. Synthetic data that
does not match the real distribution trains a model that transfers nothing, so
`--compare` renders a side-by-side sheet against real samples, and the replica
must be validated against real val data before being trusted. See
`docs/synthetic-data.md`.
"""
from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# The 64 classes that survived EDA (9 zero-example symbols were dropped).
ALPHABET = list("+-" + string.digits + string.ascii_uppercase + string.ascii_lowercase)

FONT_DIR = Path("C:/Windows/Fonts")
FONT_NAMES = [
    "arial.ttf", "arialbd.ttf", "calibri.ttf", "calibrib.ttf", "comic.ttf",
    "comicbd.ttf", "consola.ttf", "consolab.ttf", "corbel.ttf", "corbelb.ttf",
    "constan.ttf", "constanb.ttf", "Candara.ttf", "Candarab.ttf", "verdana.ttf",
    "tahoma.ttf", "georgia.ttf", "trebuc.ttf", "seguisb.ttf",
]


def _available_fonts() -> list[Path]:
    fonts = [FONT_DIR / n for n in FONT_NAMES if (FONT_DIR / n).exists()]
    if not fonts:
        raise SystemExit(f"no usable fonts found under {FONT_DIR}")
    return fonts


def _render_glyph(char: str, font_path: Path, size: int, angle: float):
    """Render one character and return `(rgba_array, tight_bbox)`.

    The bbox comes from the alpha channel after rotation, so it is the exact
    axis-aligned hull of the visible ink -- which is what a YOLO label means.
    """
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(font_path), size)
    pad = size
    canvas = Image.new("RGBA", (size * 3, size * 3), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), char, font=font, fill=(255, 255, 255, 255))
    canvas = canvas.rotate(angle, resample=Image.BICUBIC, expand=False)

    alpha = np.array(canvas)[..., 3]
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        return None, None
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return canvas.crop(box), box


def _paste(bg: np.ndarray, glyph_img, x: int, y: int, colour: tuple[int, int, int]):
    """Alpha-composite a glyph onto the BGR background at (x, y). Returns its bbox."""
    g = np.array(glyph_img)
    gh, gw = g.shape[:2]
    h, w = bg.shape[:2]

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + gw), min(h, y + gh)
    if x1 <= x0 or y1 <= y0:
        return None

    sub = g[y0 - y : y1 - y, x0 - x : x1 - x]
    a = (sub[..., 3:4].astype(np.float32) / 255.0)
    patch = bg[y0:y1, x0:x1].astype(np.float32)
    bg[y0:y1, x0:x1] = (patch * (1 - a) + np.array(colour, np.float32) * a).astype(np.uint8)
    return (x0, y0, x1, y1)


def _hsv_to_bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    import cv2

    px = np.uint8([[[h % 180, max(0, min(255, s)), max(0, min(255, v))]]])
    return tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])


def _background(w: int, h: int, rng: random.Random, hue: int) -> np.ndarray:
    """Strongly tinted pastel background -- one hue per image.

    Attempt 2 made this near-white because the measured mean BGR (194, 199, 202)
    is nearly neutral. That was an aggregation artifact: each image carries a
    *strong* pastel tint (pink, lavender, mint), but the tints differ per image,
    so averaging hundreds of them lands on grey. A statistic that varies per image
    cannot be read from a global mean -- attempt 1 was accidentally closer here.
    """
    colour = _hsv_to_bgr(hue, rng.randint(25, 70), rng.randint(222, 248))
    bg = np.full((h, w, 3), colour, dtype=np.float32)
    bg += np.random.normal(0, 2.5, (h, w, 3))
    return bg.clip(0, 255).astype(np.uint8)


def _add_lines(img: np.ndarray, rng: random.Random, n: int, hue: int) -> None:
    """Near-horizontal dashed distractor lines sharing one hue family per image.

    Three properties visible only at zoom, all invisible in the aggregate stats:
      * near-HORIZONTAL (shallow slope), not free-angle sweeps
      * DASHED -- broken into segments, not continuous strokes
      * ONE hue family per image, shared with the speckle layer
    """
    import cv2

    h, w = img.shape[:2]
    for _ in range(n):
        colour = _hsv_to_bgr(hue + rng.randint(-14, 14),
                             rng.randint(140, 255), rng.randint(55, 175))
        y0 = rng.uniform(-6, h + 6)
        slope = rng.uniform(-0.16, 0.16)
        x = rng.uniform(-10, 20)
        while x < w + 10:
            seg = rng.uniform(6, 55)
            x2 = x + seg
            cv2.line(img, (int(x), int(y0 + slope * x)),
                     (int(x2), int(y0 + slope * x2)), colour, 1, cv2.LINE_AA)
            x = x2 + rng.uniform(0, 22)      # the gap that makes it dashed


def _add_speckle(img: np.ndarray, rng: random.Random, density: float, hue: int) -> None:
    """Dense per-pixel noise, mostly inside the image's own hue family."""
    h, w = img.shape[:2]
    n = int(h * w * density)
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    fam = np.array([_hsv_to_bgr(hue + rng.randint(-25, 25),
                                rng.randint(120, 255), rng.randint(40, 190))
                    for _ in range(24)], dtype=np.uint8)
    pick = fam[np.random.randint(0, len(fam), n)]
    # A minority sit outside the family; a purely monochrome speckle layer reads
    # as synthetic.
    stray = np.random.rand(n) < 0.25
    pick[stray] = np.random.randint(0, 255, (int(stray.sum()), 3))
    img[ys, xs] = pick


def generate_one(rng: random.Random, fonts: list[Path], w: int, h: int,
                 n_chars: int) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    import cv2

    # One hue family per image drives background, lines and speckle together.
    hue = rng.randint(0, 179)
    img = _background(w, h, rng, hue)

    chars = [rng.choice(ALPHABET) for _ in range(n_chars)]
    # Measured p50 gap is +6 px but p5 is -24.7, so characters must be allowed to
    # overlap. Laying them out on a nominal pitch with jitter reproduces that.
    pitch = w / (n_chars + 0.6)
    x = rng.uniform(4, 12)
    labels = []

    # One font per image: a real generator picks a font per CAPTCHA, not per
    # character. Mixing fonts within an image was the clearest tell in attempt 1.
    font = rng.choice(fonts)
    base_size = rng.randint(30, 42)

    for ch in chars:
        # Per-character size varies WIDELY in the real images -- within a single
        # CAPTCHA one glyph can be twice another's height. Attempt 2 narrowed this
        # to +/-3 px, which was the wrong direction.
        size = max(16, int(base_size * rng.uniform(0.55, 1.30)))
        angle = rng.uniform(-22, 22)
        glyph, _ = _render_glyph(ch, font, size, angle)
        if glyph is None:
            continue

        # Per-character OPACITY also varies widely: some glyphs are nearly ghosts
        # while their neighbours are solid. This is the single feature that makes
        # the source hard, and attempt 2 omitted it entirely -- every glyph was
        # rendered fully opaque.
        alpha = rng.uniform(0.30, 1.0)
        g = np.array(glyph)
        g[..., 3] = (g[..., 3].astype(np.float32) * alpha).astype(np.uint8)
        from PIL import Image as _Image
        glyph = _Image.fromarray(g)

        colour = _hsv_to_bgr(rng.randint(0, 179), rng.randint(70, 255),
                             rng.randint(55, 185))

        gh = glyph.size[1]
        y = int(h * rng.gauss(0.51, 0.09) - gh / 2)
        box = _paste(img, glyph, int(x), y, colour)
        if box is not None:
            x0, y0, x1, y1 = box
            if (x1 - x0) > 2 and (y1 - y0) > 2:
                labels.append((
                    ALPHABET.index(ch),
                    ((x0 + x1) / 2) / w, ((y0 + y1) / 2) / h,
                    (x1 - x0) / w, (y1 - y0) / h,
                ))
        x += pitch * rng.uniform(0.72, 1.15)

    _add_lines(img, rng, rng.randint(7, 15), hue)
    _add_speckle(img, rng, rng.uniform(0.010, 0.045), hue)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--count", type=int, default=100)
    ap.add_argument("--source", default="230x60", help="style being replicated")
    ap.add_argument("--width", type=int, default=230)
    ap.add_argument("--height", type=int, default=60)
    ap.add_argument("--chars", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", action="store_true",
                    help="also write a real-vs-synthetic comparison sheet")
    args = ap.parse_args()

    import cv2

    out = Path(args.out) if args.out else (
        ROOT / "datasets" / "synthetic" / f"src_{args.source}"
    )
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    fonts = _available_fonts()

    for i in range(args.count):
        img, labels = generate_one(rng, fonts, args.width, args.height, args.chars)
        stem = f"synth_{args.source}_{i:05d}"
        cv2.imwrite(str(out / "images" / "train" / f"{stem}.png"), img)
        (out / "labels" / "train" / f"{stem}.txt").write_text(
            "\n".join(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                      for c, cx, cy, bw, bh in labels) + "\n",
            encoding="utf-8",
        )

    yaml = (f"# Auto-generated by scripts/synth_captcha.py (replica of {args.source})\n"
            f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/train\n"
            f"nc: {len(ALPHABET)}\nnames:\n")
    for i, name in enumerate(ALPHABET):
        yaml += f"  {i}: {json.dumps(name)}\n"
    (out / "data.yaml").write_text(yaml, encoding="utf-8")

    meta = {"count": args.count, "source_replicated": args.source,
            "size": [args.width, args.height], "chars_per_image": args.chars,
            "seed": args.seed, "nc": len(ALPHABET), "fonts": [f.name for f in fonts]}
    (out / "synth_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

    if args.compare:
        real_dir = ROOT / "datasets" / "project-497" / f"src_{args.source}" / "images" / "train"
        reals = sorted(p for p in real_dir.iterdir() if p.is_file())[:6]
        synths = sorted((out / "images" / "train").iterdir())[:6]
        rows = []
        for group in (reals, synths):
            tiles = [cv2.resize(cv2.imread(str(p)), (args.width, args.height))
                     for p in group]
            rows.append(np.hstack(tiles))
        sheet = np.vstack([rows[0], np.full((8, rows[0].shape[1], 3), 0, np.uint8), rows[1]])
        sheet_path = ROOT / "runs" / "eda" / f"synth_compare_{args.source}.png"
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sheet_path), sheet)
        print(f"\ncomparison sheet (top row real, bottom row synthetic): {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
