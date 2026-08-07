"""Diagnose WHERE the gate teacher's character errors come from.

The feasibility gate failed (43% solve rate vs a 95% target). Before concluding the
target is unreachable, we need to know whether the errors are model capacity --
fixable with resolution/training -- or label ambiguity, which no model can fix.

The specific worry: this label space is case-sensitive and contains pairs whose
glyphs are identical up to scale (C/c, O/o, S/s, U/u, V/v, W/w, X/x, Z/z, P/p,
K/k, J/j), plus the classic 0/O/o and 1/l/I families. In a cropped CAPTCHA glyph
with no baseline reference, several of these are genuinely undecidable -- a human
annotator cannot reliably separate them either. If those pairs dominate the
confusions, then 95% on THIS label space is not achievable by any model, and the
right response is to change the label space rather than the architecture.

Matches predictions to ground truth by IoU, then tabulates the confusion pairs.

Output: runs/eval/confusion.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets" / "project-497" / "yolo"

# Glyph pairs that are identical or near-identical in shape, differing mainly by
# size//stroke weight -- the cases where the label itself is questionable.
AMBIGUOUS_GROUPS = [
    set("Cc"), set("Oo0"), set("Ss"), set("Uu"), set("Vv"), set("Ww"),
    set("Xx"), set("Zz"), set("Pp"), set("Kk"), set("Jj"), set("Ii1lL"),
    set("Mm"), set("Nn"), set("Yy"), set("Tt"), set("Ff"), set("Rr"),
    set("Aa"), set("Bb"), set("Dd"), set("Ee"), set("Gg"), set("Hh"),
    set("Qq"), set("2Zz"), set("5Ss"), set("6b"), set("9g"), set("8B"),
]


def _same_group(a: str, b: str) -> bool:
    return any(a in g and b in g for g in AMBIGUOUS_GROUPS)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs" / "feasibility" /
                                             "feasibility_m_160" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=str(ROOT / "runs" / "eval" / "confusion.json"))
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = model.names

    img_dir = DS / "images" / "val"
    lbl_dir = DS / "labels" / "val"
    images = sorted(p for p in img_dir.iterdir() if p.is_file())

    confusions = Counter()
    matched = correct = 0
    unmatched_gt = spurious = 0

    preds = model.predict(source=[str(p) for p in images], imgsz=args.imgsz,
                          conf=args.conf, device=args.device, verbose=False, stream=True)
    for img_path, r in zip(images, preds):
        with Image.open(img_path) as im:
            W, H = im.size
        gt = []
        for ln in (lbl_dir / (img_path.stem + ".txt")).read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            c, cx, cy, w, h = ln.split()
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            gt.append((names[int(c)], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))

        pred = [(names[int(c)], tuple(xy))
                for c, xy in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]

        used = set()
        for gname, gbox in gt:
            best_i, best_v = -1, 0.0
            for i, (_, pbox) in enumerate(pred):
                if i in used:
                    continue
                v = _iou(gbox, pbox)
                if v > best_v:
                    best_i, best_v = i, v
            if best_i >= 0 and best_v >= 0.5:
                used.add(best_i)
                matched += 1
                pname = pred[best_i][0]
                if pname == gname:
                    correct += 1
                else:
                    confusions[(gname, pname)] += 1
            else:
                unmatched_gt += 1
        spurious += len(pred) - len(used)

    total_conf = sum(confusions.values())
    ambiguous = sum(v for (g, p), v in confusions.items() if _same_group(g, p))
    case_only = sum(v for (g, p), v in confusions.items() if g.lower() == p.lower())

    out = {
        "weights": args.weights,
        "conf": args.conf,
        "matched_boxes": matched,
        "classification_accuracy_on_matched": round(correct / matched, 4) if matched else 0,
        "missed_gt_boxes": unmatched_gt,
        "spurious_boxes": spurious,
        "total_confusions": total_conf,
        "confusions_case_only": case_only,
        "confusions_case_only_pct": round(100 * case_only / total_conf, 1) if total_conf else 0,
        "confusions_shape_ambiguous": ambiguous,
        "confusions_shape_ambiguous_pct": round(100 * ambiguous / total_conf, 1) if total_conf else 0,
        "top_confusions": [
            {"gt": g, "pred": p, "n": n, "case_only": g.lower() == p.lower()}
            for (g, p), n in confusions.most_common(30)
        ],
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "top_confusions"}, indent=2))
    print("\nTop confusions (gt -> pred):")
    for c in out["top_confusions"][:20]:
        flag = "  <- case-only" if c["case_only"] else ""
        print(f"  {c['gt']!r} -> {c['pred']!r}: {c['n']}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
