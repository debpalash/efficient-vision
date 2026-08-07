"""Product-level evaluation: per-image string-solve rate (task #15).

mAP is a per-box metric and is the wrong contract for this system. A CAPTCHA is
only solved if EVERY character is detected, classified, and ordered correctly --
so solve rate is strictly harsher than mAP50. At 5 characters per image, even 98%
per-character accuracy yields only ~90% solve rate. Reporting mAP alone would
overstate readiness.

Metrics reported, from harshest to most forgiving:
  - exact_match     : predicted string == ground-truth string (THE contract metric)
  - cer             : character error rate (Levenshtein / GT length)
  - char_accuracy   : 1 - cer
  - count_match     : right NUMBER of characters (isolates detection from classification)

The confidence threshold materially changes solve rate (a spurious extra box
breaks the whole string), so this sweeps thresholds and reports the best
operating point rather than assuming a default.

Output: runs/eval/solve_rate.json
"""
from __future__ import annotations

import argparse
import json
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


def _read_gt(label_path: Path, names: dict) -> str:
    """Ground-truth string, characters ordered left-to-right by box center."""
    items = []
    for ln in label_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split()
        items.append((float(p[1]), names[int(p[0])]))
    items.sort(key=lambda t: t[0])
    return "".join(c for _, c in items)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs" / "feasibility" /
                                             "feasibility_m_160" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=str(ROOT / "runs" / "eval" / "solve_rate.json"))
    ap.add_argument("--confs", default="0.1,0.2,0.25,0.3,0.4,0.5,0.6")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on N images (0=all)")
    # Duplicate/missing boxes -- not misclassification -- dominate solve-rate loss,
    # and both are decode-side knobs. Sweeping them separates a genuinely wrong
    # model from a merely mis-tuned decoder.
    ap.add_argument("--ious", default="0.5,0.7")
    ap.add_argument("--data-dir", default=str(DS),
                    help="split root holding images/val and labels/val; "
                         "point at a src_* tree to score one CAPTCHA source")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = model.names

    root = Path(args.data_dir)
    img_dir = root / "images" / "val"
    lbl_dir = root / "labels" / "val"
    images = sorted(p for p in img_dir.iterdir() if p.is_file())
    if args.limit:
        images = images[: args.limit]
    gts = [_read_gt(lbl_dir / (p.stem + ".txt"), names) for p in images]

    results = {}
    for iou in [float(i) for i in args.ious.split(",")]:
        for conf in [float(c) for c in args.confs.split(",")]:
            exact = count_ok = over = under = 0
            edits = gt_chars = 0
            preds = model.predict(source=[str(p) for p in images], imgsz=args.imgsz,
                                  conf=conf, iou=iou, device=args.device,
                                  verbose=False, stream=True)
            for gt, r in zip(gts, preds):
                b = r.boxes
                items = sorted(
                    zip(b.xywh[:, 0].tolist(), [names[int(c)] for c in b.cls.tolist()]),
                    key=lambda t: t[0],
                )
                pred = "".join(c for _, c in items)
                exact += pred == gt
                count_ok += len(pred) == len(gt)
                over += len(pred) > len(gt)
                under += len(pred) < len(gt)
                edits += levenshtein(pred, gt)
                gt_chars += len(gt)

            n = len(images)
            key = f"iou{iou:.2f}_conf{conf:.2f}"
            results[key] = {
                "iou": iou,
                "conf": conf,
                "exact_match": round(exact / n, 4),
                "count_match": round(count_ok / n, 4),
                # Split the count failures: too many boxes points at NMS/duplicates,
                # too few at missed detections. They need opposite fixes.
                "too_many_boxes": round(over / n, 4),
                "too_few_boxes": round(under / n, 4),
                "cer": round(edits / gt_chars, 4),
                "char_accuracy": round(1 - edits / gt_chars, 4),
            }
            print(f"iou={iou:.2f} conf={conf:.2f}  solve={exact/n:.4f}  "
                  f"char_acc={1-edits/gt_chars:.4f}  count={count_ok/n:.4f}  "
                  f"over={over/n:.3f} under={under/n:.3f}")

    best_conf = max(results, key=lambda k: results[k]["exact_match"])
    out = {
        "weights": args.weights,
        "imgsz": args.imgsz,
        "data_dir": str(root),
        "n_val_images": len(images),
        "contract_metric": "exact_match (per-image string solve rate)",
        "by_conf": results,
        "best_conf": best_conf,
        "best": results[best_conf],
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nBEST:", json.dumps({"conf": best_conf, **results[best_conf]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
