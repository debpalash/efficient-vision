"""Dataset EDA for the CAPTCHA character-detection set (task #2).

Emits a JSON summary under runs/eda/ and prints a terse digest. No plots —
per repo rules, the report is a committed markdown table built from this JSON.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets" / "project-497"
IMG_DIR = DS / "images"
LBL_DIR = DS / "labels"
OUT = ROOT / "runs" / "eda"


def main() -> int:
    classes = (DS / "classes.txt").read_text(encoding="utf-8").splitlines()
    n_classes = len(classes)

    class_counts = Counter()
    boxes_per_img = []
    box_w = []
    box_h = []
    malformed = []
    empty_labels = 0
    token_widths = Counter()

    label_files = sorted(LBL_DIR.glob("*.txt"))
    for lf in label_files:
        lines = [ln for ln in lf.read_text().splitlines() if ln.strip()]
        if not lines:
            empty_labels += 1
        boxes_per_img.append(len(lines))
        for ln in lines:
            parts = ln.split()
            token_widths[len(parts)] += 1
            if len(parts) != 5:
                malformed.append(lf.name)
                continue
            cid = int(parts[0])
            class_counts[cid] += 1
            box_w.append(float(parts[3]))
            box_h.append(float(parts[4]))

    # image size sampling (every Nth to keep it cheap)
    img_files = sorted(IMG_DIR.glob("*"))
    sizes = Counter()
    exts = Counter()
    step = max(1, len(img_files) // 400)
    for i, im in enumerate(img_files):
        exts[im.suffix.lower()] += 1
        if i % step == 0:
            try:
                with Image.open(im) as img:
                    sizes[img.size] += 1
            except Exception:
                pass

    def pct(vals, p):
        if not vals:
            return None
        s = sorted(vals)
        k = int(round((p / 100) * (len(s) - 1)))
        return s[k]

    counts_sorted = sorted(class_counts.values())
    per_class = {classes[cid]: class_counts.get(cid, 0) for cid in range(n_classes)}
    missing = [classes[cid] for cid in range(n_classes) if class_counts.get(cid, 0) == 0]

    summary = {
        "n_images": len(img_files),
        "n_label_files": len(label_files),
        "n_classes": n_classes,
        "empty_labels": empty_labels,
        "malformed_count": len(malformed),
        "token_width_histogram": dict(token_widths),
        "total_boxes": sum(class_counts.values()),
        "boxes_per_image": {
            "min": min(boxes_per_img) if boxes_per_img else 0,
            "p50": pct(boxes_per_img, 50),
            "p90": pct(boxes_per_img, 90),
            "max": max(boxes_per_img) if boxes_per_img else 0,
            "mean": round(sum(boxes_per_img) / len(boxes_per_img), 2) if boxes_per_img else 0,
        },
        "box_w_norm": {"p5": pct(box_w, 5), "p50": pct(box_w, 50), "p95": pct(box_w, 95)},
        "box_h_norm": {"p5": pct(box_h, 5), "p50": pct(box_h, 50), "p95": pct(box_h, 95)},
        "class_count": {
            "min": counts_sorted[0] if counts_sorted else 0,
            "p50": pct(counts_sorted, 50),
            "max": counts_sorted[-1] if counts_sorted else 0,
            "imbalance_ratio": round(counts_sorted[-1] / counts_sorted[0], 1) if counts_sorted and counts_sorted[0] else None,
        },
        "classes_missing": missing,
        "classes_under_50": [classes[cid] for cid in range(n_classes) if 0 < class_counts.get(cid, 0) < 50],
        "image_ext_histogram": {k: v for k, v in exts.items()},
        "image_size_sample": {f"{w}x{h}": c for (w, h), c in sizes.most_common(12)},
        "per_class_count": per_class,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    d = summary
    print(f"images={d['n_images']} boxes={d['total_boxes']} classes={d['n_classes']}")
    print(f"token_widths={d['token_width_histogram']} malformed={d['malformed_count']} empty={d['empty_labels']}")
    print(f"boxes/img: {d['boxes_per_image']}")
    print(f"box_w_norm={d['box_w_norm']} box_h_norm={d['box_h_norm']}")
    print(f"class_count={d['class_count']}")
    print(f"missing={d['classes_missing']}")
    print(f"under_50={d['classes_under_50']}")
    print(f"sizes={d['image_size_sample']}")
    print(f"exts={d['image_ext_histogram']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
