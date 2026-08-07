"""Compare standard threshold decoding against count-prior decoding (task #24).

This is a **decode-time** change: same weights, same detections, different
decision about which boxes constitute the answer. So it costs one evaluation pass
rather than a training run, and any gain is free at deployment apart from a
negligible search over ~5 candidate counts.

The count prior is measured from the **training** labels only. Fitting it on val
would leak the answer distribution into the metric.

Output: runs/eval/count_prior_<name>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "datasets" / "project-497" / "yolo3"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="base3_320")
    ap.add_argument("--base-conf", type=float, default=0.05)
    ap.add_argument("--std-confs", default="0.25,0.30,0.40")
    ap.add_argument("--gammas", default="1.0,0.7,0.5,0.35,0.25,0.15")
    ap.add_argument("--chunk", type=int, default=64,
                    help="images per predict call; bounds peak VRAM")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    from efficientvision.eval.decode import CountPrior, decode_with_count_prior
    from efficientvision.eval.harness import decode_to_string

    root = Path(args.data_dir)
    model = YOLO(args.weights)
    names_map = model.names
    names = [names_map[i] for i in range(len(names_map))]

    prior = CountPrior.from_label_dir(root / "labels" / "train")
    print("count prior (from TRAIN labels):")
    for k in prior.support:
        print(f"  {k}: {torch.tensor(prior.logp(k)).exp():.4f}")

    img_dir, lbl_dir = root / "images" / args.split, root / "labels" / args.split
    images = sorted(p for p in img_dir.iterdir() if p.is_file())

    gts = []
    for p in images:
        items = []
        for ln in (lbl_dir / (p.stem + ".txt")).read_text().splitlines():
            if ln.strip():
                q = ln.split()
                items.append((float(q[1]), names[int(q[0])]))
        items.sort(key=lambda t: t[0])
        gts.append("".join(c for _, c in items))

    # One inference pass at the permissive floor; both decoders consume the same
    # detections, so the comparison isolates the decode rule.
    # Chunked: handing Ultralytics the whole file list lets it build one oversized
    # batch, which OOMs at 448px on an 8 GB card. Chunk size bounds peak VRAM
    # independently of split size and resolution.
    all_boxes, all_scores = [], []
    for start in range(0, len(images), args.chunk):
        batch = [str(p) for p in images[start:start + args.chunk]]
        for r in model.predict(source=batch, imgsz=args.imgsz, conf=args.base_conf,
                               iou=0.95, device=args.device, verbose=False,
                               stream=True):
            b = r.boxes
            s = torch.zeros(len(b.cls), len(names))
            for i, (c, cf) in enumerate(zip(b.cls.tolist(), b.conf.tolist())):
                s[i, int(c)] = cf
            all_boxes.append(b.xyxy.cpu())
            all_scores.append(s)

    def score(decoder) -> dict:
        exact = count_ok = over = under = 0
        edits = chars = 0
        for gt, bx, sc in zip(gts, all_boxes, all_scores):
            pred = decoder(bx, sc)
            exact += pred == gt
            count_ok += len(pred) == len(gt)
            over += len(pred) > len(gt)
            under += len(pred) < len(gt)
            edits += levenshtein(pred, gt)
            chars += len(gt)
        n = len(gts)
        return {
            "solve_rate": round(exact / n, 4),
            "char_accuracy": round(1 - edits / max(chars, 1), 4),
            "count_match": round(count_ok / n, 4),
            "too_many_boxes": round(over / n, 4),
            "too_few_boxes": round(under / n, 4),
        }

    results = {"standard": {}, "count_prior": {}}
    for c in [float(x) for x in args.std_confs.split(",")]:
        results["standard"][f"conf{c:.2f}"] = score(
            lambda bx, sc, c=c: decode_to_string(bx, sc, names, c, 0.7)
        )
        print(f"standard   conf={c:.2f}  {results['standard'][f'conf{c:.2f}']}")

    # gamma calibrates raw detector confidence into P(is a character); see
    # efficientvision.eval.decode.calibrate. Swept because the right value depends
    # on how the detector was trained, not on anything knowable a priori.
    for g in [float(x) for x in args.gammas.split(",")]:
        r = score(lambda bx, sc, g=g: decode_with_count_prior(
            bx, sc, names, prior, base_conf=args.base_conf, gamma=g))
        results["count_prior"][f"gamma{g:.2f}"] = r
        print(f"count_prior gamma={g:.2f}  {r}")

    cp = max(results["count_prior"].values(), key=lambda r: r["solve_rate"])
    best_std = max(results["standard"].values(), key=lambda r: r["solve_rate"])
    out = {
        "weights": args.weights, "split": args.split, "n_images": len(images),
        "count_prior_support": {str(k): round(float(torch.tensor(prior.logp(k)).exp()), 4)
                                for k in prior.support},
        "best_standard": best_std,
        "count_prior": cp,
        "delta_solve_rate": round(cp["solve_rate"] - best_std["solve_rate"], 4),
        "all": results,
    }
    outp = ROOT / "runs" / "eval" / f"count_prior_{args.name}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nbest standard : {best_std['solve_rate']}")
    print(f"count prior   : {cp['solve_rate']}")
    print(f"delta         : {out['delta_solve_rate']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
