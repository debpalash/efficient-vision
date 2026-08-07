"""End-to-end evaluation of the two-stage pipeline (task #23).

Stage 1 detector supplies boxes; stage 2 classifier reads each box as an upscaled
64x64 crop. The two roles factor cleanly:

  * detector confidence  -> P(a character is here)      [used for the count decision]
  * classifier softmax   -> P(which character it is)    [used for the identity]

That factorization is the actual argument for two stages. The single-stage model
predicts both from one ~24x28 px region; here the identity decision gets ~6x the
linear detail, and the count decision keeps the detector's objectness, which is
what it is good at.

Four configurations are reported so the gain (or loss) is attributable:

  1. single-stage, standard threshold decode      -- the original baseline
  2. single-stage, count-prior decode             -- current best
  3. two-stage, standard threshold decode
  4. two-stage, count-prior decode                -- the full stack

Output: runs/eval/two_stage_<name>.json
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
    ap.add_argument("--detector", default=str(
        ROOT / "runs" / "feasibility" / "base3_320" / "weights" / "best.pt"))
    ap.add_argument("--classifier", default=str(
        ROOT / "runs" / "classifier" / "char_cls" / "best.pt"))
    ap.add_argument("--data-dir", default=str(ROOT / "datasets" / "project-497" / "yolo3"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--base-conf", type=float, default=0.05)
    ap.add_argument("--pad", type=float, default=0.25)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--gammas", default="0.5,0.35,0.25")
    ap.add_argument("--std-confs", default="0.25,0.30")
    ap.add_argument("--name", default="base3_320")
    args = ap.parse_args()

    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18
    from ultralytics import YOLO

    from efficientvision.eval.decode import CountPrior, decode_with_count_prior
    from efficientvision.eval.harness import decode_to_string
    from scripts.build_crops import safe_dirname

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = Path(args.data_dir)

    det = YOLO(args.detector)
    names = [det.names[i] for i in range(len(det.names))]

    ckpt = torch.load(args.classifier, map_location=device, weights_only=False)
    clf = resnet18()
    clf.fc = nn.Linear(clf.fc.in_features, len(ckpt["classes"]))
    clf.load_state_dict(ckpt["model"])
    clf = clf.to(device).eval()
    csize = ckpt["size"]

    # ImageFolder sorts class directories alphabetically, which is NOT the YOLO
    # class-id order. Mapping through the sanitized directory names keeps the two
    # label spaces aligned; getting this wrong silently permutes every prediction.
    dir_to_idx = {d: i for i, d in enumerate(ckpt["classes"])}
    yolo_to_clf = [dir_to_idx[safe_dirname(n)] for n in names]
    clf_to_yolo = [0] * len(ckpt["classes"])
    for y, c in enumerate(yolo_to_clf):
        clf_to_yolo[c] = y

    prior = CountPrior.from_label_dir(root / "labels" / "train")
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

    single_boxes, single_scores, two_scores = [], [], []
    for start in range(0, len(images), args.chunk):
        chunk = images[start:start + args.chunk]
        preds = list(det.predict(source=[str(p) for p in chunk], imgsz=args.imgsz,
                                 conf=args.base_conf, iou=0.95, device=args.device,
                                 verbose=False, stream=True))
        for path, r in zip(chunk, preds):
            b = r.boxes
            xyxy = b.xyxy.cpu()
            n = len(b.cls)

            s1 = torch.zeros(n, len(names))
            for i, (c, cf) in enumerate(zip(b.cls.tolist(), b.conf.tolist())):
                s1[i, int(c)] = cf
            single_boxes.append(xyxy)
            single_scores.append(s1)

            if n == 0:
                two_scores.append(torch.zeros(0, len(names)))
                continue

            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            crops = []
            for x1, y1, x2, y2 in xyxy.tolist():
                bw, bh = x2 - x1, y2 - y1
                px, py = bw * args.pad, bh * args.pad
                cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
                cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
                if cx2 - cx1 < 3 or cy2 - cy1 < 3:
                    crops.append(np.zeros((csize, csize, 3), np.uint8))
                    continue
                crops.append(cv2.resize(img[cy1:cy2, cx1:cx2], (csize, csize),
                                        interpolation=cv2.INTER_CUBIC))

            batch = torch.from_numpy(
                np.stack(crops)[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0
            with torch.no_grad():
                probs = clf(batch.to(device)).softmax(-1).cpu()

            # Reorder classifier columns into YOLO class-id order, then scale by
            # detector confidence so the score means P(character here) *
            # P(identity). The count-prior decoder consumes exactly that product.
            s2 = torch.zeros(n, len(names))
            for y, c in enumerate(yolo_to_clf):
                s2[:, y] = probs[:, c]
            s2 = s2 * b.conf.cpu().reshape(-1, 1)
            two_scores.append(s2)

    def score(decoder, boxes_list, scores_list) -> dict:
        exact = count_ok = over = under = 0
        edits = chars = 0
        for gt, bx, sc in zip(gts, boxes_list, scores_list):
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

    results: dict[str, dict] = {}
    for tag, sc_list in (("single", single_scores), ("two_stage", two_scores)):
        for c in [float(x) for x in args.std_confs.split(",")]:
            k = f"{tag}_standard_conf{c:.2f}"
            results[k] = score(
                lambda bx, sc, c=c: decode_to_string(bx, sc, names, c, 0.7),
                single_boxes, sc_list)
            print(f"{k:<34} {results[k]}")
        for g in [float(x) for x in args.gammas.split(",")]:
            k = f"{tag}_countprior_gamma{g:.2f}"
            results[k] = score(
                lambda bx, sc, g=g: decode_with_count_prior(
                    bx, sc, names, prior, base_conf=args.base_conf, gamma=g),
                single_boxes, sc_list)
            print(f"{k:<34} {results[k]}")

    best = max(results.items(), key=lambda kv: kv[1]["solve_rate"])
    out = {"detector": args.detector, "classifier": args.classifier,
           "classifier_crop_val_acc": ckpt.get("val_acc"),
           "split": args.split, "n_images": len(images),
           "results": results, "best_config": best[0], "best": best[1]}
    outp = ROOT / "runs" / "eval" / f"two_stage_{args.name}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nBEST: {best[0]} -> {best[1]['solve_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
