"""Unified evaluation for the student detector (task #15).

Reports the **contract metric** -- per-image string-solve rate -- alongside mAP50
and latency, because this project has now been misled by mAP three separate times
on this dataset (see `docs/feasibility-gate.md`):

  * 160px -> 320px: mAP50 +0.030, solve rate +0.010
  * mosaic on -> off: mAP50 **-0.018**, solve rate **+0.115**

mAP is a per-box average; solve rate requires every character in an image to be
detected, classified, and ordered correctly. They can and do move in opposite
directions. Any harness that reported only mAP would have accepted the mosaic
setting that was costing 11 points of the thing we actually ship.

Accuracy and latency are always reported together (CLAUDE.md): a change is never
accepted on one axis while silently regressing the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


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


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """Class-agnostic NMS. Returns kept indices, highest score first.

    Class-agnostic rather than per-class on purpose: two boxes on the same glyph
    with different predicted classes are duplicates of one character, and keeping
    both inserts a spurious character that breaks the whole string. Per-class NMS
    would keep them.
    """
    from torchvision.ops import nms as tv_nms

    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    return tv_nms(boxes, scores, iou_thresh)


def decode_to_string(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    names: list[str],
    conf: float,
    iou: float,
) -> str:
    """Detections -> the character string, ordered left to right by box centre."""
    best_score, best_cls = scores.max(-1)
    keep = best_score >= conf
    if not keep.any():
        return ""
    b, s, c = boxes[keep], best_score[keep], best_cls[keep]
    idx = nms(b, s, iou)
    b, c = b[idx], c[idx]
    order = ((b[:, 0] + b[:, 2]) / 2).argsort()
    return "".join(names[int(k)] for k in c[order])


@dataclass
class EvalResult:
    conf: float
    iou: float
    solve_rate: float
    char_accuracy: float
    count_match: float
    too_many_boxes: float
    too_few_boxes: float
    n_images: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@torch.no_grad()
def evaluate_solve_rate(
    model,
    dataloader,
    names: list[str],
    conf: float = 0.25,
    iou: float = 0.7,
    device: str = "cpu",
) -> EvalResult:
    """Solve rate over a dataloader yielding `(imgs, gt_boxes, gt_labels, gt_mask)`.

    Ground-truth strings are built with the same left-to-right ordering rule as
    predictions, so an ordering bug would cancel out and go unnoticed -- which is
    why `tests/test_eval_harness.py` asserts the ordering rule directly rather
    than only end-to-end.
    """
    model = model.to(device).eval()
    exact = count_ok = over = under = 0
    edits = gt_chars = n = 0

    for imgs, gt_boxes, gt_labels, gt_mask in dataloader:
        boxes, scores = model.predict(imgs.to(device))
        for i in range(imgs.shape[0]):
            m = gt_mask[i]
            gb, gl = gt_boxes[i][m], gt_labels[i][m]
            order = ((gb[:, 0] + gb[:, 2]) / 2).argsort()
            gt = "".join(names[int(k)] for k in gl[order])

            pred = decode_to_string(boxes[i].cpu(), scores[i].cpu(), names, conf, iou)

            exact += pred == gt
            count_ok += len(pred) == len(gt)
            over += len(pred) > len(gt)
            under += len(pred) < len(gt)
            edits += levenshtein(pred, gt)
            gt_chars += len(gt)
            n += 1

    n = max(n, 1)
    return EvalResult(
        conf=conf, iou=iou,
        solve_rate=exact / n,
        char_accuracy=1 - edits / max(gt_chars, 1),
        count_match=count_ok / n,
        too_many_boxes=over / n,
        too_few_boxes=under / n,
        n_images=n,
    )


def sweep_confidence(
    model, dataloader, names: list[str],
    confs=(0.1, 0.2, 0.25, 0.3, 0.4, 0.5), iou: float = 0.7, device: str = "cpu",
) -> tuple[EvalResult, list[EvalResult]]:
    """Solve rate is sharply peaked in the confidence threshold -- one spurious box
    breaks an entire string, while one missing box breaks it equally. Reporting a
    single default threshold would understate a model that is simply mis-tuned, so
    the operating point is always swept rather than assumed.

    Returns `(best, all_results)`.
    """
    results = [
        evaluate_solve_rate(model, dataloader, names, conf=c, iou=iou, device=device)
        for c in confs
    ]
    return max(results, key=lambda r: r.solve_rate), results
