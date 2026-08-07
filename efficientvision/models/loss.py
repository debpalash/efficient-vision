"""Losses and label assignment for the axis-aligned detector (task #8).

Assignment: TaskAlignedAssigner
------------------------------
Which grid cells are positives for which ground-truth box? Fixed IoU thresholds
assign by localisation quality alone, which lets a cell that localises well but
classifies the object wrongly become a high-quality positive. Task alignment
(TOOD, Feng et al. 2021) scores each candidate by

    t = score^alpha * iou^beta

so a cell must be good at *both* jobs to be selected, and picks the top-k per
ground-truth box. On this dataset that matters more than usual: adjacent CAPTCHA
characters overlap heavily in their bounding boxes, so IoU alone would happily
assign a cell to the neighbouring glyph.

Losses
------
- **Classification**: BCE weighted by the normalised alignment metric, so the
  target for a positive is its alignment score rather than a hard 1. This keeps
  classification confidence calibrated against localisation quality, which is what
  makes confidence thresholding behave predictably at decode time -- and solve
  rate is extremely sensitive to the confidence threshold.
- **Box**: CIoU, which adds centre-distance and aspect-ratio terms to IoU so the
  gradient does not vanish when boxes do not overlap.
- **DFL**: cross-entropy against the two bins bracketing the true distance,
  weighted by their distance to it. This is what teaches the distribution to
  concentrate near the true edge instead of drifting.

All three are normalised by the summed target score rather than the positive
count, so images with few characters are not down-weighted relative to crowded
ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_iou(a: torch.Tensor, b: torch.Tensor, ciou: bool = False, eps: float = 1e-7):
    """IoU (or CIoU) between two broadcastable sets of xyxy boxes."""
    ax1, ay1, ax2, ay2 = a.unbind(-1)
    bx1, by1, bx2, by2 = b.unbind(-1)

    inter = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp_(0) * (
        torch.min(ay2, by2) - torch.max(ay1, by1)
    ).clamp_(0)
    aw, ah = (ax2 - ax1).clamp_(0), (ay2 - ay1).clamp_(0)
    bw, bh = (bx2 - bx1).clamp_(0), (by2 - by1).clamp_(0)
    union = aw * ah + bw * bh - inter + eps
    iou = inter / union
    if not ciou:
        return iou

    cw = torch.max(ax2, bx2) - torch.min(ax1, bx1)
    ch = torch.max(ay2, by2) - torch.min(ay1, by1)
    c2 = cw**2 + ch**2 + eps
    rho2 = ((bx1 + bx2 - ax1 - ax2) ** 2 + (by1 + by2 - ay1 - ay2) ** 2) / 4
    v = (4 / torch.pi**2) * (torch.atan(bw / (bh + eps)) - torch.atan(aw / (ah + eps))) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


class TaskAlignedAssigner(nn.Module):
    """Assign anchors to ground-truth boxes by joint classification/localisation quality.

    Args:
        topk: candidates kept per ground-truth box.
        alpha, beta: exponents on score and IoU in the alignment metric.
    """

    def __init__(self, topk: int = 10, alpha: float = 1.0, beta: float = 6.0) -> None:
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def forward(
        self,
        pred_scores: torch.Tensor,   # (B, A, nc) sigmoid probabilities
        pred_boxes: torch.Tensor,    # (B, A, 4) xyxy pixels
        anchors: torch.Tensor,       # (A, 2) cell centres
        gt_labels: torch.Tensor,     # (B, M) int64
        gt_boxes: torch.Tensor,      # (B, M, 4) xyxy pixels
        gt_mask: torch.Tensor,       # (B, M) bool, False for padding
    ):
        """Returns `(target_labels, target_boxes, target_scores, fg_mask)`."""
        b, a, nc = pred_scores.shape
        m = gt_boxes.shape[1]
        if m == 0 or not gt_mask.any():
            return (
                torch.zeros(b, a, dtype=torch.long, device=pred_scores.device),
                torch.zeros(b, a, 4, device=pred_scores.device),
                torch.zeros(b, a, nc, device=pred_scores.device),
                torch.zeros(b, a, dtype=torch.bool, device=pred_scores.device),
            )

        # A candidate must have its cell centre inside the ground-truth box.
        # Without this, top-k on the alignment metric alone can select cells far
        # outside the object that happen to score highly early in training.
        pts = anchors.reshape(1, 1, a, 2)
        lt = pts - gt_boxes[:, :, None, :2]
        rb = gt_boxes[:, :, None, 2:] - pts
        in_gt = torch.cat((lt, rb), -1).amin(-1) > 0          # (B, M, A)

        ious = bbox_iou(pred_boxes[:, None, :, :], gt_boxes[:, :, None, :]).clamp_(0)
        # clamp (not clamp_): padded slots may carry -1, and gather needs a valid
        # index, but mutating the caller's label tensor would be a silent side
        # effect on data the training loop still owns.
        safe_labels = gt_labels.clamp(min=0)
        cls_scores = pred_scores.gather(
            2, safe_labels.unsqueeze(-1).expand(b, m, a).transpose(1, 2)
        ).transpose(1, 2)                                      # (B, M, A)

        metric = cls_scores.pow(self.alpha) * ious.pow(self.beta)
        metric = metric * in_gt * gt_mask.unsqueeze(-1)

        topk = min(self.topk, a)
        _, idx = metric.topk(topk, dim=-1)
        cand = torch.zeros_like(metric, dtype=torch.bool).scatter_(-1, idx, True)
        cand = cand & (metric > 0)

        # One anchor may be a candidate for several overlapping glyphs. Adjacent
        # CAPTCHA characters overlap constantly, so this is the common case, not a
        # rare tie: give the anchor to whichever box it aligns with best.
        overlap = cand.sum(1)                                  # (B, A)
        if overlap.max() > 1:
            multi = (overlap > 1).unsqueeze(1).expand_as(cand)
            best = metric.argmax(1)                            # (B, A)
            keep = F.one_hot(best, m).permute(0, 2, 1).bool()
            cand = torch.where(multi, keep & cand, cand)

        fg_mask = cand.any(1)                                  # (B, A)
        assigned = cand.float().argmax(1)                      # (B, A) index into M

        batch_idx = torch.arange(b, device=pred_scores.device).unsqueeze(-1)
        target_labels = safe_labels[batch_idx, assigned]
        target_boxes = gt_boxes[batch_idx, assigned]

        # Normalise the alignment metric per ground-truth box so the best-matched
        # anchor gets a soft target near its IoU rather than an arbitrary scale.
        align = metric * cand
        pos_max = align.amax(-1, keepdim=True)
        iou_max = (ious * cand).amax(-1, keepdim=True)
        norm = (align / (pos_max + 1e-9) * iou_max).amax(1)    # (B, A)

        target_scores = F.one_hot(target_labels, nc).float() * norm.unsqueeze(-1)
        target_scores = target_scores * fg_mask.unsqueeze(-1)
        return target_labels, target_boxes, target_scores, fg_mask


def dfl_loss(
    pred_dist: torch.Tensor,   # (N, 4, reg_max+1) raw logits for positives
    target: torch.Tensor,      # (N, 4) true distances in stride units
    reg_max: int,
) -> torch.Tensor:
    """Cross-entropy against the two bins bracketing each true distance.

    A distance of 3.25 puts 0.75 of the mass on bin 3 and 0.25 on bin 4. Targets
    are clamped just under `reg_max` so the upper bin index stays in range.
    """
    # clamp, not clamp_: `target` is derived from the caller's assigned boxes.
    target = target.clamp(0, reg_max - 0.01)
    lo = target.long()
    hi = lo + 1
    w_hi = target - lo.float()
    w_lo = 1.0 - w_hi
    logp = F.log_softmax(pred_dist, dim=-1)
    loss_lo = -logp.gather(-1, lo.unsqueeze(-1)).squeeze(-1) * w_lo
    loss_hi = -logp.gather(-1, hi.unsqueeze(-1)).squeeze(-1) * w_hi
    return (loss_lo + loss_hi).mean(-1)


class DetectionLoss(nn.Module):
    """Combined classification + CIoU + DFL loss over the assigned targets."""

    def __init__(
        self,
        nc: int,
        reg_max: int = 8,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
        assigner: TaskAlignedAssigner | None = None,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.assigner = assigner or TaskAlignedAssigner()

    def forward(
        self,
        box_dist: torch.Tensor,   # (B, 4*(reg_max+1), A) raw
        cls_logits: torch.Tensor, # (B, nc, A) raw
        pred_boxes: torch.Tensor, # (B, A, 4) decoded xyxy pixels
        anchors: torch.Tensor,    # (A, 2)
        strides: torch.Tensor,    # (A,)
        gt_labels: torch.Tensor,  # (B, M)
        gt_boxes: torch.Tensor,   # (B, M, 4)
        gt_mask: torch.Tensor,    # (B, M) bool
    ) -> tuple[torch.Tensor, dict[str, float]]:
        b, _, a = cls_logits.shape
        scores = cls_logits.permute(0, 2, 1)                   # (B, A, nc)

        _, t_boxes, t_scores, fg = self.assigner(
            scores.detach().sigmoid(), pred_boxes.detach(), anchors,
            gt_labels, gt_boxes, gt_mask,
        )

        # Normalising by summed target score rather than positive count keeps
        # sparse images (3 characters) on the same footing as crowded ones (7).
        denom = t_scores.sum().clamp_(min=1.0)

        loss_cls = F.binary_cross_entropy_with_logits(
            scores, t_scores, reduction="sum"
        ) / denom

        if fg.any():
            weight = t_scores.sum(-1)[fg]
            iou = bbox_iou(pred_boxes[fg], t_boxes[fg], ciou=True)
            loss_box = ((1.0 - iou) * weight).sum() / denom

            # Convert the assigned box back to l,t,r,b distances in stride units,
            # which is the space the DFL bins live in.
            s = strides.reshape(1, -1, 1).expand(b, a, 1)[fg]
            pts = anchors.unsqueeze(0).expand(b, a, 2)[fg]
            lt = (pts - t_boxes[fg][:, :2]) / s
            rb = (t_boxes[fg][:, 2:] - pts) / s
            t_dist = torch.cat((lt, rb), -1)

            dist = box_dist.permute(0, 2, 1).reshape(b, a, 4, self.reg_max + 1)[fg]
            loss_dfl = (dfl_loss(dist, t_dist, self.reg_max) * weight).sum() / denom
        else:
            loss_box = pred_boxes.sum() * 0.0
            loss_dfl = box_dist.sum() * 0.0

        total = (
            self.cls_gain * loss_cls
            + self.box_gain * loss_box
            + self.dfl_gain * loss_dfl
        )
        return total, {
            "cls": float(loss_cls.detach()),
            "box": float(loss_box.detach()),
            "dfl": float(loss_dfl.detach()),
            "n_pos": int(fg.sum()),
        }
