"""Guard against silent in-place mutation of caller-owned tensors (task #8).

Two earlier versions of `loss.py` used `clamp_` on `gt_labels` and on the DFL
target. Both trained fine and every accuracy test still passed -- the damage is
invisible until a caller reuses the tensor it handed in. These tests pin the
contract: assignment and loss are pure with respect to their inputs.
"""

from __future__ import annotations

import torch

from efficientvision.models.head import DFL, decode_boxes, make_anchor_points
from efficientvision.models.loss import DetectionLoss, TaskAlignedAssigner, dfl_loss


def test_assigner_does_not_mutate_ground_truth():
    feat = torch.zeros(1, 1, 8, 8)
    anchors = make_anchor_points((feat,), (8,))
    a = anchors.shape[0]

    # -1 is the natural padding sentinel for an unused ground-truth slot.
    gt_labels = torch.tensor([[0, -1]])
    gt_boxes = torch.tensor([[[8.0, 8.0, 24.0, 24.0], [0.0, 0.0, 0.0, 0.0]]])
    gt_mask = torch.tensor([[True, False]])
    before = gt_labels.clone()

    TaskAlignedAssigner(topk=4)(
        torch.full((1, a, 2), 0.9),
        gt_boxes[0, 0].reshape(1, 1, 4).expand(1, a, 4).clone(),
        anchors, gt_labels, gt_boxes, gt_mask,
    )
    torch.testing.assert_close(gt_labels, before)


def test_dfl_loss_does_not_mutate_its_target():
    target = torch.tensor([[3.0, 20.0, 3.0, 3.0]])   # 20 exceeds reg_max
    before = target.clone()
    dfl_loss(torch.zeros(1, 4, 9), target, reg_max=8)
    torch.testing.assert_close(target, before)


def test_full_loss_does_not_mutate_inputs():
    feat = torch.zeros(1, 1, 8, 8)
    anchors = make_anchor_points((feat,), (8,))
    a = anchors.shape[0]
    reg_max, nc = 8, 4

    gt_labels = torch.tensor([[0, 1]])
    gt_boxes = torch.tensor([[[8.0, 8.0, 24.0, 24.0], [40.0, 40.0, 56.0, 56.0]]])
    gt_mask = torch.tensor([[True, True]])
    strides = torch.full((a,), 8.0)
    snapshots = [t.clone() for t in (gt_labels, gt_boxes, anchors, strides)]

    box_dist = torch.randn(1, 4 * (reg_max + 1), a)
    cls_logits = torch.randn(1, nc, a)
    pred_boxes = decode_boxes(box_dist, anchors, strides, DFL(reg_max))

    DetectionLoss(nc=nc, reg_max=reg_max)(
        box_dist, cls_logits, pred_boxes, anchors, strides, gt_labels, gt_boxes, gt_mask
    )
    for tensor, snap in zip((gt_labels, gt_boxes, anchors, strides), snapshots):
        torch.testing.assert_close(tensor, snap)
