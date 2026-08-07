"""Tests for assignment and losses (task #8).

The assigner is the component most likely to be subtly wrong while still training
to a plausible-looking loss curve, so it gets the most direct assertions: perfect
predictions must produce near-zero loss, padded ground truth must never be
assigned, and an anchor contested by two overlapping glyphs must go to exactly one
of them. That last case is the norm on this dataset, not an edge case.
"""

from __future__ import annotations

import torch

from efficientvision.models.head import DFL, decode_boxes, make_anchor_points
from efficientvision.models.loss import (
    DetectionLoss,
    TaskAlignedAssigner,
    bbox_iou,
    dfl_loss,
)


def test_iou_of_identical_boxes_is_one():
    box = torch.tensor([[10.0, 10.0, 20.0, 30.0]])
    torch.testing.assert_close(bbox_iou(box, box), torch.tensor([1.0]))
    torch.testing.assert_close(bbox_iou(box, box, ciou=True), torch.tensor([1.0]))


def test_iou_of_disjoint_boxes_is_zero_but_ciou_is_negative():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
    torch.testing.assert_close(bbox_iou(a, b), torch.tensor([0.0]), atol=1e-6, rtol=0)
    # CIoU must stay informative when boxes do not overlap -- that is its point.
    assert bbox_iou(a, b, ciou=True).item() < -0.5


def test_dfl_loss_is_zero_for_a_confident_correct_prediction():
    reg_max = 8
    target = torch.tensor([[3.0, 3.0, 3.0, 3.0]])
    logits = torch.full((1, 4, reg_max + 1), -50.0)
    logits[:, :, 3] = 50.0
    assert dfl_loss(logits, target, reg_max).item() < 1e-4


def test_dfl_loss_splits_between_bracketing_bins():
    """A target of 3.5 should be best served by mass split evenly on bins 3 and 4."""
    reg_max = 8
    target = torch.tensor([[3.5, 3.5, 3.5, 3.5]])
    split = torch.full((1, 4, reg_max + 1), -50.0)
    split[:, :, 3] = split[:, :, 4] = 50.0
    exact = torch.full((1, 4, reg_max + 1), -50.0)
    exact[:, :, 3] = 50.0
    assert dfl_loss(split, target, reg_max).item() < dfl_loss(exact, target, reg_max).item()


def _toy_batch():
    """One 64x64 image, two well-separated boxes, a single P3-like level."""
    feat = torch.zeros(1, 1, 8, 8)
    anchors = make_anchor_points((feat,), (8,))
    strides = torch.full((anchors.shape[0],), 8.0)
    gt_boxes = torch.tensor([[[8.0, 8.0, 24.0, 24.0], [40.0, 40.0, 56.0, 56.0]]])
    gt_labels = torch.tensor([[0, 1]])
    gt_mask = torch.tensor([[True, True]])
    return anchors, strides, gt_boxes, gt_labels, gt_mask


def test_assigner_selects_anchors_inside_the_boxes():
    anchors, _, gt_boxes, gt_labels, gt_mask = _toy_batch()
    a = anchors.shape[0]
    # Predictions equal to ground truth for every anchor, high confidence.
    pred_boxes = gt_boxes[0, 0].reshape(1, 1, 4).expand(1, a, 4).clone()
    pred_scores = torch.full((1, a, 2), 0.9)

    assigner = TaskAlignedAssigner(topk=4)
    _, _, _, fg = assigner(pred_scores, pred_boxes, anchors, gt_labels, gt_boxes, gt_mask)

    chosen = anchors[fg[0]]
    inside = (
        (chosen[:, 0] > 8) & (chosen[:, 0] < 24) & (chosen[:, 1] > 8) & (chosen[:, 1] < 24)
    )
    assert fg.any() and inside.all()


def test_padded_ground_truth_is_never_assigned():
    anchors, _, gt_boxes, gt_labels, _ = _toy_batch()
    a = anchors.shape[0]
    gt_mask = torch.tensor([[True, False]])          # second box is padding
    pred_boxes = gt_boxes[0, 1].reshape(1, 1, 4).expand(1, a, 4).clone()
    pred_scores = torch.full((1, a, 2), 0.9)

    assigner = TaskAlignedAssigner(topk=4)
    labels, _, _, fg = assigner(
        pred_scores, pred_boxes, anchors, gt_labels, gt_boxes, gt_mask
    )
    # Predictions match the *padded* box exactly, so a broken mask would assign
    # label 1 to those anchors. Nothing may carry label 1.
    assert not (labels[fg] == 1).any()


def test_contested_anchor_goes_to_exactly_one_box():
    """Overlapping neighbours are the norm for adjacent CAPTCHA glyphs."""
    feat = torch.zeros(1, 1, 8, 8)
    anchors = make_anchor_points((feat,), (8,))
    a = anchors.shape[0]
    gt_boxes = torch.tensor([[[8.0, 8.0, 40.0, 40.0], [16.0, 8.0, 48.0, 40.0]]])
    gt_labels = torch.tensor([[0, 1]])
    gt_mask = torch.tensor([[True, True]])
    pred_boxes = gt_boxes[0, 0].reshape(1, 1, 4).expand(1, a, 4).clone()
    pred_scores = torch.full((1, a, 2), 0.9)

    assigner = TaskAlignedAssigner(topk=8)
    _, t_boxes, _, fg = assigner(
        pred_scores, pred_boxes, anchors, gt_labels, gt_boxes, gt_mask
    )
    assert fg.any()
    # Each assigned anchor must carry one of the two boxes verbatim, never a blend.
    for box in t_boxes[fg]:
        assert torch.allclose(box, gt_boxes[0, 0]) or torch.allclose(box, gt_boxes[0, 1])


def test_empty_ground_truth_produces_finite_loss_and_no_positives():
    anchors, strides, _, _, _ = _toy_batch()
    a = anchors.shape[0]
    reg_max, nc = 8, 4
    box_dist = torch.zeros(1, 4 * (reg_max + 1), a, requires_grad=True)
    cls_logits = torch.zeros(1, nc, a, requires_grad=True)
    pred_boxes = decode_boxes(box_dist, anchors, strides, DFL(reg_max))

    loss_fn = DetectionLoss(nc=nc, reg_max=reg_max)
    total, parts = loss_fn(
        box_dist, cls_logits, pred_boxes, anchors, strides,
        torch.zeros(1, 0, dtype=torch.long), torch.zeros(1, 0, 4),
        torch.zeros(1, 0, dtype=torch.bool),
    )
    assert parts["n_pos"] == 0
    assert torch.isfinite(total)
    total.backward()          # must not raise: an all-background image is valid


def test_loss_is_lower_for_better_predictions():
    anchors, strides, gt_boxes, gt_labels, gt_mask = _toy_batch()
    a = anchors.shape[0]
    reg_max, nc = 8, 4
    dfl = DFL(reg_max)
    loss_fn = DetectionLoss(nc=nc, reg_max=reg_max)

    def run(seed: int, scale: float):
        torch.manual_seed(seed)
        box_dist = torch.randn(1, 4 * (reg_max + 1), a) * scale
        cls_logits = torch.randn(1, nc, a) * scale
        boxes = decode_boxes(box_dist, anchors, strides, dfl)
        total, _ = loss_fn(
            box_dist, cls_logits, boxes, anchors, strides, gt_labels, gt_boxes, gt_mask
        )
        return float(total)

    # A near-uniform distribution decodes to a sensible mid-range box; a wildly
    # scaled one does not. The badly scaled model must not score better.
    assert run(0, 0.1) < run(0, 8.0)


def test_loss_backward_populates_gradients():
    anchors, strides, gt_boxes, gt_labels, gt_mask = _toy_batch()
    a = anchors.shape[0]
    reg_max, nc = 8, 4
    box_dist = torch.randn(1, 4 * (reg_max + 1), a, requires_grad=True)
    cls_logits = torch.randn(1, nc, a, requires_grad=True)
    pred_boxes = decode_boxes(box_dist, anchors, strides, DFL(reg_max))

    total, parts = DetectionLoss(nc=nc, reg_max=reg_max)(
        box_dist, cls_logits, pred_boxes, anchors, strides, gt_labels, gt_boxes, gt_mask
    )
    total.backward()
    assert parts["n_pos"] > 0
    assert box_dist.grad is not None and box_dist.grad.abs().sum() > 0
    assert cls_logits.grad is not None and cls_logits.grad.abs().sum() > 0
