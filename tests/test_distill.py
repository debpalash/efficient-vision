"""Tests for distillation losses (task #11).

Distillation is easy to get wrong in ways that still train: a KL with the
arguments swapped, a temperature that silently rescales the loss weight, or a
confidence mask that lets background anchors dominate. Each gets a direct test.
"""

from __future__ import annotations

import pytest
import torch

from efficientvision.train.distill import DistillLoss, _confidence_mask, kl_logits


def test_kl_is_zero_for_identical_logits():
    x = torch.randn(8, 16)
    assert kl_logits(x, x, temperature=2.0).item() == pytest.approx(0.0, abs=1e-6)


def test_kl_grows_as_distributions_diverge():
    t = torch.zeros(4, 8)
    t[:, 0] = 10.0
    near = torch.zeros(4, 8)
    near[:, 0] = 5.0
    far = torch.zeros(4, 8)
    far[:, 7] = 10.0
    assert kl_logits(near, t, 1.0).item() < kl_logits(far, t, 1.0).item()


def test_temperature_squared_scaling_keeps_gradients_comparable():
    """Without the T^2 factor, raising T would quietly shrink this term's weight."""
    torch.manual_seed(0)
    s = torch.randn(32, 16, requires_grad=True)
    t = torch.randn(32, 16)

    grads = []
    for temp in (1.0, 4.0):
        s.grad = None
        kl_logits(s, t, temp).backward()
        grads.append(s.grad.abs().mean().item())

    # Same order of magnitude across a 4x temperature change.
    assert 0.2 < grads[1] / grads[0] < 5.0


def test_confidence_mask_selects_only_confident_anchors():
    scores = torch.tensor([[[0.9, 0.05], [0.1, 0.1], [0.3, 0.2]]])
    mask = _confidence_mask(scores, threshold=0.25)
    assert mask.tolist() == [[True, False, True]]


def test_identical_teacher_and_student_gives_zero_loss():
    b, nc, a, reg_max = 2, 8, 20, 8
    cls = torch.randn(b, nc, a) * 3.0
    dist = torch.randn(b, 4 * (reg_max + 1), a)

    total, parts = DistillLoss()(cls, cls, dist, dist, reg_max)
    assert total.item() == pytest.approx(0.0, abs=1e-5)
    assert parts["kd_anchors"] > 0, "test needs some anchors above threshold"


def test_no_confident_anchors_yields_finite_zero_loss_and_backprops():
    b, nc, a, reg_max = 1, 4, 10, 8
    # Strongly negative logits -> every sigmoid score far below threshold.
    cls_s = torch.full((b, nc, a), -20.0, requires_grad=True)
    cls_t = torch.full((b, nc, a), -20.0)
    dist_s = torch.zeros(b, 4 * (reg_max + 1), a, requires_grad=True)
    dist_t = torch.zeros(b, 4 * (reg_max + 1), a)

    total, parts = DistillLoss(conf_threshold=0.25)(
        cls_s, cls_t, dist_s, dist_t, reg_max
    )
    assert parts["kd_anchors"] == 0
    assert torch.isfinite(total)
    total.backward()          # an all-background batch is valid, not an error


def test_feature_distillation_penalises_mismatch():
    b, nc, a, reg_max = 1, 4, 10, 8
    cls = torch.randn(b, nc, a) * 3.0
    dist = torch.randn(b, 4 * (reg_max + 1), a)
    feat = torch.randn(b, 8, 4, 4)

    loss_fn = DistillLoss()
    same, _ = loss_fn(cls, cls, dist, dist, reg_max, (feat,), (feat,))
    diff, parts = loss_fn(cls, cls, dist, dist, reg_max, (feat,), (feat + 1.0,))

    assert diff.item() > same.item()
    assert parts["kd_feat"] == pytest.approx(1.0, rel=1e-3)


def test_feature_shape_mismatch_is_an_error_not_a_broadcast():
    b, nc, a, reg_max = 1, 4, 10, 8
    cls = torch.randn(b, nc, a) * 3.0
    dist = torch.randn(b, 4 * (reg_max + 1), a)
    with pytest.raises(ValueError, match="shape mismatch"):
        DistillLoss()(
            cls, cls, dist, dist, reg_max,
            (torch.randn(b, 8, 4, 4),), (torch.randn(b, 16, 4, 4),),
        )


def test_gradients_reach_the_student_only():
    b, nc, a, reg_max = 1, 4, 10, 8
    cls_s = torch.randn(b, nc, a, requires_grad=True)
    cls_t = (torch.randn(b, nc, a) * 3.0).requires_grad_(True)
    dist_s = torch.randn(b, 4 * (reg_max + 1), a, requires_grad=True)
    dist_t = torch.randn(b, 4 * (reg_max + 1), a)

    total, _ = DistillLoss()(cls_s, cls_t.detach(), dist_s, dist_t, reg_max)
    total.backward()
    assert cls_s.grad is not None and cls_s.grad.abs().sum() > 0
    assert dist_s.grad is not None and dist_s.grad.abs().sum() > 0
    assert cls_t.grad is None
