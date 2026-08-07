"""Tests for the RepVGG-style backbone (task #4).

The load-bearing property is that fusion is *exact*: the deployed single-kernel
graph must produce the same numbers as the trained multi-branch graph. A backbone
that quietly changes its output when fused would invalidate every accuracy
measurement taken before export.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from efficientvision.models.backbone import RepBackbone, _round_ch
from efficientvision.models.blocks import RepConv, fuse_model


def test_pyramid_strides_and_widths():
    model = RepBackbone().eval()
    p3, p4, p5 = model(torch.randn(2, 3, 320, 320))
    assert p3.shape[-2:] == (40, 40)
    assert p4.shape[-2:] == (20, 20)
    assert p5.shape[-2:] == (10, 10)
    assert (p3.shape[1], p4.shape[1], p5.shape[1]) == model.out_channels


def test_non_square_input():
    """CAPTCHA images are wide, so non-square inputs must survive unpadded."""
    model = RepBackbone().eval()
    p3, p4, p5 = model(torch.randn(1, 3, 96, 320))
    assert p3.shape[-2:] == (12, 40)
    assert p4.shape[-2:] == (6, 20)
    assert p5.shape[-2:] == (3, 10)


def test_fusion_is_numerically_exact():
    torch.manual_seed(0)
    model = RepBackbone(base_ch=16, depths=(1, 2, 2, 1)).eval()
    # Random running stats: with freshly initialised BatchNorms (mean 0, var 1)
    # the fold is near-identity and would pass even if the algebra were wrong.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.normal_()
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.normal_(1.0, 0.2)
            m.bias.data.normal_(0.0, 0.2)

    x = torch.randn(2, 3, 160, 160)
    with torch.no_grad():
        before = model(x)
    fuse_model(model)
    with torch.no_grad():
        after = model(x)

    for b, a in zip(before, after):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-5)


def test_fusion_removes_training_branches():
    model = RepBackbone(base_ch=16, depths=(1, 1, 1, 1)).eval()
    fuse_model(model)
    reps = [m for m in model.modules() if isinstance(m, RepConv)]
    assert reps and all(m.fused for m in reps)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())


def test_fuse_refuses_in_training_mode():
    model = RepBackbone(base_ch=16, depths=(1, 1, 1, 1))
    try:
        fuse_model(model)
    except RuntimeError as exc:
        assert "eval" in str(exc)
    else:
        raise AssertionError("fuse_model() must refuse to run in training mode")


def test_identity_branch_only_where_shape_is_preserved():
    model = RepBackbone(base_ch=16, depths=(3, 1, 1, 1))
    for m in model.modules():
        if isinstance(m, RepConv):
            expected = m.in_ch == m.out_ch and m.stride == 1
            assert (m.branch_id is not None) == expected


def test_width_cap_applies():
    model = RepBackbone(base_ch=32, width_mult=4.0, max_width=128)
    assert max(model.out_channels) <= 128


def test_round_ch_never_drops_more_than_ten_percent():
    for v in range(1, 400):
        assert _round_ch(v) >= 0.9 * v
        assert _round_ch(v) % 8 == 0
