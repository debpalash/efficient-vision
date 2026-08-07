"""Tests for the PAN neck (task #5).

Beyond shape plumbing, two things matter: fusion stays exact end-to-end when the
neck is stacked on the backbone, and odd spatial sizes do not crash. The second is
not hypothetical -- CAPTCHA inputs are wide strips, not padded squares, so a
stride-8 map of odd height is the normal case rather than an edge case.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from efficientvision.models.backbone import RepBackbone
from efficientvision.models.blocks import fuse_model
from efficientvision.models.neck import PANNeck


def test_output_shapes_match_input_strides():
    backbone = RepBackbone(base_ch=16, depths=(1, 1, 1, 1)).eval()
    neck = PANNeck(backbone.out_channels, neck_ch=32).eval()
    feats = backbone(torch.randn(2, 3, 320, 320))
    n3, n4, n5 = neck(feats)
    assert n3.shape[-2:] == feats[0].shape[-2:] == (40, 40)
    assert n4.shape[-2:] == feats[1].shape[-2:] == (20, 20)
    assert n5.shape[-2:] == feats[2].shape[-2:] == (10, 10)
    assert n3.shape[1] == n4.shape[1] == n5.shape[1] == 32


@pytest.mark.parametrize("hw", [(96, 320), (100, 300), (60, 230), (54, 280)])
def test_odd_and_non_square_sizes_survive(hw):
    """Real CAPTCHA aspect ratios, including ones that produce odd feature maps."""
    h, w = hw
    backbone = RepBackbone(base_ch=16, depths=(1, 1, 1, 1)).eval()
    neck = PANNeck(backbone.out_channels, neck_ch=32).eval()
    feats = backbone(torch.randn(1, 3, h, w))
    outs = neck(feats)
    for out, feat in zip(outs, feats):
        assert out.shape[-2:] == feat.shape[-2:]


def test_fusion_exact_through_backbone_and_neck():
    torch.manual_seed(0)
    backbone = RepBackbone(base_ch=16, depths=(1, 2, 2, 1)).eval()
    neck = PANNeck(backbone.out_channels, neck_ch=32).eval()
    model = nn.ModuleList([backbone, neck]).eval()

    # Perturb running stats so the BatchNorm fold is a real transformation and not
    # a near-identity that would pass even with wrong algebra.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.normal_()
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.normal_(1.0, 0.2)
            m.bias.data.normal_(0.0, 0.2)

    x = torch.randn(2, 3, 160, 160)
    with torch.no_grad():
        before = neck(backbone(x))
    fuse_model(model)
    with torch.no_grad():
        after = neck(backbone(x))

    for b, a in zip(before, after):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-5)


def test_rejects_wrong_level_count():
    with pytest.raises(ValueError):
        PANNeck((64, 128), neck_ch=32)
