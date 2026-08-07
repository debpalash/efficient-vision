"""Tests for the detection head (tasks #6/#7).

The properties worth asserting are the ones that fail silently: DFL decoding must
actually compute an expectation (not just produce plausibly-shaped numbers),
anchor centres must be half-cell offset, and decode must invert a known box.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from efficientvision.models.backbone import RepBackbone
from efficientvision.models.blocks import fuse_model
from efficientvision.models.head import (
    DFL,
    DetectHead,
    decode_boxes,
    make_anchor_points,
)
from efficientvision.models.neck import PANNeck


def test_dfl_computes_expectation_over_bins():
    """A one-hot distribution on bin k must decode to exactly k."""
    dfl = DFL(reg_max=8)
    for k in range(9):
        logits = torch.full((1, 4 * 9, 1), -50.0)
        for side in range(4):
            logits[0, side * 9 + k, 0] = 50.0
        out = dfl(logits)
        torch.testing.assert_close(out, torch.full((1, 4, 1), float(k)), atol=1e-3, rtol=0)


def test_dfl_is_not_trainable():
    dfl = DFL(reg_max=8)
    assert not dfl.proj.weight.requires_grad


def test_anchor_points_are_half_cell_offset():
    feat = torch.zeros(1, 1, 2, 3)
    pts = make_anchor_points((feat,), (8,))
    expected = torch.tensor(
        [[4.0, 4.0], [12.0, 4.0], [20.0, 4.0],
         [4.0, 12.0], [12.0, 12.0], [20.0, 12.0]]
    )
    torch.testing.assert_close(pts, expected)


def test_decode_recovers_a_known_box():
    """Craft distances of (l,t,r,b)=(2,2,3,3) stride units at one anchor."""
    dfl = DFL(reg_max=8)
    dists = [2, 2, 3, 3]
    logits = torch.full((1, 4 * 9, 1), -50.0)
    for side, k in enumerate(dists):
        logits[0, side * 9 + k, 0] = 50.0

    anchors = torch.tensor([[40.0, 24.0]])
    strides = torch.tensor([8.0])
    box = decode_boxes(logits, anchors, strides, dfl)[0, 0]
    torch.testing.assert_close(
        box, torch.tensor([40 - 16.0, 24 - 16.0, 40 + 24.0, 24 + 24.0]), atol=1e-2, rtol=0
    )


def test_head_output_shapes_and_anchor_count():
    backbone = RepBackbone(base_ch=16, depths=(1, 1, 1, 1)).eval()
    neck = PANNeck(backbone.out_channels, neck_ch=32).eval()
    head = DetectHead(neck.out_channels, nc=64, reg_max=8, tower_ch=32).eval()

    feats = neck(backbone(torch.randn(2, 3, 320, 320)))
    box, cls, strides = head(feats)

    expected_anchors = 40 * 40 + 20 * 20 + 10 * 10
    assert box.shape == (2, 4 * 9, expected_anchors)
    assert cls.shape == (2, 64, expected_anchors)
    assert strides.shape == (expected_anchors,)
    # Anchors are ordered P3, P4, P5.
    assert strides[0] == 8 and strides[-1] == 32


def test_class_bias_starts_at_background_prior():
    head = DetectHead((32, 32, 32), nc=64)
    expected = -math.log((1 - 0.01) / 0.01)
    for layer in head.cls_pred:
        torch.testing.assert_close(
            layer.bias, torch.full_like(layer.bias, expected), atol=1e-5, rtol=0
        )


def test_full_stack_fusion_is_exact():
    torch.manual_seed(0)
    backbone = RepBackbone(base_ch=16, depths=(1, 1, 1, 1)).eval()
    neck = PANNeck(backbone.out_channels, neck_ch=32).eval()
    head = DetectHead(neck.out_channels, nc=16, tower_ch=32).eval()
    model = nn.ModuleList([backbone, neck, head]).eval()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.normal_()
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.normal_(1.0, 0.2)
            m.bias.data.normal_(0.0, 0.2)

    x = torch.randn(1, 3, 160, 160)
    with torch.no_grad():
        before = head(neck(backbone(x)))
    fuse_model(model)
    with torch.no_grad():
        after = head(neck(backbone(x)))

    for b, a in zip(before, after):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-5)


def test_rejects_stride_level_mismatch():
    with pytest.raises(ValueError):
        DetectHead((32, 32), nc=8, strides=(8, 16, 32))
