"""Tests for INT8 fake quantization (task #13).

Quantization bugs are unusually quiet: a wrong scale still produces finite numbers
and a model that trains, just to lower accuracy. So the tests assert the maths
directly -- that quantized values land on the grid, that the straight-through
estimator actually passes gradients, and that per-channel scaling does what it is
there for.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from efficientvision.models.blocks import fuse_model
from efficientvision.models.detector import EfficientDetector
from efficientvision.quant.qat import (
    FakeQuantize,
    QuantConv2d,
    calibrate,
    prepare_qat,
    quantize_dequantize,
    set_fake_quant,
    set_observing,
)


def test_quantize_dequantize_lands_on_the_grid():
    x = torch.linspace(-1, 1, 257)
    scale = torch.tensor(2.0 / 255)
    zp = torch.tensor(0.0)
    out = quantize_dequantize(x, scale, zp, -128, 127)
    # Every output must be an integer multiple of the scale.
    steps = out / scale
    torch.testing.assert_close(steps, steps.round(), atol=1e-4, rtol=0)


def test_quantization_error_is_bounded_by_half_a_step():
    x = torch.randn(1000).clamp(-1, 1)
    scale = torch.tensor(2.0 / 255)
    out = quantize_dequantize(x, scale, torch.tensor(0.0), -128, 127)
    assert (out - x).abs().max() <= scale.item() / 2 + 1e-6


def test_straight_through_estimator_passes_gradient():
    x = torch.randn(64, requires_grad=True)
    out = quantize_dequantize(x, torch.tensor(0.01), torch.tensor(0.0), -128, 127)
    out.sum().backward()
    # STE => identity gradient. Plain rounding would give zeros everywhere.
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_symmetric_quantization_keeps_zero_exact():
    fq = FakeQuantize(per_channel=False, symmetric=True)
    fq(torch.tensor([-3.0, 5.0]))
    out = fq(torch.zeros(4))
    torch.testing.assert_close(out, torch.zeros(4), atol=1e-6, rtol=0)


def test_per_channel_gives_each_channel_its_own_scale():
    """A weight tensor with wildly different per-channel ranges.

    Fusing three RepConv branches produces exactly this: channels with genuinely
    different dynamic ranges. One shared scale would be set by the widest channel
    and crush the narrow ones to a handful of levels.
    """
    w = torch.zeros(2, 4, 3, 3)
    w[0] = torch.randn(4, 3, 3) * 100.0
    w[1] = torch.randn(4, 3, 3) * 0.01

    per_ch = FakeQuantize(per_channel=True, symmetric=True)
    per_ch(w)
    err_pc = (per_ch(w) - w)[1].abs().max()

    per_t = FakeQuantize(per_channel=False, symmetric=True)
    per_t(w)
    err_pt = (per_t(w) - w)[1].abs().max()

    # The small-range channel must be far better served by per-channel scaling.
    assert err_pc < err_pt / 100


def test_disabled_fake_quant_is_a_passthrough():
    fq = FakeQuantize()
    x = torch.randn(16)
    fq(x)
    fq.enabled = False
    torch.testing.assert_close(fq(x), x)


def test_frozen_observer_stops_tracking_new_ranges():
    fq = FakeQuantize(per_channel=False, symmetric=True, momentum=0.5)
    fq(torch.tensor([-1.0, 1.0]))
    fq.observing = False
    before = fq.running_max.clone()
    fq(torch.tensor([-100.0, 100.0]))
    torch.testing.assert_close(fq.running_max, before)


def _fused_model():
    model = EfficientDetector(
        nc=4, base_ch=8, depths=(1, 1, 1, 1), neck_ch=16, tower_ch=16
    ).eval()
    fuse_model(model)
    return model


def test_prepare_qat_refuses_an_unfused_model():
    model = EfficientDetector(
        nc=4, base_ch=8, depths=(1, 1, 1, 1), neck_ch=16, tower_ch=16
    ).eval()
    with pytest.raises(RuntimeError, match="unfused"):
        prepare_qat(model)


def test_prepare_qat_wraps_every_conv_and_leaves_the_original_alone():
    model = _fused_model()
    qmodel = prepare_qat(model)

    assert any(isinstance(m, QuantConv2d) for m in qmodel.modules())
    assert not any(isinstance(m, QuantConv2d) for m in model.modules())
    # No bare Conv2d should survive outside a QuantConv2d wrapper.
    for module in qmodel.modules():
        if isinstance(module, QuantConv2d):
            continue
        for child in module.children():
            assert not isinstance(child, nn.Conv2d) or isinstance(module, QuantConv2d)


def test_quantized_model_runs_and_stays_close_to_fp32():
    torch.manual_seed(0)
    model = _fused_model()
    qmodel = prepare_qat(model)

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        qmodel(x)                       # populate observers
        set_observing(qmodel, False)
        fp32 = model(x)[2]
        int8 = qmodel(x)[2]

    assert torch.isfinite(int8).all()
    # Box coordinates in pixels: INT8 should not move them by more than a few px.
    assert (fp32 - int8).abs().max() < 10.0


def test_disabling_fake_quant_recovers_fp32_exactly():
    torch.manual_seed(0)
    model = _fused_model()
    qmodel = prepare_qat(model)

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        qmodel(x)
        set_fake_quant(qmodel, False)
        torch.testing.assert_close(qmodel(x)[2], model(x)[2], rtol=1e-5, atol=1e-5)


def test_qat_model_is_trainable():
    model = _fused_model()
    qmodel = prepare_qat(model)
    out = qmodel(torch.randn(1, 3, 64, 64))[1]
    out.sum().backward()
    grads = [p.grad for p in qmodel.parameters() if p.requires_grad and p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_calibrate_populates_observers():
    model = _fused_model()
    qmodel = prepare_qat(model)
    batches = [(torch.randn(2, 3, 64, 64),) for _ in range(3)]
    calibrate(qmodel, batches, n_batches=3)

    observers = [m for m in qmodel.modules() if isinstance(m, FakeQuantize)]
    assert observers
    assert all(torch.isfinite(o.running_min).all() for o in observers)
