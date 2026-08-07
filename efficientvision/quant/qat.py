"""INT8 quantization-aware training (task #13).

Post-training quantization is tried first because it is nearly free; QAT exists
for when PTQ loses too much. The whole reason the backbone uses the *QARepVGG*
block rather than the original RepVGG one is that RepVGG's fused weights quantize
catastrophically (72.4% -> 52.2% top-1 on ImageNet). That claim is currently
adopted on the strength of the published result: `scripts/quant_friendliness.py`
measured a ratio of only 1.031 between the two blocks on randomly-initialised
weights, which is **inconclusive**, not confirmatory. Random weights have no
reason to develop the pathological outlier statistics that training produces, so
the comparison must be re-run on trained weights before the claim is treated as
verified here. See `docs/research-plan.md`.

Fake quantization
-----------------
QAT inserts fake-quant nodes that round activations and weights to the INT8 grid
in the forward pass while passing gradients straight through (STE). The model
therefore learns weights that survive rounding, instead of being rounded after the
fact and hoping for the best.

Weights use **per-channel symmetric** quantization, activations **per-tensor
affine**. Per-channel matters for a fused RepConv specifically: fusion sums three
branches, so output channels end up with genuinely different dynamic ranges, and
a single per-tensor weight scale would be set by the widest channel and crush all
the others.

Fuse before quantizing. Quantizing the unfused training graph calibrates ranges
for tensors that do not exist at inference.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


def quantize_dequantize(
    x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, qmin: int, qmax: int
) -> torch.Tensor:
    """Round to the INT8 grid and back, with a straight-through gradient.

    The rounding itself has zero gradient almost everywhere, so `(q - x).detach()`
    makes the forward pass quantized while the backward pass sees the identity.
    Without the STE no gradient would reach anything upstream.
    """
    q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    dq = (q - zero_point) * scale
    return x + (dq - x).detach()


class FakeQuantize(nn.Module):
    """Observes ranges, then fake-quantizes.

    Args:
        per_channel: per-output-channel scales (weights) vs one scale (activations).
        symmetric: zero_point pinned to 0. Correct for weights, which are roughly
            centred; wrong for post-ReLU activations, which are one-sided.
        momentum: EMA rate for observed min/max.
    """

    def __init__(
        self,
        per_channel: bool = False,
        symmetric: bool = True,
        momentum: float = 0.01,
        qmin: int = -128,
        qmax: int = 127,
    ) -> None:
        super().__init__()
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.momentum = momentum
        self.qmin, self.qmax = qmin, qmax
        self.register_buffer("running_min", torch.tensor(float("inf")))
        self.register_buffer("running_max", torch.tensor(float("-inf")))
        self.enabled = True
        self.observing = True

    def _observe(self, x: torch.Tensor) -> None:
        if self.per_channel:
            flat = x.reshape(x.shape[0], -1)
            lo, hi = flat.min(1).values, flat.max(1).values
        else:
            lo, hi = x.min().reshape(1), x.max().reshape(1)

        if not torch.isfinite(self.running_min).all() or self.running_min.shape != lo.shape:
            self.running_min = lo.detach().clone()
            self.running_max = hi.detach().clone()
        else:
            m = self.momentum
            self.running_min = (1 - m) * self.running_min + m * lo.detach()
            self.running_max = (1 - m) * self.running_max + m * hi.detach()

    def _params(self) -> tuple[torch.Tensor, torch.Tensor]:
        lo = torch.minimum(self.running_min, torch.zeros_like(self.running_min))
        hi = torch.maximum(self.running_max, torch.zeros_like(self.running_max))
        if self.symmetric:
            span = torch.maximum(hi.abs(), lo.abs())
            scale = (2 * span / (self.qmax - self.qmin)).clamp(min=1e-8)
            zp = torch.zeros_like(scale)
        else:
            scale = ((hi - lo) / (self.qmax - self.qmin)).clamp(min=1e-8)
            zp = torch.round(self.qmin - lo / scale)
        return scale, zp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if self.observing:
            with torch.no_grad():
                self._observe(x)
        if not torch.isfinite(self.running_min).all():
            return x
        scale, zp = self._params()
        if self.per_channel:
            shape = [-1] + [1] * (x.ndim - 1)
            scale, zp = scale.reshape(shape), zp.reshape(shape)
        return quantize_dequantize(x, scale, zp, self.qmin, self.qmax)


class QuantConv2d(nn.Module):
    """A Conv2d with fake-quant on its input activations and its weights."""

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        self.conv = conv
        self.act_fq = FakeQuantize(per_channel=False, symmetric=False)
        self.wt_fq = FakeQuantize(per_channel=True, symmetric=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(
            self.act_fq(x),
            self.wt_fq(self.conv.weight),
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )


def prepare_qat(model: nn.Module, inplace: bool = False) -> nn.Module:
    """Wrap every Conv2d in fake-quant nodes.

    The model must already be fused: quantizing the unfused graph calibrates
    ranges for tensors that do not exist at inference time.
    """
    from efficientvision.models.blocks import RepConv

    target = model if inplace else copy.deepcopy(model)
    unfused = [m for m in target.modules() if isinstance(m, RepConv) and not m.fused]
    if unfused:
        raise RuntimeError(
            f"{len(unfused)} RepConv blocks are still unfused. Call fuse_model() "
            "first -- quantizing the training graph calibrates ranges for tensors "
            "that do not exist at inference."
        )

    def convert(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                setattr(module, name, QuantConv2d(child))
            else:
                convert(child)

    convert(target)
    return target


def set_observing(model: nn.Module, observing: bool) -> None:
    """Freeze or unfreeze range collection.

    Ranges are frozen partway through QAT: once weights have adapted, letting the
    observers keep chasing a moving distribution keeps the effective quantization
    grid shifting under the optimizer and the loss stops settling.
    """
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.observing = observing


def set_fake_quant(model: nn.Module, enabled: bool) -> None:
    """Turn fake quantization on or off, e.g. to measure the FP32 reference."""
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.enabled = enabled


@torch.no_grad()
def calibrate(model: nn.Module, dataloader, n_batches: int = 32, device: str = "cpu") -> None:
    """Run batches forward to populate observer ranges before QAT begins."""
    model = model.to(device).eval()
    set_observing(model, True)
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        model(imgs.to(device))
