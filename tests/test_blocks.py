"""Correctness tests for structural reparameterization.

Fusion is an exactness claim: the deployed single-conv block must compute the
same function as the trained multi-branch block. If it does not, the model loses
accuracy silently at export time -- there is no error, just worse predictions --
so these tests are the guardrail for the whole approach.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from efficientvision.models.blocks import RepConv, fuse_model


def _train_stats(block: RepConv, in_ch: int, steps: int = 5) -> None:
    """Populate BatchNorm running statistics with non-trivial values.

    Freshly constructed BatchNorm has running_mean=0, running_var=1 and identity
    affine parameters, which makes it very close to a no-op. Fusing that would
    pass trivially and prove nothing, so we push real data through and randomize
    the affine parameters to make the test meaningful.
    """
    block.train()
    for _ in range(steps):
        block(torch.randn(4, in_ch, 16, 16))

    for m in block.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, 0.5, 1.5)
            nn.init.uniform_(m.bias, -0.5, 0.5)


@pytest.mark.parametrize(
    "in_ch,out_ch,stride",
    [
        (16, 16, 1),  # identity branch active
        (16, 32, 1),  # channel change -> no identity branch
        (16, 16, 2),  # downsample -> no identity branch
        (32, 16, 2),  # both change
        (1, 1, 1),    # degenerate width
    ],
)
def test_fusion_preserves_output(in_ch: int, out_ch: int, stride: int) -> None:
    """The fused block must match the multi-branch block on identical input."""
    torch.manual_seed(0)
    block = RepConv(in_ch, out_ch, stride=stride)
    _train_stats(block, in_ch)

    block.eval()
    x = torch.randn(2, in_ch, 16, 16)
    with torch.no_grad():
        before = block(x)
        block.fuse()
        after = block(x)

    # Tolerance covers float32 accumulation-order differences between summing
    # three branch outputs and applying one pre-summed kernel. The magnitudes
    # here are O(1), so 1e-5 is tight.
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)


def test_identity_branch_presence() -> None:
    """The identity branch exists only when input and output are addable."""
    assert RepConv(16, 16, stride=1).branch_id is not None
    assert RepConv(16, 32, stride=1).branch_id is None
    assert RepConv(16, 16, stride=2).branch_id is None


def test_qarepvgg_bn_placement() -> None:
    """Only the 3x3 branch may carry a BatchNorm, plus one after the summation.

    This is the QARepVGG structural invariant and it is INT8-critical: putting an
    independent BatchNorm on the 1x1 or identity branch is what causes the fused
    kernel's weight outliers, and with them the ~20-point INT8 accuracy collapse
    measured for original RepVGG. The failure mode is invisible in FP32 -- fusion
    stays exact and every other test here still passes -- so it can only be caught
    structurally, before quantization is ever run.
    """
    block = RepConv(16, 16, stride=1)

    # 1x1 branch: a bare convolution, no BatchNorm, no bias.
    assert isinstance(block.branch_1x1, nn.Conv2d)
    assert block.branch_1x1.bias is None

    # Identity branch: genuinely the input, not a normalized copy of it.
    assert isinstance(block.branch_id, nn.Identity)

    # Exactly two BatchNorms total: one inside the 3x3 branch, one after the sum.
    bns = [m for m in block.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bns) == 2, f"expected 2 BatchNorms (3x3 branch + post-sum), got {len(bns)}"
    assert isinstance(block.post_bn, nn.BatchNorm2d)
    assert any(m is block.post_bn for m in bns)


def test_post_bn_is_actually_applied() -> None:
    """The post-summation BatchNorm must affect the forward pass.

    Guards against the BN being constructed but left out of `forward`, which
    would make fusion and training disagree while every shape check still passes.
    """
    torch.manual_seed(0)
    block = RepConv(8, 8)
    _train_stats(block, 8)
    block.eval()

    x = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        baseline = block(x)
        # Perturb only the post-sum BN; the output must move.
        block.post_bn.bias.add_(1.0)
        shifted = block(x)

    assert not torch.allclose(baseline, shifted), "post_bn has no effect on forward()"


def test_fuse_is_idempotent() -> None:
    """Fusing twice must not corrupt the block."""
    torch.manual_seed(0)
    block = RepConv(16, 16)
    _train_stats(block, 16)
    block.eval()

    x = torch.randn(2, 16, 16, 16)
    with torch.no_grad():
        block.fuse()
        once = block(x)
        block.fuse()  # no-op
        twice = block(x)

    torch.testing.assert_close(once, twice, rtol=0, atol=0)


def test_fuse_rejects_training_mode() -> None:
    """Fusing mid-training would bake in stale running statistics."""
    block = RepConv(16, 16)
    block.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        block.fuse()


def test_fusion_reduces_parameter_count() -> None:
    """Fusion should collapse three branches into one kernel."""
    block = RepConv(32, 32)
    before = sum(p.numel() for p in block.parameters())
    block.eval()
    block.fuse()
    after = sum(p.numel() for p in block.parameters())
    assert after < before, f"expected fewer params after fusion, got {before} -> {after}"


def test_fuse_model_walks_nested_modules() -> None:
    """fuse_model must reach RepConvs nested inside containers."""
    torch.manual_seed(0)
    model = nn.Sequential(RepConv(8, 8), nn.Sequential(RepConv(8, 16, stride=2)))
    model.eval()

    x = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        before = model(x)
        fuse_model(model)
        after = model(x)

    assert all(m.fused for m in model.modules() if isinstance(m, RepConv))
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)


def test_fuse_model_rejects_training_mode() -> None:
    model = nn.Sequential(RepConv(8, 8))
    model.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        fuse_model(model)
