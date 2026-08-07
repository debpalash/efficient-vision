"""Reparameterizable convolution blocks.

See ARCHITECTURE.md for why the backbone uses dense 3x3 convolutions rather than
depthwise separable ones. The short version: depthwise convs have excellent FLOP
counts but very low arithmetic intensity, so on CPU they are memory-bound and
cannot saturate AVX2/AVX-512 vector units. Dense 3x3 convs are what oneDNN and
XNNPACK GEMM kernels are fastest at.

The trick that makes plain 3x3 stacks trainable to high accuracy is structural
reparameterization (RepVGG, Ding et al. 2021): train as a multi-branch block for
gradient diversity, then fold the branches into a single 3x3 convolution for
inference. The fusion is exact algebra, not an approximation -- `fuse()` produces
bit-comparable outputs (to floating-point tolerance), which `tests/test_blocks.py`
asserts.

We use the *QARepVGG* variant of the block (Chu et al. 2022, arXiv 2212.01593),
not the original RepVGG one. This matters more than it looks:

Original RepVGG puts an independent BatchNorm on every branch --
`BN(W3*x) + BN(W1*x) + BN(x)`. Each BN rescales its branch by a different factor
before the sum, and the fused kernel inherits the combination. The result has
pathological weight statistics: a few large outliers stretch the quantization
range so far that INT8 collapses. Measured on ImageNet, RepVGG-A0 drops 72.4% ->
52.2% top-1 under standard INT8 post-training quantization -- a 20-point loss
caused entirely by where the BatchNorms sit.

QARepVGG restructures to `BN(W3*x + W1*x + x)`: no BatchNorm on the identity or
1x1 branches, and a single BatchNorm after the summation. This keeps the
multi-branch training signal while producing a fused kernel with well-behaved
statistics -- QARepVGG-A0 quantizes to within 1 point of its FP32 accuracy. The
inference cost is identical, since every branch folds away regardless.

Since INT8 is one of the top latency levers for this project, a block that
quantizes badly is disqualifying. See `docs/research-plan.md`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn(in_ch: int, out_ch: int, kernel: int, stride: int, padding: int) -> nn.Sequential:
    """Conv without bias followed by BatchNorm.

    Bias is omitted because BatchNorm's shift subsumes it; carrying both would be
    redundant parameters and would complicate fusion.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
    )


class RepConv(nn.Module):
    """3x3 convolution, trained multi-branch and deployed as a single kernel.

    QARepVGG block structure (see module docstring for why, not RepVGG's):

        Training graph:  BN( 3x3+BN(x)  +  1x1(x)  +  x )
        Inference graph: one 3x3 convolution with bias.

    Only the 3x3 branch carries its own BatchNorm; the 1x1 branch is a bare
    bias-free convolution and the identity branch is the untouched input. A
    single BatchNorm follows the summation. The identity branch exists only when
    the block preserves both resolution and width (stride 1, equal channels).

    Call `fuse()` before export. The block is idempotent under repeated `fuse()`
    calls and refuses to fuse while in training mode, since folding BatchNorm
    using running statistics mid-training would silently corrupt the model.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.fused = False

        self.branch_3x3 = _conv_bn(in_ch, out_ch, 3, stride, 1)
        # No BatchNorm and no bias here: an independent BN on this branch is
        # exactly what wrecks the fused weight distribution under INT8.
        self.branch_1x1 = nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False)
        # Bare identity -- also deliberately un-normalized.
        self.branch_id = (
            nn.Identity() if (in_ch == out_ch and stride == 1) else None
        )
        # The one BatchNorm, applied to the summed branches.
        self.post_bn = nn.BatchNorm2d(out_ch)

        # ReLU by default: it vectorizes to a single `max`, fuses into the
        # preceding convolution, and quantizes to INT8 far better than SiLU.
        self.act = act()
        self.reparam: nn.Conv2d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            assert self.reparam is not None
            return self.act(self.reparam(x))

        out = self.branch_3x3(x) + self.branch_1x1(x)
        if self.branch_id is not None:
            out = out + self.branch_id(x)
        return self.act(self.post_bn(out))

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _fuse_conv_bn(self, block: nn.Sequential) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold a BatchNorm into its preceding convolution.

        For conv weight W and BN parameters (gamma, beta, mean, var, eps):
            scale = gamma / sqrt(var + eps)
            W' = W * scale        (broadcast over output channels)
            b' = beta - mean * scale
        """
        conv, bn = block[0], block[1]
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        weight = conv.weight * scale.reshape(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * scale
        return weight, bias

    def _identity_kernel(self, ref: torch.Tensor) -> torch.Tensor:
        """The 3x3 kernel that reproduces the input unchanged.

        1 at the centre tap for each matching input/output channel, 0 elsewhere.
        Unlike RepVGG, this branch carries no BatchNorm, so there is nothing to
        fold into it -- the kernel is used as-is.
        """
        kernel = torch.zeros(
            self.in_ch, self.in_ch, 3, 3, device=ref.device, dtype=ref.dtype
        )
        for c in range(self.in_ch):
            kernel[c, c, 1, 1] = 1.0
        return kernel

    @torch.no_grad()
    def fuse(self) -> None:
        """Collapse all branches into one 3x3 convolution.

        Raises:
            RuntimeError: if called while training. BatchNorm uses running
                statistics for fusion, so fusing mid-training would bake in stale
                estimates and diverge from the training-time forward pass.
        """
        if self.fused:
            return
        if self.training:
            raise RuntimeError(
                "RepConv.fuse() requires eval mode -- fusing folds BatchNorm "
                "running statistics, which are still being updated during "
                "training. Call model.eval() first."
            )

        # Step 1: collapse the branches into a single (weight, bias) pair. Only
        # the 3x3 branch contributes a bias -- the other two are bias-free.
        w, b = self._fuse_conv_bn(self.branch_3x3)

        # A 1x1 kernel is a 3x3 kernel whose only non-zero tap is the centre.
        w = w + torch.nn.functional.pad(self.branch_1x1.weight, [1, 1, 1, 1])

        if self.branch_id is not None:
            w = w + self._identity_kernel(w)

        # Step 2: fold the post-summation BatchNorm into that pair. For the summed
        # branch output y = W*x + b, BN gives
        #     BN(y) = scale*(W*x + b - mean) + beta,   scale = gamma/sqrt(var+eps)
        # so the final kernel is scale*W and the final bias is scale*(b-mean)+beta.
        # This step is what the original RepVGG block cannot do, and the reason
        # its fused weights quantize badly.
        bn = self.post_bn
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = w * scale.reshape(-1, 1, 1, 1)
        b = (b - bn.running_mean) * scale + bn.bias

        self.reparam = nn.Conv2d(
            self.in_ch, self.out_ch, 3, self.stride, 1, bias=True
        ).to(w.device)
        self.reparam.weight.copy_(w)
        self.reparam.bias.copy_(b)

        # Drop the training branches so they are neither exported nor counted in
        # parameter totals.
        del self.branch_3x3, self.branch_1x1, self.branch_id, self.post_bn
        self.branch_3x3 = self.branch_1x1 = self.branch_id = None  # type: ignore[assignment]
        self.post_bn = None  # type: ignore[assignment]
        self.fused = True


def fuse_model(model: nn.Module) -> nn.Module:
    """Fuse every RepConv in a model tree, in place.

    Call once before ONNX export or CPU benchmarking. Benchmarking an unfused
    model measures the training graph and badly understates deployed speed.
    """
    if model.training:
        raise RuntimeError("fuse_model() requires eval mode; call model.eval() first.")
    for module in model.modules():
        if isinstance(module, RepConv):
            module.fuse()
    return model
