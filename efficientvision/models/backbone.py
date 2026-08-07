"""RepVGG-style backbone built from `RepConv` (task #4).

A plain stack of 3x3 convolutions with no branches at inference time. See
`blocks.py` for why dense 3x3 beats depthwise-separable on CPU, and why the block
is the QARepVGG variant rather than the original RepVGG one.

Shape of the network
--------------------
Stem (stride 2) then four stages, each opening with a stride-2 RepConv followed by
`depth[i] - 1` stride-1 RepConvs. Total stride 32. The last three stages are
returned as the P3/P4/P5 pyramid at strides 8/16/32.

Why the widths taper the way they do
------------------------------------
Cost per stage is roughly `H*W*Cin*Cout`. Halving resolution quarters the spatial
term, so doubling width per stage keeps per-stage cost roughly flat -- the
standard schedule. But the *last* stage is where parameters concentrate while
contributing least to small-object detection, so `width_mult` is applied with a
cap (`max_width`) instead of doubling without limit. On a character-detection
workload the stride-32 map is 10x10 for a 320px input, which is coarser than any
glyph; spending 512+ channels there buys very little.

Nothing here is claimed to be faster than the baseline yet. Latency is measured in
task #16 against a locally measured YOLO26n, after fusing -- benchmarking the
unfused training graph would understate deployed speed (see CLAUDE.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from efficientvision.models.blocks import RepConv


def _round_ch(ch: float, divisor: int = 8) -> int:
    """Round a channel count to a multiple of `divisor`.

    GEMM kernels in oneDNN/XNNPACK are written against SIMD register widths;
    channel counts that are not a multiple of 8 force a padded or scalar tail
    loop and cost real time for no accuracy. Never rounds down by more than 10%.
    """
    rounded = max(divisor, int(ch + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * ch:
        rounded += divisor
    return int(rounded)


class RepBackbone(nn.Module):
    """Reparameterizable CPU-first backbone returning a P3/P4/P5 feature pyramid.

    Args:
        in_ch: input channels (3 for RGB).
        base_ch: width of the stem. Stage widths are derived from this.
        depths: number of RepConv blocks per stage, including the stride-2 opener.
        width_mult: global width scaler for the depth/width search in task #12.
        max_width: cap on stage width; see the module docstring.
        act: activation constructor, passed through to every RepConv.

    Returns:
        `forward` yields `(p3, p4, p5)` at strides 8, 16 and 32.
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 32,
        depths: tuple[int, int, int, int] = (2, 4, 6, 2),
        width_mult: float = 1.0,
        max_width: int = 256,
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if len(depths) != 4:
            raise ValueError(f"depths must have 4 entries, got {len(depths)}")
        if min(depths) < 1:
            raise ValueError("each stage needs at least its stride-2 opener")

        stem_ch = _round_ch(base_ch * width_mult)
        self.stem = RepConv(in_ch, stem_ch, stride=2, act=act)

        widths = [
            min(_round_ch(base_ch * (2 ** (i + 1)) * width_mult), max_width)
            for i in range(4)
        ]

        stages = []
        prev = stem_ch
        for width, depth in zip(widths, depths):
            blocks = [RepConv(prev, width, stride=2, act=act)]
            blocks += [RepConv(width, width, stride=1, act=act) for _ in range(depth - 1)]
            stages.append(nn.Sequential(*blocks))
            prev = width
        self.stages = nn.ModuleList(stages)

        # Consumed by the neck (task #5) so it does not have to re-derive widths.
        self.out_channels = tuple(widths[1:])
        self.out_strides = (8, 16, 32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stages[0](x)          # stride 4  -- too fine to carry forward
        p3 = self.stages[1](x)         # stride 8
        p4 = self.stages[2](p3)        # stride 16
        p5 = self.stages[3](p4)        # stride 32
        return p3, p4, p5
