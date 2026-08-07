"""Lightweight PAN neck over the P3/P4/P5 pyramid (task #5).

Standard PAN: a top-down path that carries semantics down to the fine map, then a
bottom-up path that carries localisation back up. Both fusion steps use `RepConv`,
so the whole neck folds into plain 3x3 convolutions at export time alongside the
backbone.

Two deliberate departures from a stock FPN/PAN:

1. **A single narrow `neck_ch` for every level.** Stock necks keep each level at
   the backbone's native width, which makes the P5 path the most expensive part of
   the network while contributing least to small objects. Unifying to one narrow
   width via 1x1 laterals makes the top-down and bottom-up convs cheap and equal.

2. **Nearest-neighbour upsampling, not transposed convolution.** Nearest is a pure
   gather with no arithmetic, it has no weights to quantize, and every CPU runtime
   in the export matrix implements it natively. Transposed conv would add
   parameters, add a quantization-sensitive op, and buy nothing measurable at
   these resolutions.

Addition is used for the top-down merge rather than concatenation: concat doubles
the channel count entering the fusion conv, roughly doubling its cost, and the
laterals have already projected every level to a common width so the tensors are
directly summable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from efficientvision.models.blocks import RepConv


class PANNeck(nn.Module):
    """Top-down then bottom-up feature fusion across three pyramid levels.

    Args:
        in_channels: `(c3, c4, c5)` widths coming from the backbone.
        neck_ch: common width every level is projected to.
        act: activation constructor, passed through to every RepConv.

    Returns:
        `forward` yields `(n3, n4, n5)`, each with `neck_ch` channels, at the same
        strides as the inputs.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        neck_ch: int = 96,
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError(f"expected 3 pyramid levels, got {len(in_channels)}")

        c3, c4, c5 = in_channels
        # 1x1 laterals: pure channel projection, no spatial mixing. Bias-free and
        # BatchNorm-backed so they fold into a single conv like everything else.
        self.lat3 = self._lateral(c3, neck_ch)
        self.lat4 = self._lateral(c4, neck_ch)
        self.lat5 = self._lateral(c5, neck_ch)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down: P5 -> P4 -> P3.
        self.td4 = RepConv(neck_ch, neck_ch, stride=1, act=act)
        self.td3 = RepConv(neck_ch, neck_ch, stride=1, act=act)

        # Bottom-up: N3 -> N4 -> N5. The stride-2 convs do the downsampling, so no
        # pooling layer is needed.
        self.down3 = RepConv(neck_ch, neck_ch, stride=2, act=act)
        self.bu4 = RepConv(neck_ch, neck_ch, stride=1, act=act)
        self.down4 = RepConv(neck_ch, neck_ch, stride=2, act=act)
        self.bu5 = RepConv(neck_ch, neck_ch, stride=1, act=act)

        self.out_channels = (neck_ch, neck_ch, neck_ch)

    @staticmethod
    def _lateral(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _up_to(x: torch.Tensor, ref: torch.Tensor, up: nn.Module) -> torch.Tensor:
        """Upsample `x` to `ref`'s spatial size.

        A 2x upsample only lands exactly on the finer map when that map's size is
        even. Odd sizes arise routinely here because the inputs are wide CAPTCHA
        strips rather than padded squares, so the result is trimmed rather than
        assumed to match.
        """
        x = up(x)
        return x[..., : ref.shape[-2], : ref.shape[-1]]

    def forward(
        self, feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = feats
        l3, l4, l5 = self.lat3(p3), self.lat4(p4), self.lat5(p5)

        t4 = self.td4(l4 + self._up_to(l5, l4, self.up))
        t3 = self.td3(l3 + self._up_to(t4, l3, self.up))

        n3 = t3
        n4 = self.bu4(t4 + self._crop_to(self.down3(n3), t4))
        n5 = self.bu5(l5 + self._crop_to(self.down4(n4), l5))
        return n3, n4, n5

    @staticmethod
    def _crop_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Trim a stride-2 output down to the reference map's size.

        Downsampling an odd-sized map rounds up, so `down3(n3)` can be one pixel
        larger than the level it is being added to.
        """
        return x[..., : ref.shape[-2], : ref.shape[-1]]
