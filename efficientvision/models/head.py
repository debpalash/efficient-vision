"""Anchor-free detection head with DFL box regression (tasks #6/#7).

Axis-aligned only. The OBB head is deferred (task #6): every box in project-497 is
5-token axis-aligned, so an angle branch could be written but never trained or
validated on this data, and an unvalidated branch is worse than no branch.

Box representation
------------------
Distances from each grid cell centre to the four box edges (left, top, right,
bottom), in stride units. Anchor-free, so there are no anchor hyperparameters to
tune per dataset.

Each distance is predicted as a **distribution** over `reg_max + 1` discrete bins
rather than a single scalar (Distribution Focal Loss, Li et al. 2020). The decoded
distance is the softmax-weighted expectation over bins. This matters here: CAPTCHA
glyphs sit on noisy backgrounds with ambiguous edges, and a distribution can
represent "the edge is somewhere in this range" while a scalar regression is
forced to commit. The expectation is also sub-pixel accurate without extra cost.

Separate box and class towers
-----------------------------
Localisation wants texture and edges; classification wants shape identity. Sharing
one tower makes the two objectives compete for the same features. Two shallow
towers cost little at these resolutions and train more cleanly.

The class tower's final bias is initialised to a large negative value. At init,
essentially every one of the thousands of grid cells is background, so a neutral
bias makes the classification loss start enormous and dominated by background --
the well-known focal-loss init from RetinaNet. Without it, early training wastes
epochs just driving all logits down.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from efficientvision.models.blocks import RepConv


class DFL(nn.Module):
    """Decode a per-bin distribution into an expected distance.

    Implemented as a fixed 1x1 convolution holding the bin indices `0..reg_max`,
    so the softmax-weighted sum is a single kernel call that every export target
    supports. The weights are constant and excluded from optimisation.
    """

    def __init__(self, reg_max: int = 8) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.proj = nn.Conv2d(reg_max + 1, 1, 1, bias=False)
        self.proj.weight.data.copy_(
            torch.arange(reg_max + 1, dtype=torch.float32).reshape(1, -1, 1, 1)
        )
        self.proj.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: (B, 4*(reg_max+1), A) -> (B, 4, A)."""
        b, _, a = x.shape
        x = x.reshape(b, 4, self.reg_max + 1, a).transpose(1, 2)
        return self.proj(x.softmax(dim=1)).reshape(b, 4, a)


class DetectHead(nn.Module):
    """Multi-level anchor-free head producing raw box distributions and class logits.

    Args:
        in_channels: per-level input widths from the neck.
        nc: number of classes.
        reg_max: number of DFL bins minus one. Distances are capped at `reg_max`
            stride units, so this bounds the largest representable box: at stride
            8 with `reg_max=8` a P3 cell can describe an object up to 128 px wide.
        tower_ch: width of the box/class towers.
        act: activation constructor.

    Returns:
        `forward` yields `(box_dist, cls_logits, strides)`:
          - `box_dist`:   (B, 4*(reg_max+1), A) raw logits, undecoded
          - `cls_logits`: (B, nc, A) raw logits, no sigmoid applied
          - `strides`:    (A,) the stride each anchor belongs to
        where A is the total anchor count across levels. Decoding to boxes is
        `decode_boxes`; loss operates on the raw outputs.
    """

    def __init__(
        self,
        in_channels: tuple[int, ...],
        nc: int,
        reg_max: int = 8,
        tower_ch: int = 64,
        strides: tuple[int, ...] = (8, 16, 32),
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if len(in_channels) != len(strides):
            raise ValueError(
                f"got {len(in_channels)} feature levels but {len(strides)} strides"
            )
        self.nc = nc
        self.reg_max = reg_max
        self.strides = strides
        self.no_box = 4 * (reg_max + 1)

        self.box_towers = nn.ModuleList(
            nn.Sequential(
                RepConv(c, tower_ch, stride=1, act=act),
                RepConv(tower_ch, tower_ch, stride=1, act=act),
            )
            for c in in_channels
        )
        self.cls_towers = nn.ModuleList(
            nn.Sequential(
                RepConv(c, tower_ch, stride=1, act=act),
                RepConv(tower_ch, tower_ch, stride=1, act=act),
            )
            for c in in_channels
        )
        self.box_pred = nn.ModuleList(
            nn.Conv2d(tower_ch, self.no_box, 1) for _ in in_channels
        )
        self.cls_pred = nn.ModuleList(
            nn.Conv2d(tower_ch, nc, 1) for _ in in_channels
        )
        self.dfl = DFL(reg_max)
        self._init_bias()

    def _init_bias(self) -> None:
        # Prior probability that any given cell is a positive. See module docstring.
        prior = 0.01
        bias = -math.log((1 - prior) / prior)
        for layer in self.cls_pred:
            nn.init.constant_(layer.bias, bias)
        for layer in self.box_pred:
            nn.init.constant_(layer.bias, 0.0)

    def forward(
        self, feats: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes, classes, strides = [], [], []
        for i, f in enumerate(feats):
            b = self.box_pred[i](self.box_towers[i](f))
            c = self.cls_pred[i](self.cls_towers[i](f))
            n = b.shape[-2] * b.shape[-1]
            boxes.append(b.flatten(2))
            classes.append(c.flatten(2))
            strides.append(
                torch.full((n,), float(self.strides[i]), device=f.device, dtype=f.dtype)
            )
        return torch.cat(boxes, 2), torch.cat(classes, 2), torch.cat(strides)


def make_anchor_points(
    feats: tuple[torch.Tensor, ...], strides: tuple[int, ...]
) -> torch.Tensor:
    """Cell-centre coordinates in input-image pixels, concatenated across levels.

    Centres sit at `(i + 0.5) * stride`, not `i * stride`. The half-cell offset is
    what makes the four predicted distances symmetric about the cell rather than
    biased toward its top-left corner.

    Returns: (A, 2) tensor of (x, y).
    """
    points = []
    for f, s in zip(feats, strides):
        h, w = f.shape[-2:]
        ys, xs = torch.meshgrid(
            torch.arange(h, device=f.device, dtype=f.dtype),
            torch.arange(w, device=f.device, dtype=f.dtype),
            indexing="ij",
        )
        pts = torch.stack(((xs + 0.5) * s, (ys + 0.5) * s), dim=-1)
        points.append(pts.reshape(-1, 2))
    return torch.cat(points)


def decode_boxes(
    box_dist: torch.Tensor,
    anchors: torch.Tensor,
    strides: torch.Tensor,
    dfl: DFL,
) -> torch.Tensor:
    """Turn raw DFL logits into xyxy boxes in input-image pixels.

    Args:
        box_dist: (B, 4*(reg_max+1), A) raw logits.
        anchors: (A, 2) cell centres in pixels.
        strides: (A,) stride per anchor.
        dfl: the head's DFL module.

    Returns: (B, A, 4) boxes as (x1, y1, x2, y2).
    """
    d = dfl(box_dist) * strides.reshape(1, 1, -1)   # (B, 4, A), pixels
    d = d.permute(0, 2, 1)                          # (B, A, 4) = l, t, r, b
    cx, cy = anchors[:, 0], anchors[:, 1]
    x1 = cx - d[..., 0]
    y1 = cy - d[..., 1]
    x2 = cx + d[..., 2]
    y2 = cy + d[..., 3]
    return torch.stack((x1, y1, x2, y2), dim=-1)
