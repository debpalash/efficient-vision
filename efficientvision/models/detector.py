"""The assembled student detector: backbone -> neck -> head (task #10 support).

Keeping assembly in one place means the training loop, the exporter, and the
benchmark all instantiate the same graph. A model that is wired slightly
differently in the exporter than in training is the classic source of "the ONNX
is less accurate than the checkpoint".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from efficientvision.models.backbone import RepBackbone
from efficientvision.models.head import DetectHead, decode_boxes, make_anchor_points
from efficientvision.models.neck import PANNeck


class EfficientDetector(nn.Module):
    """Anchor-free axis-aligned detector.

    `forward` returns raw head outputs plus the anchor geometry, because the loss
    needs the undecoded distributions while inference needs decoded boxes. Use
    `predict` for the decoded form.
    """

    def __init__(
        self,
        nc: int,
        base_ch: int = 32,
        depths: tuple[int, int, int, int] = (2, 4, 6, 2),
        width_mult: float = 1.0,
        max_width: int = 256,
        neck_ch: int = 96,
        tower_ch: int = 64,
        reg_max: int = 8,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.backbone = RepBackbone(
            base_ch=base_ch, depths=depths, width_mult=width_mult, max_width=max_width
        )
        self.neck = PANNeck(self.backbone.out_channels, neck_ch=neck_ch)
        self.head = DetectHead(
            self.neck.out_channels, nc=nc, reg_max=reg_max, tower_ch=tower_ch,
            strides=self.backbone.out_strides,
        )

    def forward(self, x: torch.Tensor):
        feats = self.neck(self.backbone(x))
        box_dist, cls_logits, strides = self.head(feats)
        anchors = make_anchor_points(feats, self.backbone.out_strides)
        boxes = decode_boxes(box_dist, anchors, strides, self.head.dfl)
        return box_dist, cls_logits, boxes, anchors, strides

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(boxes, scores)` as (B, A, 4) xyxy pixels and (B, A, nc)."""
        _, cls_logits, boxes, _, _ = self(x)
        return boxes, cls_logits.permute(0, 2, 1).sigmoid()
