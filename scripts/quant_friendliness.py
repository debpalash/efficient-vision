"""Verify the QARepVGG BN placement actually improves quantization-friendliness.

We adopted QARepVGG on the strength of a published claim (RepVGG-A0 collapsing
72.4 -> 52.2 top-1 under INT8). This script checks the *mechanism* holds in our
own implementation rather than trusting the paper transitively: it builds both BN
placements, drives identical data through both, fuses, and measures how much
damage per-tensor INT8 quantization does to the fused kernel.

This is not an accuracy measurement -- no model is trained here, and nothing in
the output should be read as an accuracy number. It measures weight-distribution
damage, which is the mechanism the accuracy collapse is attributed to.

RESULT (2026-07-18): INCONCLUSIVE. On randomly-initialized blocks driven with
random data, the two placements differ by only ~3% in relative INT8 round-trip
error -- nowhere near enough to explain a 20-point accuracy collapse. This does
NOT refute QARepVGG; it means this probe is too weak to test it. The published
effect arises from BN statistics that diverge across branches over real training,
which 40 steps of random data does not produce. Treat the QARepVGG adoption as
resting on the published result, not on anything measured here. The real test is
INT8 accuracy on a trained backbone (task #13); re-run this against trained
weights at that point.

Output: runs/quant/qarepvgg_vs_repvgg.json
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from efficientvision.models.blocks import RepConv, _conv_bn

ROOT = Path(__file__).resolve().parents[1]


class LegacyRepConv(nn.Module):
    """Original RepVGG placement: an independent BN on every branch.

    Kept here only as the control for this comparison -- it is deliberately not
    exported from the models package.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.in_ch = self.out_ch = ch
        self.branch_3x3 = _conv_bn(ch, ch, 3, 1, 1)
        self.branch_1x1 = _conv_bn(ch, ch, 1, 1, 0)
        self.branch_id = nn.BatchNorm2d(ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.branch_3x3(x) + self.branch_1x1(x) + self.branch_id(x))

    @torch.no_grad()
    def fused_weight(self) -> torch.Tensor:
        def fold(block):
            conv, bn = block[0], block[1]
            s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            return conv.weight * s.reshape(-1, 1, 1, 1)

        w = fold(self.branch_3x3)
        w = w + torch.nn.functional.pad(fold(self.branch_1x1), [1, 1, 1, 1])
        k = torch.zeros(self.in_ch, self.in_ch, 3, 3)
        for c in range(self.in_ch):
            k[c, c, 1, 1] = 1.0
        bn = self.branch_id
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return w + k * s.reshape(-1, 1, 1, 1)


def quant_error(w: torch.Tensor) -> dict:
    """Per-tensor symmetric INT8 round-trip error on a fused kernel.

    Per-tensor (not per-channel) is the point: one scale must cover every
    channel, so a few outlier channels stretch the range and crush the rest.
    """
    scale = w.abs().max() / 127.0
    deq = torch.clamp(torch.round(w / scale), -127, 127) * scale
    err = (deq - w).abs()
    # Outlier ratio: how far the extreme weight sits from the bulk of the tensor.
    outlier_ratio = (w.abs().max() / w.abs().mean()).item()
    return {
        "max_abs": w.abs().max().item(),
        "mean_abs": w.abs().mean().item(),
        "outlier_ratio": outlier_ratio,
        "rel_l2_error": (err.norm() / w.norm()).item(),
    }


def main() -> int:
    torch.manual_seed(0)
    ch, steps = 64, 40

    qa = RepConv(ch, ch)
    legacy = LegacyRepConv(ch)

    # Identical data through both so the comparison isn't confounded by inputs.
    qa.train()
    legacy.train()
    for _ in range(steps):
        x = torch.randn(8, ch, 32, 32)
        qa(x)
        legacy(x)

    # Randomize BN affine params away from the identity-like init, otherwise the
    # branches barely rescale and the effect under test is suppressed.
    for m in list(qa.modules()) + list(legacy.modules()):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, 0.5, 1.5)
            nn.init.uniform_(m.bias, -0.5, 0.5)

    qa.eval()
    legacy.eval()
    qa.fuse()

    res = {
        "note": "weight-distribution comparison only; not an accuracy measurement",
        "verdict": (
            "INCONCLUSIVE — randomly-initialized blocks on random data do not "
            "reproduce the published gap. Too weak to validate or refute "
            "QARepVGG. Re-run on trained weights during task #13."
        ),
        "channels": ch,
        "qarepvgg": quant_error(qa.reparam.weight.detach()),
        "repvgg_legacy": quant_error(legacy.fused_weight()),
    }
    res["error_ratio_legacy_over_qa"] = (
        res["repvgg_legacy"]["rel_l2_error"] / res["qarepvgg"]["rel_l2_error"]
    )

    out = ROOT / "runs" / "quant"
    out.mkdir(parents=True, exist_ok=True)
    (out / "qarepvgg_vs_repvgg.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
