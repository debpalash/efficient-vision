"""Export the student detector to deployable formats (task #14).

Every export path begins by fusing: `RepConv.fuse()` collapses the multi-branch
training graph into single 3x3 convolutions. Exporting the unfused graph would
ship three convolutions and a BatchNorm where one convolution suffices, and every
latency number taken from it would understate deployed speed (CLAUDE.md).

Fusion happens on a **deep copy**, not on the live model. Fusion is destructive --
it deletes the training branches - so fusing in place would silently end the
caller's ability to keep training, which is exactly the sort of surprise an
exporter should not spring on a training script.

Parity is not assumed. `check_parity` runs the exported graph against the PyTorch
one on the same inputs and reports the max absolute deviation, so a broken export
fails loudly here rather than as an unexplained accuracy drop later.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn

from efficientvision.models.blocks import fuse_model


class _InferenceWrapper(nn.Module):
    """Emit `(boxes, scores)` so the exported graph needs no Python post-processing.

    NMS is deliberately left outside the graph: its implementation and its
    thresholds differ across ONNX Runtime, OpenVINO, NCNN and TensorRT, so baking
    one in would make the exports non-comparable to each other. The decode maths,
    which must be identical everywhere, is inside.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        _, cls_logits, boxes, _, _ = self.model(x)
        return boxes, cls_logits.permute(0, 2, 1).sigmoid()


def fused_copy(model: nn.Module) -> nn.Module:
    """An eval-mode, fused deep copy of `model`, leaving the original untouched."""
    clone = copy.deepcopy(model).eval()
    fuse_model(clone)
    return clone


def export_onnx(
    model: nn.Module,
    path: str | Path,
    imgsz: int = 320,
    batch: int = 1,
    opset: int = 17,
    dynamic_batch: bool = False,
) -> Path:
    """Fuse and export to ONNX. Returns the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wrapped = _InferenceWrapper(fused_copy(model)).eval()
    dummy = torch.zeros(batch, 3, imgsz, imgsz)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch"}, "boxes": {0: "batch"}, "scores": {0: "batch"}
        }

    torch.onnx.export(
        wrapped,
        dummy,
        str(path),
        opset_version=opset,
        input_names=["images"],
        output_names=["boxes", "scores"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    return path


def export_torchscript(
    model: nn.Module, path: str | Path, imgsz: int = 320, batch: int = 1
) -> Path:
    """Fuse and export a traced TorchScript module."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = _InferenceWrapper(fused_copy(model)).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, torch.zeros(batch, 3, imgsz, imgsz))
    traced.save(str(path))
    return path


def check_parity(
    model: nn.Module,
    onnx_path: str | Path,
    imgsz: int = 320,
    batch: int = 1,
    threads: int = 1,
    seed: int = 0,
) -> dict:
    """Compare the ONNX graph against PyTorch on identical random input.

    Returns per-output max absolute deviation. `intra_op_num_threads` is pinned
    explicitly -- ONNX Runtime's default thread count collapses on this machine's
    hybrid CPU (see BASELINE.md), and an unpinned session is not reproducible.
    """
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(seed)
    x = torch.randn(batch, 3, imgsz, imgsz)

    ref = _InferenceWrapper(fused_copy(model)).eval()
    with torch.no_grad():
        torch_out = [t.numpy() for t in ref(x)]

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    onnx_out = sess.run(None, {"images": x.numpy()})

    devs = {
        name: float(np.abs(a - b).max())
        for name, a, b in zip(("boxes", "scores"), torch_out, onnx_out)
    }
    devs["max"] = max(devs.values())
    return devs
