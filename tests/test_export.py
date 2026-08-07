"""Export and parity tests (task #14).

The one that matters is `test_onnx_matches_pytorch`: it is the only check that the
whole chain -- QARepVGG fusion algebra, DFL decode, anchor geometry -- survives
into the artifact that actually ships. Everything upstream can be correct while
the export is silently wrong.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from efficientvision.export.onnx import (
    check_parity,
    export_onnx,
    export_torchscript,
    fused_copy,
)
from efficientvision.models.blocks import RepConv
from efficientvision.models.detector import EfficientDetector


def _tiny_model():
    torch.manual_seed(0)
    model = EfficientDetector(
        nc=8, base_ch=8, depths=(1, 1, 1, 1), neck_ch=16, tower_ch=16
    ).eval()
    # Give the BatchNorms non-trivial statistics so fusion is a real transform.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.normal_()
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.normal_(1.0, 0.2)
            m.bias.data.normal_(0.0, 0.2)
    return model


def test_fused_copy_leaves_the_original_trainable():
    model = _tiny_model()
    clone = fused_copy(model)

    assert all(m.fused for m in clone.modules() if isinstance(m, RepConv))
    # The original must still have its training branches intact.
    originals = [m for m in model.modules() if isinstance(m, RepConv)]
    assert originals and not any(m.fused for m in originals)
    assert any(isinstance(m, nn.BatchNorm2d) for m in model.modules())


def test_fused_copy_preserves_outputs():
    model = _tiny_model()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        before = model(x)[2]
        after = fused_copy(model)(x)[2]
    torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-4)


def test_torchscript_roundtrip(tmp_path):
    model = _tiny_model()
    path = export_torchscript(model, tmp_path / "m.pt", imgsz=128)
    loaded = torch.jit.load(str(path))

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        boxes_ref, scores_ref = fused_copy(model).predict(x)
        boxes, scores = loaded(x)
    torch.testing.assert_close(boxes, boxes_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(scores, scores_ref, rtol=1e-4, atol=1e-4)


def test_onnx_export_writes_a_file(tmp_path):
    pytest.importorskip("onnx")
    model = _tiny_model()
    path = export_onnx(model, tmp_path / "m.onnx", imgsz=128)
    assert path.exists() and path.stat().st_size > 0


def test_onnx_matches_pytorch(tmp_path):
    """End-to-end parity: fusion algebra + DFL decode + anchors must all survive."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    model = _tiny_model()
    path = export_onnx(model, tmp_path / "m.onnx", imgsz=128)
    devs = check_parity(model, path, imgsz=128, threads=1)

    # Box coordinates are in pixels, so 1e-3 is well below a pixel; scores are
    # probabilities in [0, 1].
    assert devs["boxes"] < 1e-3, devs
    assert devs["scores"] < 1e-4, devs


def test_onnx_output_shapes(tmp_path):
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    model = _tiny_model()
    path = export_onnx(model, tmp_path / "m.onnx", imgsz=128)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    boxes, scores = sess.run(None, {"images": torch.zeros(1, 3, 128, 128).numpy()})

    anchors = 16 * 16 + 8 * 8 + 4 * 4
    assert boxes.shape == (1, anchors, 4)
    assert scores.shape == (1, anchors, 8)
