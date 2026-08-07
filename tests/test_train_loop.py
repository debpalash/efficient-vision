"""Tests for the dataset and training loop (task #10).

These run on CPU against a synthetic split, so they stay fast and do not contend
with whatever is occupying the GPU. What they pin down is the machinery that
fails silently: label coordinate transforms through letterbox, padding semantics,
and -- most importantly -- that resume restores optimizer state and not just
weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2")

from efficientvision.data.dataset import CharDetectionDataset, collate, letterbox
from efficientvision.train.loop import Trainer, TrainConfig, _lr_at


@pytest.fixture
def split(tmp_path):
    """A 4-image split of 300x100 CAPTCHA-shaped images with 3 boxes each."""
    for name in ("train", "val"):
        (tmp_path / "images" / name).mkdir(parents=True)
        (tmp_path / "labels" / name).mkdir(parents=True)
        for i in range(4):
            img = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / "images" / name / f"{i}.png"), img)
            lines = [f"{c} {0.2 + 0.25 * c:.4f} 0.5 0.1 0.4" for c in range(3)]
            (tmp_path / "labels" / name / f"{i}.txt").write_text("\n".join(lines))
    return tmp_path


def test_letterbox_preserves_aspect_ratio_and_centres_padding():
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    out, s, px, py = letterbox(img, (320, 320))
    assert out.shape == (320, 320, 3)
    assert s == pytest.approx(320 / 300)
    assert px == 0                       # width is the limiting dimension
    assert py == pytest.approx((320 - round(100 * 320 / 300)) // 2)


def test_boxes_survive_the_letterbox_transform(split):
    ds = CharDetectionDataset(split, "val", imgsz=320, augment=False)
    img, boxes, labels, mask = ds[0]

    assert img.shape == (3, 320, 320)
    assert mask.sum() == 3
    valid = boxes[mask]
    # Every box must land inside the letterboxed frame.
    assert (valid[:, 0] >= 0).all() and (valid[:, 2] <= 320).all()
    assert (valid[:, 2] > valid[:, 0]).all() and (valid[:, 3] > valid[:, 1]).all()

    # First box: normalized cx=0.2, w=0.1 on a 300px-wide image, scale 320/300,
    # no x padding -> x1 = (0.2-0.05)*300*(320/300) = 48.
    torch.testing.assert_close(valid[0, 0], torch.tensor(48.0), atol=1.0, rtol=0)


def test_padded_slots_carry_sentinel_label(split):
    ds = CharDetectionDataset(split, "val", imgsz=320)
    _, _, labels, mask = ds[0]
    assert (labels[~mask] == -1).all()
    assert (labels[mask] >= 0).all()


def test_collate_stacks_a_batch(split):
    ds = CharDetectionDataset(split, "train", imgsz=160)
    imgs, boxes, labels, masks = collate([ds[0], ds[1]])
    assert imgs.shape == (2, 3, 160, 160)
    assert boxes.shape == (2, 16, 4)
    assert labels.shape == masks.shape == (2, 16)


def test_augmentation_keeps_boxes_in_frame(split):
    ds = CharDetectionDataset(split, "train", imgsz=320, augment=True)
    for _ in range(20):
        _, boxes, _, mask = ds[0]
        valid = boxes[mask]
        if len(valid):
            assert (valid >= 0).all() and (valid <= 320).all()


def test_lr_warms_up_then_decays():
    cfg = TrainConfig(data_root=".", nc=4, epochs=10, warmup_epochs=2, lr=1e-3)
    assert _lr_at(cfg, 0, 0, 10) == 0.0
    assert _lr_at(cfg, 1, 0, 10) == pytest.approx(5e-4)      # halfway through warmup
    peak = _lr_at(cfg, 2, 0, 10)
    assert peak == pytest.approx(1e-3, rel=1e-2)
    assert _lr_at(cfg, 9, 9, 10) < peak / 10                 # decayed by the end


def _cfg(split, tmp_path, **kw):
    opts = dict(
        data_root=str(split), nc=3, imgsz=64, epochs=1, batch=2, workers=0,
        device="cpu", amp=False, base_ch=8, depths=(1, 1, 1, 1), neck_ch=16,
        tower_ch=16, out_dir=str(tmp_path / "run"),
    )
    opts.update(kw)
    return TrainConfig(**opts)


def test_one_epoch_runs_and_writes_checkpoints(split, tmp_path):
    trainer = Trainer(_cfg(split, tmp_path))
    result = trainer.fit()
    assert len(result["history"]) == 1
    assert np.isfinite(result["history"][0]["loss"])
    assert (tmp_path / "run" / "last.pt").exists()
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "history.json").exists()


def test_resume_restores_optimizer_state_not_just_weights(split, tmp_path):
    """Restoring weights alone silently resets AdamW momentum and the LR position."""
    first = Trainer(_cfg(split, tmp_path, epochs=2))
    first.fit()

    second = Trainer(_cfg(split, tmp_path, epochs=4))
    second.resume(tmp_path / "run" / "last.pt")

    assert second.start_epoch == 2
    assert second.best == first.best

    # AdamW's exp_avg buffers must have come across, not been re-initialised.
    state = second.opt.state_dict()["state"]
    assert state, "optimizer state is empty -- resume restored weights only"
    assert any(v["exp_avg"].abs().sum() > 0 for v in state.values())
