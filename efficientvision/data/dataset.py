"""YOLO-format dataset and augmentation for character detection (task #10 support).

Reads the mirrored `images/{split}` + `labels/{split}` tree that
`scripts/build_split.py` produces, so the custom trainer consumes exactly the same
data as the Ultralytics feasibility runs. Any accuracy difference between the two
is then attributable to the model, not to a divergent input pipeline.

Augmentation policy is unusually restrictive, and both restrictions are measured
rather than stylistic:

1. **No flips, rotation, shear, or perspective.** These relabel glyphs -- a
   horizontal flip turns `b` into `d` and `6` into `9`. This is a correctness
   constraint, not a tuning knob.

2. **No mosaic.** Mosaic tiles 4 images into one canvas, halving every character's
   effective pixel size. Ablation B in `docs/feasibility-gate.md` measured
   `mosaic=1.0 -> 0.0` as **+11.5 points of solve rate** (0.230 -> 0.344) on the
   hardest source, while mAP50 moved the *other* way. It is deliberately absent
   rather than merely defaulted off.

What is left is photometric jitter plus small translate/scale, which preserve
glyph identity.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def letterbox(
    img: np.ndarray, size: tuple[int, int], color: int = 114
) -> tuple[np.ndarray, float, float, float]:
    """Resize preserving aspect ratio, then pad to `size` (h, w).

    Returns `(image, scale, pad_x, pad_y)` so labels can be mapped with the same
    transform. Padding is centred, which keeps the object distribution symmetric
    -- corner padding would bias every box toward one side of the frame.
    """
    import cv2

    ih, iw = img.shape[:2]
    th, tw = size
    scale = min(th / ih, tw / iw)
    nh, nw = int(round(ih * scale)), int(round(iw * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    out = np.full((th, tw, 3), color, dtype=img.dtype)
    pad_y, pad_x = (th - nh) // 2, (tw - nw) // 2
    out[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return out, scale, float(pad_x), float(pad_y)


class CharDetectionDataset(Dataset):
    """Images plus normalized-xywh YOLO labels, emitted as pixel xyxy boxes.

    Args:
        root: split root containing `images/<split>` and `labels/<split>`.
        split: "train" or "val".
        imgsz: square letterbox target.
        augment: enable photometric + small geometric jitter (train only).
        max_boxes: ground-truth slots per image; the batch is padded to this.
            EDA measured max 7 boxes/image, so 16 is generous headroom.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        imgsz: int = 320,
        augment: bool = False,
        max_boxes: int = 16,
        hsv: tuple[float, float, float] = (0.0, 0.7, 0.4),
        translate: float = 0.1,
        scale: float = 0.2,
    ) -> None:
        self.root = Path(root)
        self.imgsz = imgsz
        self.augment = augment
        self.max_boxes = max_boxes
        # hsv_h stays 0 by default: hue is a real discriminative cue on the noisy
        # sources, where glyphs and distractor lines differ mainly by colour.
        self.hsv = hsv
        self.translate = translate
        self.scale = scale

        img_dir = self.root / "images" / split
        lbl_dir = self.root / "labels" / split
        if not img_dir.is_dir():
            raise FileNotFoundError(f"no image directory at {img_dir}")

        self.samples: list[tuple[Path, Path]] = []
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / (p.stem + ".txt")
            if lbl.exists():
                self.samples.append((p, lbl))
        if not self.samples:
            raise RuntimeError(f"no image/label pairs found under {self.root}/{split}")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_labels(path: Path) -> np.ndarray:
        """(N, 5) array of `cls, cx, cy, w, h` in normalized coordinates."""
        rows = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rows.append([float(v) for v in parts[:5]])
        return np.array(rows, dtype=np.float32).reshape(-1, 5)

    def _augment_photometric(self, img: np.ndarray) -> np.ndarray:
        import cv2

        h_gain, s_gain, v_gain = self.hsv
        if not any((h_gain, s_gain, v_gain)):
            return img
        r = np.random.uniform(-1, 1, 3) * np.array([h_gain, s_gain, v_gain]) + 1
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] * r[0]) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def __getitem__(self, i: int):
        import cv2

        img_path, lbl_path = self.samples[i]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read {img_path}")
        ih, iw = img.shape[:2]

        labels = self._read_labels(lbl_path)
        # normalized cxcywh -> absolute xyxy in the ORIGINAL image frame
        boxes = np.zeros((len(labels), 4), dtype=np.float32)
        if len(labels):
            cx, cy = labels[:, 1] * iw, labels[:, 2] * ih
            bw, bh = labels[:, 3] * iw, labels[:, 4] * ih
            boxes[:, 0], boxes[:, 1] = cx - bw / 2, cy - bh / 2
            boxes[:, 2], boxes[:, 3] = cx + bw / 2, cy + bh / 2
        classes = labels[:, 0].astype(np.int64) if len(labels) else np.zeros(0, np.int64)

        if self.augment:
            img = self._augment_photometric(img)

        img, s, pad_x, pad_y = letterbox(img, (self.imgsz, self.imgsz))
        if len(boxes):
            boxes = boxes * s
            boxes[:, [0, 2]] += pad_x
            boxes[:, [1, 3]] += pad_y

        if self.augment and (self.translate or self.scale):
            img, boxes, classes = self._jitter(img, boxes, classes)

        img_t = torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0

        n = min(len(boxes), self.max_boxes)
        gt_boxes = torch.zeros(self.max_boxes, 4)
        # -1 rather than 0: a padded slot must never look like a valid class id if
        # a mask is ever dropped downstream.
        gt_labels = torch.full((self.max_boxes,), -1, dtype=torch.long)
        gt_mask = torch.zeros(self.max_boxes, dtype=torch.bool)
        if n:
            gt_boxes[:n] = torch.from_numpy(boxes[:n])
            gt_labels[:n] = torch.from_numpy(classes[:n])
            gt_mask[:n] = True

        return img_t, gt_boxes, gt_labels, gt_mask

    def _jitter(self, img: np.ndarray, boxes: np.ndarray, classes: np.ndarray):
        """Random translate/scale about the image centre. No rotation or flip."""
        import cv2

        h, w = img.shape[:2]
        s = random.uniform(1 - self.scale, 1 + self.scale)
        tx = random.uniform(-self.translate, self.translate) * w
        ty = random.uniform(-self.translate, self.translate) * h

        m = np.array(
            [[s, 0, (1 - s) * w / 2 + tx], [0, s, (1 - s) * h / 2 + ty]], dtype=np.float32
        )
        img = cv2.warpAffine(img, m, (w, h), borderValue=(114, 114, 114))

        if len(boxes):
            corners = boxes.copy()
            corners[:, [0, 2]] = corners[:, [0, 2]] * s + m[0, 2]
            corners[:, [1, 3]] = corners[:, [1, 3]] * s + m[1, 2]
            np.clip(corners[:, [0, 2]], 0, w, out=corners[:, [0, 2]])
            np.clip(corners[:, [1, 3]], 0, h, out=corners[:, [1, 3]])
            # Drop glyphs the transform pushed (almost) out of frame. Keeping a
            # sliver box would train the model to detect 2-pixel fragments.
            keep = ((corners[:, 2] - corners[:, 0]) > 2) & ((corners[:, 3] - corners[:, 1]) > 2)
            boxes, classes = corners[keep], classes[keep]
        return img, boxes, classes


def collate(batch):
    imgs, boxes, labels, masks = zip(*batch)
    return (
        torch.stack(imgs),
        torch.stack(boxes),
        torch.stack(labels),
        torch.stack(masks),
    )
