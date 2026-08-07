"""GPU training loop with BF16 AMP and resume (task #10).

BF16 rather than FP16: BF16 carries FP32's exponent range, so it does not need a
gradient scaler and cannot silently produce inf/NaN on a loss spike. On Ada (the
RTX 4070 Laptop this project trains on) BF16 and FP16 run at the same tensor-core
rate, so the stability is free.

Resume restores model, optimizer, scheduler, epoch, and best-metric state
together. Restoring weights alone -- the common shortcut -- silently resets the
optimizer's momentum and the LR schedule position, which shows up as a visible
loss bump at the resume point and makes a resumed run non-comparable to an
uninterrupted one.

VRAM discipline (see CLAUDE.md): CUDA System Memory Fallback is enabled on this
machine, so an oversized batch does NOT raise OOM -- it spills to host RAM and
trains 5-20x slower. `peak_vram_gb` is recorded every epoch and compared against
a budget so the spill is caught by a number rather than by a slow run nobody
questions.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from efficientvision.data.dataset import CharDetectionDataset, collate
from efficientvision.models.detector import EfficientDetector
from efficientvision.models.loss import DetectionLoss


@dataclass
class TrainConfig:
    data_root: str
    nc: int
    imgsz: int = 320
    epochs: int = 200
    batch: int = 32
    lr: float = 1e-3
    weight_decay: float = 5e-4
    warmup_epochs: int = 3
    workers: int = 8
    device: str = "cuda:0"
    amp: bool = True
    out_dir: str = "runs/student/exp"
    # 85% of an 8 GB card. Anything above this is assumed to be spilling to host
    # RAM rather than fitting -- "it did not OOM" is not evidence it fits.
    vram_budget_gb: float = 6.96
    base_ch: int = 32
    depths: tuple[int, int, int, int] = (2, 4, 6, 2)
    width_mult: float = 1.0
    neck_ch: int = 96
    tower_ch: int = 64
    reg_max: int = 8
    seed: int = 0
    history: list = field(default_factory=list)


def build_model(cfg: TrainConfig) -> EfficientDetector:
    return EfficientDetector(
        nc=cfg.nc, base_ch=cfg.base_ch, depths=tuple(cfg.depths),
        width_mult=cfg.width_mult, neck_ch=cfg.neck_ch, tower_ch=cfg.tower_ch,
        reg_max=cfg.reg_max,
    )


def _lr_at(cfg: TrainConfig, epoch: int, step: int, steps_per_epoch: int) -> float:
    """Linear warmup then cosine decay to 1% of the peak."""
    total_warmup = max(1, cfg.warmup_epochs * steps_per_epoch)
    it = epoch * steps_per_epoch + step
    if it < total_warmup:
        return cfg.lr * it / total_warmup
    progress = (it - total_warmup) / max(1, cfg.epochs * steps_per_epoch - total_warmup)
    progress = min(1.0, progress)
    return cfg.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self.model = build_model(cfg).to(self.device)
        self.loss_fn = DetectionLoss(nc=cfg.nc, reg_max=cfg.reg_max).to(self.device)

        # No weight decay on BatchNorm parameters or biases: decaying a BN scale
        # pulls it toward zero, which is a rescaling of the layer rather than
        # regularisation, and it interacts badly with the fused deployment kernel.
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        self.opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.lr,
        )

        self.train_ds = CharDetectionDataset(
            cfg.data_root, "train", imgsz=cfg.imgsz, augment=True
        )
        self.val_ds = CharDetectionDataset(
            cfg.data_root, "val", imgsz=cfg.imgsz, augment=False
        )
        self.train_dl = DataLoader(
            self.train_ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.workers,
            collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=cfg.workers > 0,
        )
        self.val_dl = DataLoader(
            self.val_ds, batch_size=cfg.batch, shuffle=False, num_workers=cfg.workers,
            collate_fn=collate, pin_memory=True, persistent_workers=cfg.workers > 0,
        )

        self.start_epoch = 0
        self.best = float("inf")
        self.amp_dtype = torch.bfloat16 if cfg.amp else torch.float32

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------

    def save(self, path: Path, epoch: int) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.opt.state_dict(),
                "epoch": epoch,
                "best": self.best,
                "config": asdict(self.cfg),
            },
            path,
        )

    def resume(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best = ckpt["best"]
        print(f"resumed from {path} at epoch {self.start_epoch}")

    # ------------------------------------------------------------------
    # train / eval
    # ------------------------------------------------------------------

    def _step_batch(self, batch):
        imgs, gt_boxes, gt_labels, gt_mask = (t.to(self.device, non_blocking=True) for t in batch)
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.cfg.amp):
            box_dist, cls_logits, boxes, anchors, strides = self.model(imgs)
        # Loss in FP32: the assigner's top-k and the IoU terms are sensitive to
        # reduced precision, and the loss is a negligible share of total compute.
        return self.loss_fn(
            box_dist.float(), cls_logits.float(), boxes.float(), anchors.float(),
            strides.float(), gt_labels, gt_boxes, gt_mask,
        )

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        steps = len(self.train_dl)
        totals = {"loss": 0.0, "cls": 0.0, "box": 0.0, "dfl": 0.0}
        t0 = time.time()

        for i, batch in enumerate(self.train_dl):
            lr = _lr_at(self.cfg, epoch, i, steps)
            for g in self.opt.param_groups:
                g["lr"] = lr

            loss, parts = self._step_batch(batch)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.opt.step()

            totals["loss"] += float(loss.detach())
            for k in ("cls", "box", "dfl"):
                totals[k] += parts[k]

        out = {k: v / steps for k, v in totals.items()}
        out["lr"] = lr
        out["seconds"] = time.time() - t0
        if self.device.type == "cuda":
            out["peak_vram_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
            torch.cuda.reset_peak_memory_stats(self.device)
        return out

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        totals = {"loss": 0.0, "cls": 0.0, "box": 0.0, "dfl": 0.0}
        for batch in self.val_dl:
            loss, parts = self._step_batch(batch)
            totals["loss"] += float(loss.detach())
            for k in ("cls", "box", "dfl"):
                totals[k] += parts[k]
        n = max(1, len(self.val_dl))
        return {f"val_{k}": v / n for k, v in totals.items()}

    def fit(self) -> dict:
        history = []
        for epoch in range(self.start_epoch, self.cfg.epochs):
            tr = self.train_one_epoch(epoch)
            va = self.validate()
            row = {"epoch": epoch, **tr, **va}
            history.append(row)

            peak = tr.get("peak_vram_gb", 0.0)
            if peak > self.cfg.vram_budget_gb:
                # Not fatal, but it must never pass unremarked: past this point
                # the run is almost certainly paging to host RAM.
                row["vram_warning"] = (
                    f"peak {peak:.2f} GB exceeds budget {self.cfg.vram_budget_gb:.2f} GB "
                    "-- likely spilling to host RAM; reduce batch"
                )
                print("WARNING:", row["vram_warning"])

            # Update `best` BEFORE writing last.pt. Written the other way round,
            # last.pt carries the previous epoch's best value, so a run resumed
            # from it would compare against a stale threshold and could overwrite
            # a genuinely better best.pt.
            improved = va["val_loss"] < self.best
            if improved:
                self.best = va["val_loss"]
            self.save(self.out / "last.pt", epoch)
            if improved:
                self.save(self.out / "best.pt", epoch)

            (self.out / "history.json").write_text(json.dumps(history, indent=2))
            print(
                f"epoch {epoch:>4}  loss {tr['loss']:.4f}  val {va['val_loss']:.4f}  "
                f"lr {tr['lr']:.2e}  {tr['seconds']:.1f}s  vram {peak:.2f}GB"
            )
        return {"history": history, "best_val_loss": self.best}
