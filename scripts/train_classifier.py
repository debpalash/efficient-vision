"""Train the stage-2 character classifier on 64x64 crops (task #23).

At inference this classifier runs on **detector** boxes, but it trains on
**ground-truth** boxes. That mismatch is the main thing that makes naive
two-stage pipelines underperform: detector boxes are shifted, loose, or tight by
a few pixels, and a classifier trained only on perfect boxes has never seen that.
Box jitter augmentation is therefore not optional here -- it is what makes the two
stages compose.

Augmentation is otherwise the same character-semantic lock used everywhere in this
project: no horizontal or vertical flip, no rotation beyond a few degrees. A flip
turns `b` into `d` and `6` into `9`; it would relabel the data.

Output: runs/classifier/<name>/best.pt + metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=str(ROOT / "datasets" / "crops"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--name", default="char_cls")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.models import resnet18

    crops = Path(args.crops)
    out = ROOT / "runs" / "classifier" / args.name
    out.mkdir(parents=True, exist_ok=True)

    train_tf = transforms.Compose([
        # Box jitter: random resized crop with a narrow scale range simulates the
        # detector's box being slightly off. This is the augmentation that closes
        # the GT-box vs detector-box gap.
        transforms.RandomResizedCrop(args.size, scale=(0.75, 1.0), ratio=(0.85, 1.18)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.5),
        # Small rotation only. Beyond a few degrees this starts turning glyphs
        # into other glyphs, which is a labelling error, not augmentation.
        transforms.RandomRotation(7),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(crops / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(crops / "val", transform=val_tf)
    nc = len(train_ds.classes)
    print(f"{len(train_ds)} train / {len(val_ds)} val crops, {nc} classes")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True,
                          persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=8, pin_memory=True, persistent_workers=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, nc)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr * 10, total_steps=args.epochs * len(train_dl))
    # Label smoothing: the label space contains shape-identical pairs (C/c, O/0,
    # 1/l/I) where a confident wrong answer is worse than a hedged one, and solve
    # rate depends on the confidences being usable by the count-prior decoder.
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = 0.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())

        model.eval()
        correct = n = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                correct += int((model(x).argmax(1) == y).sum())
                n += len(y)
        acc = correct / max(n, 1)
        history.append({"epoch": epoch, "loss": tot / len(train_dl), "val_acc": acc})
        print(f"epoch {epoch:>3}  loss {tot/len(train_dl):.4f}  val_acc {acc:.4f}")

        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "classes": train_ds.classes,
                        "size": args.size, "val_acc": acc}, out / "best.pt")

    (out / "metrics.json").write_text(json.dumps(
        {"best_val_acc": best, "classes": train_ds.classes, "history": history},
        indent=2), encoding="utf-8")
    print(f"\nbest crop-level val accuracy: {best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
