"""Find the largest distillation teacher that trains on this GPU.

The plan depends on distilling from a high-capacity teacher, but teacher size is
capped by VRAM, and training memory is dominated by activations -- roughly
proportional to batch x resolution^2 x width -- not by parameter count. An 8 GB
card holds a model whose weights are a small fraction of that budget yet still
OOMs on activations.

Rather than estimate, this probes empirically: run a real forward + backward at
increasing batch sizes and record where it fails.

Run:
    uv run --no-sync python scripts/vram_probe.py
    uv run --no-sync python scripts/vram_probe.py --imgsz 512 --models yolo26m,yolo26l
"""

from __future__ import annotations

import argparse
import gc

import torch

# Candidate teachers, smallest to largest. The best teacher is the largest one
# that trains at a usable batch size -- a huge model at batch 1 trains too slowly
# and its BatchNorm statistics are too noisy to be worth it.
DEFAULT_MODELS = ["yolo26s", "yolo26m", "yolo26l", "yolo26x"]
BATCHES = [1, 2, 4, 8, 16]

# Below this, gradient noise and BatchNorm instability make training impractical
# regardless of whether it technically fits.
MIN_USABLE_BATCH = 4


def _grad_tensors(obj: object) -> list[torch.Tensor]:
    """Recursively collect gradient-carrying tensors from an arbitrary output.

    Detection models return tensors, lists, tuples, or dicts depending on variant
    and train/eval mode, so the probe cannot assume a structure.
    """
    if isinstance(obj, torch.Tensor):
        return [obj] if obj.requires_grad and obj.is_floating_point() else []
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _grad_tensors(v)]
    if isinstance(obj, (list, tuple)):
        return [t for v in obj for t in _grad_tensors(v)]
    return []


def _reset() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def probe(
    model_name: str, imgsz: int, batch: int, amp: bool, budget_gb: float
) -> tuple[bool, float, str]:
    """Attempt one forward + backward pass. Returns (fit, peak_GB, note).

    "Fit" is decided by measured peak memory against `budget_gb`, NOT by whether
    an OOM was raised. On Windows, the NVIDIA driver's CUDA System Memory
    Fallback silently spills oversized allocations into host RAM over PCIe rather
    than failing -- verified on this machine by allocating 10 GB on an 8 GB card
    with no error. Training in that state does not crash; it just runs many times
    slower, which is a far worse failure mode than an honest OOM.
    """
    from ultralytics import YOLO

    _reset()
    try:
        model = YOLO(f"{model_name}.pt").model.cuda().train()
        # Ultralytics loads checkpoints with every parameter frozen for inference.
        # Without re-enabling grad there is no backward graph, so the probe would
        # measure forward-only memory and wildly overstate the batch size that
        # actually trains.
        model.requires_grad_(True)
        x = torch.randn(batch, 3, imgsz, imgsz, device="cuda")

        # BF16 autocast matches the intended training config (sm_89 supports it)
        # and roughly halves activation memory versus fp32.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(x)

        # The real training loss is assignment-based and unavailable here, but
        # memory is governed by the activation graph, which a surrogate scalar
        # backward exercises identically. Model outputs vary by variant and mode
        # (tensor, list, or dict), so collect every grad-carrying tensor rather
        # than assuming a shape.
        grads = _grad_tensors(out)
        if not grads:
            _reset()
            return False, 0.0, "no grad-carrying outputs (probe cannot measure backward)"
        loss = sum(t.float().square().mean() for t in grads)
        loss.backward()

        peak = torch.cuda.max_memory_allocated() / 1024**3
        del model, x, out, loss
        _reset()
        if peak > budget_gb:
            return False, peak, f"{peak:.1f}GB SPILLED"
        return True, peak, ""

    except torch.cuda.OutOfMemoryError:
        _reset()
        return False, 0.0, "OOM"
    except Exception as exc:  # noqa: BLE001 - surfacing any probe failure is the point
        _reset()
        return False, 0.0, f"{type(exc).__name__}: {str(exc)[:60]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--no-amp", action="store_true", help="disable BF16 autocast")
    ap.add_argument(
        "--headroom",
        type=float,
        default=0.85,
        help="fraction of VRAM treated as usable (default: 0.85)",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device -- this probe measures training memory")

    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # Headroom for fragmentation, the CUDA context, and the display driver. A run
    # that peaks above this either spills to host RAM or OOMs mid-epoch.
    budget = total * args.headroom

    print(f"device: {torch.cuda.get_device_name(0)} ({total:.1f} GB)")
    print(f"usable budget: {budget:.1f} GB ({args.headroom:.0%} of VRAM)")
    print(f"imgsz: {args.imgsz} | amp: {'off' if args.no_amp else 'bf16'}\n")

    results: dict[str, int] = {}

    for name in args.models.split(","):
        name = name.strip()
        max_fit = 0
        row = []
        for b in BATCHES:
            fit, peak, note = probe(name, args.imgsz, b, amp=not args.no_amp, budget_gb=budget)
            if fit:
                max_fit = b
                row.append(f"b{b}={peak:.1f}GB")
            else:
                row.append(f"b{b}={note}")
                break  # larger batches cannot fit if this one did not
        results[name] = max_fit
        print(f"{name:>9}: max batch {max_fit:>2}   [{'  '.join(row)}]")

    print("\n--- recommendation ---")
    usable = {m: b for m, b in results.items() if b >= MIN_USABLE_BATCH}
    if usable:
        best = max(usable, key=lambda m: (DEFAULT_MODELS.index(m) if m in DEFAULT_MODELS else 0))
        print(f"largest teacher trainable at batch >= {MIN_USABLE_BATCH}: {best} "
              f"(batch {usable[best]})")
    else:
        print(f"nothing fits at batch >= {MIN_USABLE_BATCH}.")

    print(
        "\nIf the desired teacher is one tier too large, in order of preference:\n"
        "  1. gradient checkpointing  -- ~40% less activation memory, ~30% slower\n"
        "  2. gradient accumulation   -- keeps effective batch size, no memory cost\n"
        "  3. lower training imgsz    -- memory scales ~quadratically\n"
        "  4. rent a cloud GPU for the teacher only; distillation is one-off"
    )


if __name__ == "__main__":
    main()
