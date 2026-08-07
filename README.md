# efficient-vision

A CPU-first vision model for multi-class classification and oriented bounding box
(OBB) detection, targeting YOLO26n-or-better latency on commodity CPUs.

**Status: pre-training.** The benchmark harness, reparameterizable blocks, and
measured baselines exist. The model itself is not built and has no accuracy
numbers — see [ROADMAP.md](ROADMAP.md).

## Documents

| File | What it covers |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Repo rules — output format, dependency management, measurement discipline |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Design decisions and why, including the ones that run against intuition |
| [BASELINE.md](BASELINE.md) | Measured YOLO26n latency on this hardware, and thread-scaling findings |
| [ROADMAP.md](ROADMAP.md) | Phased plan to v1, hardware limits, known risks |
| [docs/yolo26-vs-efficientvision.md](docs/yolo26-vs-efficientvision.md) | Measured CPU comparison across the YOLO26 family |

## Setup

Dependencies and Python versions are managed with [uv](https://docs.astral.sh/uv/).

```sh
uv venv --python 3.11
uv pip install -e ".[baseline,dev]"
```

Torch comes from the PyTorch CUDA index, pinned in `pyproject.toml`. The PyPI
wheels are CPU-only on Windows, which silently disables GPU training — the failure
mode is a slow run rather than an error, so the pin matters.

## Scripts

```sh
# Measure the YOLO26n baseline this project must beat
uv run --no-sync python scripts/baseline.py --imgsz 640 --threads 12

# Map latency against thread count (find the knee before deploying)
uv run --no-sync python scripts/thread_sweep.py runs/baseline/yolo26n_640.onnx

# Benchmark the whole YOLO26 family to see the speed/accuracy frontier
uv run --no-sync python scripts/family_bench.py

# Find the largest distillation teacher that fits this GPU
uv run --no-sync python scripts/vram_probe.py
```

## Tests

```sh
uv run --no-sync python -m pytest tests/ -q
```

The reparameterization tests are the important ones: they assert that a fused
`RepConv` computes the same function as its multi-branch training form. If that
guarantee breaks, the model loses accuracy at export time with no error raised.

## Two hardware gotchas found on this machine

Both are silent failures — neither raises an error, and both would be misread as
"the model is slow."

1. **ONNX Runtime CPU latency collapses past ~12 threads** on the i9-13900HX
   hybrid CPU; 32 threads is slower than 1. Always pin `intra_op_num_threads`
   explicitly. See [BASELINE.md](BASELINE.md).
2. **CUDA System Memory Fallback is enabled** — a 10 GB allocation on an 8 GB card
   succeeds instead of raising `OutOfMemoryError`, spilling to host RAM over PCIe
   and training 5–20x slower. Never size a training run by "it didn't OOM"; compare
   measured peak against real VRAM. See `scripts/vram_probe.py`.
