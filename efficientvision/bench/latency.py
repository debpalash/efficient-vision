"""CPU latency benchmarking for ONNX models.

The whole project is judged against "faster than YOLO26n on CPU", so this
harness is load-bearing: if its numbers drift, every architecture decision
downstream is built on sand.

Two things make naive benchmarking unreliable on the machines we care about:

1.  Thermal throttling. A laptop CPU under sustained load clocks down partway
    through a run, so the mean silently blends two different hardware states.
    We detect this by comparing the first and last quartile of samples.
2.  Background load. A single descheduled iteration produces an outlier that
    wrecks the mean. We report percentiles and treat p50 as the headline.

Reported latency is therefore p50, not mean, and any run with detected drift
is flagged rather than quietly returned.
"""

from __future__ import annotations

import platform
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Fraction of drift between first and last quartile above which we consider the
# run thermally contaminated. 5% is comfortably above measurement noise on the
# machines tested and well below the 15-30% drop a throttling laptop shows.
THROTTLE_THRESHOLD = 0.05


@dataclass
class LatencyResult:
    """Outcome of one benchmark run. All latencies in milliseconds."""

    name: str
    p50: float
    p90: float
    p99: float
    mean: float
    stdev: float
    iterations: int
    threads: int
    input_shape: tuple[int, ...]
    model_mb: float
    provider: str
    throttle_drift: float
    warnings: list[str] = field(default_factory=list)

    @property
    def fps(self) -> float:
        """Throughput implied by median latency at this batch size."""
        batch = self.input_shape[0] if self.input_shape else 1
        return (1000.0 / self.p50) * batch if self.p50 > 0 else 0.0

    @property
    def trustworthy(self) -> bool:
        return not self.warnings

    def summary(self) -> str:
        head = f"{self.name}: {self.p50:.2f} ms (p50) | {self.fps:.1f} FPS"
        detail = (
            f"  p90 {self.p90:.2f} | p99 {self.p99:.2f} | "
            f"mean {self.mean:.2f} +/- {self.stdev:.2f}\n"
            f"  {self.iterations} iters | {self.threads} threads | "
            f"{self.model_mb:.2f} MB | {self.provider}"
        )
        warn = "".join(f"\n  ! {w}" for w in self.warnings)
        return f"{head}\n{detail}{warn}"


def _detect_throttle(samples: list[float]) -> float:
    """Relative slowdown from the first quartile of samples to the last.

    Returns a positive fraction when the run got slower over time, which is the
    signature of thermal throttling. Near-zero means stable clocks.
    """
    if len(samples) < 8:
        return 0.0
    q = len(samples) // 4
    early = statistics.median(samples[:q])
    late = statistics.median(samples[-q:])
    return (late - early) / early if early > 0 else 0.0


def benchmark_onnx(
    model_path: str | Path,
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
    iterations: int = 200,
    warmup: int = 30,
    threads: int | None = None,
    name: str | None = None,
) -> LatencyResult:
    """Measure single-stream CPU latency of an ONNX model.

    Args:
        model_path: Path to the .onnx file.
        input_shape: Input tensor shape. Batch 1 is the real-time target.
        iterations: Timed iterations. 200 gives a stable p99 without cooking
            the CPU long enough to throttle on most machines.
        warmup: Untimed iterations first. ONNX Runtime allocates arenas and
            picks kernels on early calls; timing those measures setup, not
            inference.
        threads: Intra-op threads. None lets ORT choose (usually physical core
            count). Pin this when comparing models, or thread-count differences
            masquerade as architecture differences.
        name: Label for reporting. Defaults to the filename stem.

    Returns:
        LatencyResult, with `warnings` populated if the numbers are suspect.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        opts.intra_op_num_threads = threads
        # Single-stream latency has no inter-op parallelism to exploit; leaving
        # it unpinned lets ORT spawn threads that contend with the intra-op pool.
        opts.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )

    inp = session.get_inputs()[0]
    # A model exported with a dynamic batch axis reports None/str for that dim;
    # we feed the caller's requested shape, but a static mismatch is a hard error
    # worth surfacing here rather than as an opaque ORT failure.
    static_dims = [(i, d) for i, d in enumerate(inp.shape) if isinstance(d, int)]
    for i, d in static_dims:
        if i < len(input_shape) and input_shape[i] != d:
            raise ValueError(
                f"{model_path.name} expects {inp.shape} on '{inp.name}' but "
                f"benchmark requested {input_shape} (dim {i}: {d} != {input_shape[i]})"
            )

    # Random rather than zeros: some kernels and most INT8 paths take
    # data-dependent branches, and an all-zero tensor can be unrepresentatively
    # fast.
    rng = np.random.default_rng(0)
    data = rng.random(input_shape, dtype=np.float32)
    feed = {inp.name: data}

    for _ in range(warmup):
        session.run(None, feed)

    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, feed)
        samples.append((time.perf_counter() - t0) * 1000.0)

    ordered = sorted(samples)
    drift = _detect_throttle(samples)

    warnings: list[str] = []
    if drift > THROTTLE_THRESHOLD:
        warnings.append(
            f"latency drifted +{drift * 100:.1f}% during the run "
            f"(thermal throttling or background load) -- numbers are not "
            f"comparable across runs; let the machine cool and re-measure"
        )

    p50 = ordered[len(ordered) // 2]
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    if p50 > 0 and stdev / p50 > 0.20:
        warnings.append(
            f"high variance (stdev {stdev / p50 * 100:.0f}% of p50) -- "
            f"close background applications and re-measure"
        )

    return LatencyResult(
        name=name or model_path.stem,
        p50=p50,
        p90=ordered[int(len(ordered) * 0.90)],
        p99=ordered[int(len(ordered) * 0.99)],
        mean=statistics.fmean(samples),
        stdev=stdev,
        iterations=iterations,
        threads=threads or session.get_session_options().intra_op_num_threads or 0,
        input_shape=input_shape,
        model_mb=model_path.stat().st_size / 1024 / 1024,
        provider="CPUExecutionProvider",
        throttle_drift=drift,
        warnings=warnings,
    )


def host_info() -> dict[str, str]:
    """Machine identity, so a result table can be traced to the box that made it.

    Latency numbers are meaningless without this -- YOLO26n's published 38.9 ms
    is an Intel Xeon @ 2.00 GHz figure and will not reproduce elsewhere.
    """
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
    }
