"""PyTorch profiler support for prepared TokServe CUDA workloads."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
)

from tokserve.profiling.workloads import (
    InferenceProfilingWorkloads,
    WorkloadKind,
)


@dataclass(frozen=True)
class ProfilerSettings:
    """Configuration for one focused profiler capture."""

    warmup_iterations: int = 5
    row_limit: int = 20
    record_shapes: bool = True
    profile_memory: bool = True

    def __post_init__(self) -> None:
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be nonnegative")

        if self.row_limit < 1:
            raise ValueError("row_limit must be positive")


def profile_workload(
    workloads: InferenceProfilingWorkloads,
    kind: WorkloadKind,
    *,
    settings: ProfilerSettings,
    trace_path: Path | None = None,
) -> str:
    """Profile one prepared CUDA operation and return a sorted summary."""

    if workloads.device.type != "cuda":
        raise ValueError("torch.profiler workload must use CUDA")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    for _ in range(settings.warmup_iterations):
        warmup_operation = workloads.prepare(kind)
        warmup_operation()
        torch.cuda.synchronize()

    operation = workloads.prepare(kind)

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=settings.record_shapes,
        profile_memory=settings.profile_memory,
        with_stack=False,
    ) as profiler:
        with record_function(f"tokserve_{kind}"):
            operation()
        torch.cuda.synchronize()

    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))

    summary: str = profiler.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=settings.row_limit,
    )
    return summary
