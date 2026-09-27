"""Run reproducible prefill and decode profiling workloads on CUDA."""

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median

import torch

from tokserve.profiling.workloads import (
    InferenceProfilingWorkloads,
    ProfilingConfig,
    WorkloadKind,
    environment_metadata,
)

WARMUP_ITERATIONS = 5
MEASURED_ITERATIONS = 20


@dataclass(frozen=True)
class CudaTimingResult:
    """CUDA-event durations measured in milliseconds."""

    durations_ms: tuple[float, ...]

    @property
    def median_ms(self) -> float:
        """Return the median measured duration."""

        return median(self.durations_ms)


def measure_cuda(
    prepare: Callable[[], Callable[[], torch.Tensor]],
    *,
    warmup_iterations: int,
    measured_iterations: int,
) -> CudaTimingResult:
    """Measure prepared CUDA operations without timing their setup."""

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be nonnegative")

    if measured_iterations < 1:
        raise ValueError("measured_iterations must be positive")

    for _ in range(warmup_iterations):
        operation = prepare()
        operation()
        torch.cuda.synchronize()

    durations: list[float] = []

    for _ in range(measured_iterations):
        operation = prepare()
        start = torch.cuda.Event(  # type: ignore[no-untyped-call]
            enable_timing=True
        )
        end = torch.cuda.Event(  # type: ignore[no-untyped-call]
            enable_timing=True
        )

        start.record()
        operation()
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end))

    return CudaTimingResult(tuple(durations))


def report_workload(
    workloads: InferenceProfilingWorkloads,
    kind: WorkloadKind,
) -> None:
    """Measure and print one workload."""

    result = measure_cuda(
        lambda: workloads.prepare(kind),
        warmup_iterations=WARMUP_ITERATIONS,
        measured_iterations=MEASURED_ITERATIONS,
    )

    config = workloads.config
    query_length = config.prompt_length if kind == "prefill" else 1
    key_length = config.prompt_length if kind == "prefill" else config.prompt_length + 1

    print(
        f"workload={kind} "
        f"backend={config.backend} "
        f"dtype={str(config.dtype).removeprefix('torch.')} "
        f"B={config.batch_size} "
        f"Tq={query_length} "
        f"Tk={key_length} "
        f"D={config.hidden_size} "
        f"Hq={config.num_attention_heads} "
        f"Hkv={config.num_key_value_heads} "
        f"Dh={config.head_dim} "
        f"layers={config.num_hidden_layers} "
        f"warmup={WARMUP_ITERATIONS} "
        f"iterations={MEASURED_ITERATIONS} "
        f"median={result.median_ms:.6f} ms"
    )


@torch.inference_mode()
def main() -> None:
    """Run one small prefill and decode measurement."""

    if not torch.cuda.is_available():
        raise RuntimeError("This profiling command requires CUDA")

    for name, value in environment_metadata().items():
        print(f"{name}: {value}")

    config = ProfilingConfig()
    workloads = InferenceProfilingWorkloads(
        config,
        device=torch.device("cuda:0"),
    )

    print()
    report_workload(workloads, "prefill")
    report_workload(workloads, "decode")


if __name__ == "__main__":
    main()
