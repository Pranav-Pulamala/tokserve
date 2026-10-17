"""Timing boundaries for CPU and asynchronous accelerator operations."""

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import TypeAlias

import torch

MeasuredOperation: TypeAlias = Callable[[], int]
PrepareOperation: TypeAlias = Callable[[], MeasuredOperation]


@dataclass(frozen=True)
class TimingResult:
    """Raw timing, token-count, and optional CUDA-memory samples."""

    durations_seconds: tuple[float, ...]
    processed_tokens: tuple[int, ...]
    peak_allocated_bytes: tuple[int, ...] | None
    peak_reserved_bytes: tuple[int, ...] | None

    def __post_init__(self) -> None:
        """Reject inconsistent measurement populations."""

        if not self.durations_seconds:
            raise ValueError("at least one timing sample is required")

        if len(self.processed_tokens) != len(self.durations_seconds):
            raise ValueError("token and duration sample counts must match")

        if any(duration < 0.0 for duration in self.durations_seconds):
            raise ValueError("durations must be nonnegative")

        if any(tokens < 0 for tokens in self.processed_tokens):
            raise ValueError("processed token counts must be nonnegative")

        if self.peak_allocated_bytes is not None:
            if len(self.peak_allocated_bytes) != len(self.durations_seconds):
                raise ValueError(
                    "allocated-memory and duration sample counts must match"
                )

            if any(value < 0 for value in self.peak_allocated_bytes):
                raise ValueError("allocated-memory samples must be nonnegative")

        if self.peak_reserved_bytes is not None:
            if len(self.peak_reserved_bytes) != len(self.durations_seconds):
                raise ValueError(
                    "reserved-memory and duration sample counts must match"
                )

            if any(value < 0 for value in self.peak_reserved_bytes):
                raise ValueError("reserved-memory samples must be nonnegative")

    @property
    def median_seconds(self) -> float:
        """Return the median duration."""

        return median(self.durations_seconds)

    @property
    def total_processed_tokens(self) -> int:
        """Return the total measured token count."""

        return sum(self.processed_tokens)

    @property
    def total_measured_seconds(self) -> float:
        """Return the sum of measured durations."""

        return sum(self.durations_seconds)

    @property
    def aggregate_tokens_per_second(self) -> float | None:
        """Return total tokens divided by total measured time."""

        if self.total_measured_seconds <= 0.0:
            return None

        return self.total_processed_tokens / self.total_measured_seconds

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        return {
            "durations_seconds": list(self.durations_seconds),
            "processed_tokens": list(self.processed_tokens),
            "sample_count": len(self.durations_seconds),
            "median_seconds": self.median_seconds,
            "total_processed_tokens": self.total_processed_tokens,
            "total_measured_seconds": self.total_measured_seconds,
            "aggregate_tokens_per_second": (self.aggregate_tokens_per_second),
            "peak_allocated_bytes": (
                list(self.peak_allocated_bytes)
                if self.peak_allocated_bytes is not None
                else None
            ),
            "peak_reserved_bytes": (
                list(self.peak_reserved_bytes)
                if self.peak_reserved_bytes is not None
                else None
            ),
        }


def synchronize_device(device: torch.device) -> None:
    """Wait for previously submitted accelerator work."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return

    if device.type == "mps":
        torch.mps.synchronize()


def measure_operations(
    prepare: PrepareOperation,
    *,
    device: torch.device,
    warmup_iterations: int,
    measured_repetitions: int,
    clock: Callable[[], float] = perf_counter,
) -> TimingResult:
    """Measure separately prepared operations with correct device boundaries.

    Setup performed by ``prepare`` is intentionally outside the measured
    interval. The returned operation must report the number of tokens it
    actually processed.
    """

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be nonnegative")

    if measured_repetitions < 1:
        raise ValueError("measured_repetitions must be positive")

    for _ in range(warmup_iterations):
        operation = prepare()
        synchronize_device(device)
        operation()
        synchronize_device(device)

    durations: list[float] = []
    processed_tokens: list[int] = []
    peak_allocated: list[int] | None = [] if device.type == "cuda" else None
    peak_reserved: list[int] | None = [] if device.type == "cuda" else None

    for _ in range(measured_repetitions):
        operation = prepare()
        synchronize_device(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        started = clock()
        token_count = operation()
        synchronize_device(device)
        finished = clock()

        duration = finished - started

        if duration < 0.0:
            raise ValueError("benchmark clock moved backwards")

        if token_count < 0:
            raise ValueError("measured operation returned a negative token count")

        durations.append(duration)
        processed_tokens.append(token_count)

        if peak_allocated is not None:
            peak_allocated.append(torch.cuda.max_memory_allocated(device))

        if peak_reserved is not None:
            peak_reserved.append(torch.cuda.max_memory_reserved(device))

    return TimingResult(
        durations_seconds=tuple(durations),
        processed_tokens=tuple(processed_tokens),
        peak_allocated_bytes=(
            tuple(peak_allocated) if peak_allocated is not None else None
        ),
        peak_reserved_bytes=(
            tuple(peak_reserved) if peak_reserved is not None else None
        ),
    )
