"""Bounded latency and throughput aggregation."""

from dataclasses import dataclass
from math import ceil, floor
from typing import TypeAlias

from tokserve.metrics.request_metrics import RequestMetrics

Samples: TypeAlias = tuple[float, ...]


def percentile(
    samples: Samples,
    quantile: float,
) -> float | None:
    """Return a linearly interpolated percentile."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")

    if not samples:
        return None

    ordered = sorted(samples)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * quantile
    lower_index = floor(position)
    upper_index = ceil(position)

    if lower_index == upper_index:
        return ordered[lower_index]

    weight = position - lower_index
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return lower + (upper - lower) * weight


@dataclass(frozen=True)
class DistributionSummary:
    """Percentiles for one bounded sample population."""

    sample_count: int
    p50_seconds: float | None
    p95_seconds: float | None
    p99_seconds: float | None

    def __post_init__(self) -> None:
        """Reject impossible distribution summaries."""

        if self.sample_count < 0:
            raise ValueError("sample_count must be nonnegative")

        values = (
            self.p50_seconds,
            self.p95_seconds,
            self.p99_seconds,
        )

        if any(value is not None and value < 0.0 for value in values):
            raise ValueError("percentiles must be nonnegative")

        if self.sample_count == 0 and any(value is not None for value in values):
            raise ValueError("empty distributions cannot have percentiles")


@dataclass(frozen=True)
class ObservationMetrics:
    """Aggregates over the retained terminal-request window."""

    request_latency: DistributionSummary
    ttft: DistributionSummary
    queue_delay: DistributionSummary
    prefill_duration: DistributionSummary
    inter_token_latency: DistributionSummary
    window_terminal_requests: int
    window_completed_requests: int
    window_failed_requests: int
    window_cancelled_requests: int
    window_generated_output_tokens: int
    window_duration_seconds: float | None
    aggregate_output_tokens_per_second: float | None
    success_rate: float | None

    def __post_init__(self) -> None:
        """Reject inconsistent aggregate values."""

        counts = (
            self.window_terminal_requests,
            self.window_completed_requests,
            self.window_failed_requests,
            self.window_cancelled_requests,
            self.window_generated_output_tokens,
        )

        if any(value < 0 for value in counts):
            raise ValueError("observation counts must be nonnegative")

        outcomes = (
            self.window_completed_requests
            + self.window_failed_requests
            + self.window_cancelled_requests
        )

        if outcomes != self.window_terminal_requests:
            raise ValueError("terminal outcome counts must reconcile")

        if (
            self.window_duration_seconds is not None
            and self.window_duration_seconds < 0.0
        ):
            raise ValueError("window duration must be nonnegative")

        if (
            self.aggregate_output_tokens_per_second is not None
            and self.aggregate_output_tokens_per_second < 0.0
        ):
            raise ValueError("throughput must be nonnegative")

        if self.success_rate is not None and not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("success_rate must be in [0, 1]")


def summarize(
    samples: Samples,
) -> DistributionSummary:
    """Summarize one bounded sample population."""

    return DistributionSummary(
        sample_count=len(samples),
        p50_seconds=percentile(samples, 0.50),
        p95_seconds=percentile(samples, 0.95),
        p99_seconds=percentile(samples, 0.99),
    )


def aggregate_request_metrics(
    records: tuple[RequestMetrics, ...],
) -> ObservationMetrics:
    """Aggregate retained terminal-request records."""

    request_latencies = tuple(record.request_latency_seconds for record in records)
    ttft_samples = tuple(
        ttft for record in records if (ttft := record.ttft_seconds) is not None
    )
    queue_samples = tuple(
        delay for record in records if (delay := record.queue_delay_seconds) is not None
    )
    prefill_samples = tuple(
        duration
        for record in records
        if (duration := record.prefill_duration_seconds) is not None
    )
    itl_samples = tuple(
        latency
        for record in records
        for latency in (record.inter_token_latencies_seconds)
    )

    completed = sum(record.outcome == "completed" for record in records)
    failed = sum(record.outcome == "failed" for record in records)
    cancelled = sum(record.outcome == "cancelled" for record in records)
    generated_tokens = sum(record.generated_tokens for record in records)

    window_duration: float | None = None
    throughput: float | None = None
    success_rate: float | None = None

    if records:
        window_start = min(record.request_received for record in records)
        window_end = max(record.request_finished for record in records)
        window_duration = window_end - window_start

        if window_duration > 0.0:
            throughput = generated_tokens / window_duration

        success_rate = completed / len(records)

    return ObservationMetrics(
        request_latency=summarize(request_latencies),
        ttft=summarize(ttft_samples),
        queue_delay=summarize(queue_samples),
        prefill_duration=summarize(prefill_samples),
        inter_token_latency=summarize(itl_samples),
        window_terminal_requests=len(records),
        window_completed_requests=completed,
        window_failed_requests=failed,
        window_cancelled_requests=cancelled,
        window_generated_output_tokens=(generated_tokens),
        window_duration_seconds=window_duration,
        aggregate_output_tokens_per_second=(throughput),
        success_rate=success_rate,
    )
