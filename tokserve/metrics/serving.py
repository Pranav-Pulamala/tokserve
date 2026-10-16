"""Live serving-state snapshots derived from real components."""

from dataclasses import dataclass

from tokserve.engine.paged.allocator import (
    PhysicalBlockAllocator,
)
from tokserve.metrics.aggregation import (
    ObservationMetrics,
    aggregate_request_metrics,
)
from tokserve.metrics.request_metrics import (
    RequestMetricsCollector,
)
from tokserve.scheduler.scheduler import RequestScheduler


@dataclass(frozen=True)
class CacheMetrics:
    """Live physical KV-cache block utilization."""

    capacity_blocks: int
    allocated_blocks: int
    free_blocks: int
    utilization_ratio: float

    def __post_init__(self) -> None:
        """Reject impossible cache statistics."""

        if self.capacity_blocks < 1:
            raise ValueError("cache capacity must be positive")

        if self.allocated_blocks < 0:
            raise ValueError("allocated blocks must be nonnegative")

        if self.free_blocks < 0:
            raise ValueError("free blocks must be nonnegative")

        if self.allocated_blocks + self.free_blocks != self.capacity_blocks:
            raise ValueError("allocated and free blocks must equal capacity")

        if not 0.0 <= self.utilization_ratio <= 1.0:
            raise ValueError("cache utilization must be in [0, 1]")


@dataclass(frozen=True)
class RequestCountMetrics:
    """Cumulative outcomes and live request state."""

    received_total: int
    admitted_total: int
    completed_total: int
    failed_total: int
    cancelled_total: int
    active_metrics_records: int
    waiting_current: int
    running_current: int

    def __post_init__(self) -> None:
        """Reject negative request counts."""

        values = (
            self.received_total,
            self.admitted_total,
            self.completed_total,
            self.failed_total,
            self.cancelled_total,
            self.active_metrics_records,
            self.waiting_current,
            self.running_current,
        )

        if any(value < 0 for value in values):
            raise ValueError("request counts must be nonnegative")


@dataclass(frozen=True)
class TokenMetrics:
    """Cumulative generated-output-token count."""

    generated_output_tokens_total: int

    def __post_init__(self) -> None:
        """Reject a negative token count."""

        if self.generated_output_tokens_total < 0:
            raise ValueError("generated token count must be nonnegative")


@dataclass(frozen=True)
class BatchingMetrics:
    """Continuous-batching observations."""

    iterations_total: int
    prefill_requests_total: int
    decode_requests_total: int
    last_batch_size: int
    maximum_batch_size: int

    def __post_init__(self) -> None:
        """Reject negative batching values."""

        values = (
            self.iterations_total,
            self.prefill_requests_total,
            self.decode_requests_total,
            self.last_batch_size,
            self.maximum_batch_size,
        )

        if any(value < 0 for value in values):
            raise ValueError("batching values must be nonnegative")

        if self.last_batch_size > self.maximum_batch_size:
            raise ValueError("last batch size cannot exceed maximum")


@dataclass(frozen=True)
class ServingMetricsSnapshot:
    """Public JSON-safe serving metrics snapshot."""

    schema_version: int
    duration_unit: str
    percentile_method: str
    percentile_population: str
    requests: RequestCountMetrics
    tokens: TokenMetrics
    batching: BatchingMetrics
    kv_cache: CacheMetrics
    observations: ObservationMetrics

    def __post_init__(self) -> None:
        """Validate public metadata."""

        if self.schema_version != 1:
            raise ValueError("unsupported metrics schema version")

        if self.duration_unit != "seconds":
            raise ValueError("duration_unit must be seconds")

        if self.percentile_method != "linear_interpolation":
            raise ValueError("unsupported percentile method")

        if self.percentile_population != "bounded_terminal_request_history":
            raise ValueError("unsupported percentile population")


def capture_serving_metrics(
    collector: RequestMetricsCollector,
    scheduler: RequestScheduler,
    allocator: PhysicalBlockAllocator,
) -> ServingMetricsSnapshot:
    """Capture current metrics without inference."""

    aggregate = collector.snapshot()
    capacity = allocator.num_blocks
    allocated = allocator.allocated_count
    free = allocator.free_count

    return ServingMetricsSnapshot(
        schema_version=1,
        duration_unit="seconds",
        percentile_method=("linear_interpolation"),
        percentile_population=("bounded_terminal_request_history"),
        requests=RequestCountMetrics(
            received_total=(aggregate.requests_received),
            admitted_total=(aggregate.requests_admitted),
            completed_total=(aggregate.requests_completed),
            failed_total=(aggregate.requests_failed),
            cancelled_total=(aggregate.requests_cancelled),
            active_metrics_records=(aggregate.active_requests),
            waiting_current=scheduler.waiting_count,
            running_current=scheduler.running_count,
        ),
        tokens=TokenMetrics(
            generated_output_tokens_total=(aggregate.generated_output_tokens)
        ),
        batching=BatchingMetrics(
            iterations_total=aggregate.iterations,
            prefill_requests_total=(aggregate.prefill_requests),
            decode_requests_total=(aggregate.decode_requests),
            last_batch_size=(aggregate.last_batch_size),
            maximum_batch_size=(aggregate.maximum_batch_size),
        ),
        kv_cache=CacheMetrics(
            capacity_blocks=capacity,
            allocated_blocks=allocated,
            free_blocks=free,
            utilization_ratio=allocated / capacity,
        ),
        observations=aggregate_request_metrics(aggregate.completed_records),
    )
