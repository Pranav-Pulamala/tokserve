"""Thread-safe request timing records using a monotonic clock."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite
from threading import Lock
from time import perf_counter
from typing import Literal, TypeAlias

from tokserve.generation.types import StopReason

RequestOutcome: TypeAlias = Literal[
    "completed",
    "failed",
    "cancelled",
]
Clock: TypeAlias = Callable[[], float]


def _duration(
    start: float | None,
    end: float | None,
) -> float | None:
    """Return a nonnegative elapsed duration when both values exist."""

    if start is None or end is None:
        return None

    elapsed = end - start

    if elapsed < 0.0:
        raise ValueError("duration cannot be negative")

    return elapsed


def _validate_timestamp(
    name: str,
    value: float | None,
) -> None:
    """Reject non-finite or negative monotonic timestamp values."""

    if value is None:
        return

    if not isfinite(value):
        raise ValueError(f"{name} must be finite")

    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class RequestMetrics:
    """Immutable timing record for one terminal request."""

    request_id: str
    request_received: float
    request_admitted: float | None
    prefill_start: float | None
    prefill_end: float | None
    token_produced_times: tuple[float, ...]
    request_finished: float
    outcome: RequestOutcome
    completion_reason: StopReason | None = None

    def __post_init__(self) -> None:
        """Validate timestamp presence and lifecycle ordering."""

        if not self.request_id:
            raise ValueError("request_id must be nonempty")

        _validate_timestamp(
            "request_received",
            self.request_received,
        )
        _validate_timestamp(
            "request_admitted",
            self.request_admitted,
        )
        _validate_timestamp(
            "prefill_start",
            self.prefill_start,
        )
        _validate_timestamp(
            "prefill_end",
            self.prefill_end,
        )
        _validate_timestamp(
            "request_finished",
            self.request_finished,
        )

        for token_timestamp in self.token_produced_times:
            _validate_timestamp(
                "token_produced_time",
                token_timestamp,
            )

        if self.outcome == "completed" and self.completion_reason is None:
            raise ValueError("completed requests require a completion_reason")

        if self.outcome != "completed" and self.completion_reason is not None:
            raise ValueError("only completed requests have a completion_reason")

        timeline: list[tuple[str, float]] = [
            ("request_received", self.request_received)
        ]

        optional_events = (
            ("request_admitted", self.request_admitted),
            ("prefill_start", self.prefill_start),
            ("prefill_end", self.prefill_end),
        )

        for event_name, event_timestamp in optional_events:
            if event_timestamp is not None:
                timeline.append((event_name, event_timestamp))

        timeline.extend(
            ("token_produced", token_timestamp)
            for token_timestamp in self.token_produced_times
        )
        timeline.append(("request_finished", self.request_finished))

        for (
            previous_name,
            previous_timestamp,
        ), (
            current_name,
            current_timestamp,
        ) in zip(timeline, timeline[1:]):
            if current_timestamp < previous_timestamp:
                raise ValueError(f"{current_name} cannot precede {previous_name}")

    @property
    def generated_tokens(self) -> int:
        """Return the number of observed generated tokens."""

        return len(self.token_produced_times)

    @property
    def queue_delay_seconds(self) -> float | None:
        """Return acceptance-to-admission queueing delay."""

        return _duration(
            self.request_received,
            self.request_admitted,
        )

    @property
    def prefill_duration_seconds(self) -> float | None:
        """Return CPU-observed prefill call duration."""

        return _duration(
            self.prefill_start,
            self.prefill_end,
        )

    @property
    def ttft_seconds(self) -> float | None:
        """Return acceptance-to-first-produced-token latency."""

        if not self.token_produced_times:
            return None

        return _duration(
            self.request_received,
            self.token_produced_times[0],
        )

    @property
    def request_latency_seconds(self) -> float:
        """Return acceptance-to-terminal server generation latency."""

        elapsed = _duration(
            self.request_received,
            self.request_finished,
        )

        if elapsed is None:
            raise RuntimeError("terminal request latency is unavailable")

        return elapsed

    @property
    def inter_token_latencies_seconds(
        self,
    ) -> tuple[float, ...]:
        """Return elapsed durations between produced tokens."""

        return tuple(
            current - previous
            for previous, current in zip(
                self.token_produced_times,
                self.token_produced_times[1:],
            )
        )

    @property
    def output_tokens_per_second(self) -> float | None:
        """Return tokens divided by total server generation latency."""

        if self.generated_tokens == 0:
            return None

        latency = self.request_latency_seconds

        if latency == 0.0:
            return None

        return self.generated_tokens / latency


@dataclass(frozen=True)
class MetricsSnapshot:
    """Immutable aggregate counter and bounded-history snapshot."""

    requests_received: int
    requests_admitted: int
    requests_completed: int
    requests_failed: int
    requests_cancelled: int
    generated_output_tokens: int
    active_requests: int
    completed_records: tuple[RequestMetrics, ...]

    def __post_init__(self) -> None:
        """Reject impossible negative aggregate values."""

        values = (
            self.requests_received,
            self.requests_admitted,
            self.requests_completed,
            self.requests_failed,
            self.requests_cancelled,
            self.generated_output_tokens,
            self.active_requests,
        )

        if any(value < 0 for value in values):
            raise ValueError("metric counters must be nonnegative")

    @property
    def terminal_requests(self) -> int:
        """Return all terminal outcomes."""

        return self.requests_completed + self.requests_failed + self.requests_cancelled


@dataclass
class _ActiveRequestMetrics:
    """Mutable timing state retained only while a request is active."""

    request_id: str
    request_received: float
    request_admitted: float | None = None
    prefill_start: float | None = None
    prefill_end: float | None = None
    token_produced_times: list[float] = field(default_factory=list)

    @property
    def latest_timestamp(self) -> float:
        """Return the latest recorded lifecycle timestamp."""

        timestamps = [self.request_received]

        if self.request_admitted is not None:
            timestamps.append(self.request_admitted)

        if self.prefill_start is not None:
            timestamps.append(self.prefill_start)

        if self.prefill_end is not None:
            timestamps.append(self.prefill_end)

        timestamps.extend(self.token_produced_times)
        return max(timestamps)


class RequestMetricsCollector:
    """Collect bounded request observations with thread-safe counters."""

    def __init__(
        self,
        *,
        clock: Clock = perf_counter,
        history_size: int = 1_024,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be positive")

        self._clock = clock
        self._history_size = history_size
        self._active: dict[str, _ActiveRequestMetrics] = {}
        self._history: deque[RequestMetrics] = deque()
        self._terminal_records: dict[str, RequestMetrics] = {}
        self._requests_received = 0
        self._requests_admitted = 0
        self._requests_completed = 0
        self._requests_failed = 0
        self._requests_cancelled = 0
        self._generated_output_tokens = 0
        self._lock = Lock()

    def request_received(self, request_id: str) -> None:
        """Start timing one accepted, validated request."""

        if not request_id:
            raise ValueError("request_id must be nonempty")

        with self._lock:
            if request_id in self._active or request_id in self._terminal_records:
                raise ValueError("request_id is already recorded")

            timestamp = self._read_clock()
            self._active[request_id] = _ActiveRequestMetrics(
                request_id=request_id,
                request_received=timestamp,
            )
            self._requests_received += 1

    def request_admitted(self, request_id: str) -> None:
        """Record scheduler admission."""

        with self._lock:
            observation = self._get_active(request_id)

            if observation.request_admitted is not None:
                raise RuntimeError("request admission is already recorded")

            observation.request_admitted = self._next_timestamp(observation)
            self._requests_admitted += 1

    def prefill_started(self, request_id: str) -> None:
        """Record the start of the model prefill call."""

        with self._lock:
            observation = self._get_active(request_id)

            if observation.request_admitted is None:
                raise RuntimeError("request must be admitted before prefill")

            if observation.prefill_start is not None:
                raise RuntimeError("prefill start is already recorded")

            observation.prefill_start = self._next_timestamp(observation)

    def prefill_finished(self, request_id: str) -> None:
        """Record completion of the model prefill call."""

        with self._lock:
            observation = self._get_active(request_id)

            if observation.prefill_start is None:
                raise RuntimeError("prefill must start before it finishes")

            if observation.prefill_end is not None:
                raise RuntimeError("prefill end is already recorded")

            observation.prefill_end = self._next_timestamp(observation)

    def token_produced(self, request_id: str) -> None:
        """Record one generated output token timestamp."""

        with self._lock:
            observation = self._get_active(request_id)

            if observation.prefill_end is None:
                raise RuntimeError("prefill must finish before token production")

            timestamp = self._next_timestamp(observation)
            observation.token_produced_times.append(timestamp)
            self._generated_output_tokens += 1

    def complete_request(
        self,
        request_id: str,
        completion_reason: StopReason,
    ) -> RequestMetrics:
        """Record one successful terminal request."""

        return self._finish_request(
            request_id,
            outcome="completed",
            completion_reason=completion_reason,
        )

    def fail_request(
        self,
        request_id: str,
    ) -> RequestMetrics:
        """Record one failed terminal request."""

        return self._finish_request(
            request_id,
            outcome="failed",
        )

    def cancel_request(
        self,
        request_id: str,
    ) -> RequestMetrics:
        """Record one cancelled terminal request."""

        return self._finish_request(
            request_id,
            outcome="cancelled",
        )

    def snapshot(self) -> MetricsSnapshot:
        """Return counters and an immutable bounded history copy."""

        with self._lock:
            return MetricsSnapshot(
                requests_received=self._requests_received,
                requests_admitted=self._requests_admitted,
                requests_completed=self._requests_completed,
                requests_failed=self._requests_failed,
                requests_cancelled=self._requests_cancelled,
                generated_output_tokens=(self._generated_output_tokens),
                active_requests=len(self._active),
                completed_records=tuple(self._history),
            )

    def _finish_request(
        self,
        request_id: str,
        *,
        outcome: RequestOutcome,
        completion_reason: StopReason | None = None,
    ) -> RequestMetrics:
        """Create one terminal record without double-counting."""

        with self._lock:
            existing = self._terminal_records.get(request_id)

            if existing is not None:
                return existing

            observation = self._get_active(request_id)
            finished = self._next_timestamp(observation)
            record = RequestMetrics(
                request_id=request_id,
                request_received=(observation.request_received),
                request_admitted=(observation.request_admitted),
                prefill_start=observation.prefill_start,
                prefill_end=observation.prefill_end,
                token_produced_times=tuple(observation.token_produced_times),
                request_finished=finished,
                outcome=outcome,
                completion_reason=completion_reason,
            )

            del self._active[request_id]

            if len(self._history) == self._history_size:
                evicted = self._history.popleft()
                self._terminal_records.pop(
                    evicted.request_id,
                    None,
                )

            self._history.append(record)
            self._terminal_records[request_id] = record

            if outcome == "completed":
                self._requests_completed += 1
            elif outcome == "failed":
                self._requests_failed += 1
            else:
                self._requests_cancelled += 1

            return record

    def _get_active(
        self,
        request_id: str,
    ) -> _ActiveRequestMetrics:
        """Return active timing state for one request."""

        try:
            return self._active[request_id]
        except KeyError as error:
            raise KeyError("unknown active request_id") from error

    def _next_timestamp(
        self,
        observation: _ActiveRequestMetrics,
    ) -> float:
        """Read and validate the next lifecycle timestamp."""

        timestamp = self._read_clock()

        if timestamp < observation.latest_timestamp:
            raise ValueError("monotonic clock moved backwards")

        return timestamp

    def _read_clock(self) -> float:
        """Read one finite, nonnegative clock value."""

        timestamp = self._clock()
        _validate_timestamp("clock value", timestamp)
        return timestamp
