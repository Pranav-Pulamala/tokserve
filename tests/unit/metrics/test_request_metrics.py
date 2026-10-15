from collections.abc import Iterable

import pytest

from tokserve.metrics.request_metrics import (
    RequestMetrics,
    RequestMetricsCollector,
)


class FakeClock:
    """Return explicit deterministic timestamps."""

    def __init__(
        self,
        timestamps: Iterable[float],
    ) -> None:
        self._timestamps = iter(timestamps)

    def __call__(self) -> float:
        return next(self._timestamps)


def test_complete_request_timings_are_deterministic() -> None:
    collector = RequestMetricsCollector(
        clock=FakeClock(
            [
                10.0,
                11.0,
                12.0,
                14.0,
                15.0,
                17.0,
                20.0,
            ]
        )
    )

    collector.request_received("request-1")
    collector.request_admitted("request-1")
    collector.prefill_started("request-1")
    collector.prefill_finished("request-1")
    collector.token_produced("request-1")
    collector.token_produced("request-1")
    record = collector.complete_request(
        "request-1",
        "max_new_tokens",
    )

    assert record.queue_delay_seconds == 1.0
    assert record.prefill_duration_seconds == 2.0
    assert record.ttft_seconds == 5.0
    assert record.request_latency_seconds == 10.0
    assert record.inter_token_latencies_seconds == (2.0,)
    assert record.generated_tokens == 2
    assert record.output_tokens_per_second == 0.2
    assert record.outcome == "completed"
    assert record.completion_reason == "max_new_tokens"


def test_one_token_has_ttft_but_no_itl() -> None:
    collector = RequestMetricsCollector(
        clock=FakeClock(
            [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )
    )

    collector.request_received("A")
    collector.request_admitted("A")
    collector.prefill_started("A")
    collector.prefill_finished("A")
    collector.token_produced("A")
    record = collector.complete_request(
        "A",
        "eos",
    )

    assert record.ttft_seconds == 4.0
    assert record.inter_token_latencies_seconds == ()
    assert record.generated_tokens == 1


def test_zero_token_completion_has_no_ttft() -> None:
    collector = RequestMetricsCollector(
        clock=FakeClock(
            [
                0.0,
                1.0,
                2.0,
            ]
        )
    )

    collector.request_received("A")
    collector.request_admitted("A")
    record = collector.complete_request(
        "A",
        "max_new_tokens",
    )

    assert record.ttft_seconds is None
    assert record.inter_token_latencies_seconds == ()
    assert record.generated_tokens == 0
    assert record.output_tokens_per_second is None


def test_cancellation_preserves_observed_timing() -> None:
    collector = RequestMetricsCollector(
        clock=FakeClock(
            [
                3.0,
                5.0,
                8.0,
            ]
        )
    )

    collector.request_received("A")
    collector.request_admitted("A")
    record = collector.cancel_request("A")

    assert record.outcome == "cancelled"
    assert record.queue_delay_seconds == 2.0
    assert record.request_latency_seconds == 5.0
    assert record.ttft_seconds is None

    snapshot = collector.snapshot()

    assert snapshot.requests_cancelled == 1
    assert snapshot.terminal_requests == 1


def test_failure_is_distinct_from_cancellation() -> None:
    collector = RequestMetricsCollector(
        clock=FakeClock(
            [
                1.0,
                2.0,
            ]
        )
    )

    collector.request_received("A")
    record = collector.fail_request("A")

    assert record.outcome == "failed"

    snapshot = collector.snapshot()

    assert snapshot.requests_failed == 1
    assert snapshot.requests_cancelled == 0
    assert snapshot.requests_completed == 0


def test_repeated_terminal_event_is_idempotent() -> None:
    collector = RequestMetricsCollector(
        clock=FakeClock(
            [
                0.0,
                1.0,
            ]
        )
    )

    collector.request_received("A")
    first = collector.cancel_request("A")
    second = collector.cancel_request("A")

    assert second is first

    snapshot = collector.snapshot()

    assert snapshot.requests_cancelled == 1
    assert snapshot.terminal_requests == 1
    assert len(snapshot.completed_records) == 1


def test_history_is_bounded() -> None:
    collector = RequestMetricsCollector(
        clock=FakeClock(
            [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        ),
        history_size=2,
    )

    for request_id in ("A", "B", "C"):
        collector.request_received(request_id)
        collector.cancel_request(request_id)

    snapshot = collector.snapshot()

    assert [record.request_id for record in snapshot.completed_records] == ["B", "C"]
    assert snapshot.requests_received == 3
    assert snapshot.requests_cancelled == 3


def test_snapshot_counts_tokens_and_active_requests() -> None:
    collector = RequestMetricsCollector(
        clock=FakeClock(
            [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
            ]
        )
    )

    collector.request_received("A")
    collector.request_admitted("A")
    collector.prefill_started("A")
    collector.prefill_finished("A")
    collector.token_produced("A")

    snapshot = collector.snapshot()

    assert snapshot.requests_received == 1
    assert snapshot.requests_admitted == 1
    assert snapshot.generated_output_tokens == 1
    assert snapshot.active_requests == 1
    assert snapshot.terminal_requests == 0


def test_iteration_activity_is_aggregated() -> None:
    collector = RequestMetricsCollector()

    collector.record_iteration(
        prefill_requests=2,
        decode_requests=1,
    )
    collector.record_iteration(
        prefill_requests=0,
        decode_requests=4,
    )

    snapshot = collector.snapshot()

    assert snapshot.iterations == 2
    assert snapshot.prefill_requests == 2
    assert snapshot.decode_requests == 5
    assert snapshot.last_batch_size == 4
    assert snapshot.maximum_batch_size == 4


def test_negative_iteration_counts_are_rejected() -> None:
    collector = RequestMetricsCollector()

    with pytest.raises(
        ValueError,
        match="must be nonnegative",
    ):
        collector.record_iteration(
            prefill_requests=-1,
            decode_requests=0,
        )


def test_clock_moving_backwards_is_rejected() -> None:
    collector = RequestMetricsCollector(
        clock=FakeClock(
            [
                5.0,
                4.0,
            ]
        )
    )

    collector.request_received("A")

    with pytest.raises(
        ValueError,
        match="clock moved backwards",
    ):
        collector.request_admitted("A")


def test_invalid_timestamp_order_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="request_finished cannot precede",
    ):
        RequestMetrics(
            request_id="A",
            request_received=5.0,
            request_admitted=None,
            prefill_start=None,
            prefill_end=None,
            token_produced_times=(),
            request_finished=4.0,
            outcome="cancelled",
        )


@pytest.mark.parametrize(
    "history_size",
    [
        0,
        -1,
    ],
)
def test_history_size_must_be_positive(
    history_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="history_size must be positive",
    ):
        RequestMetricsCollector(history_size=history_size)
