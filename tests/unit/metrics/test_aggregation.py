import pytest

from tokserve.metrics.aggregation import (
    aggregate_request_metrics,
    percentile,
    summarize,
)
from tokserve.metrics.request_metrics import (
    RequestMetrics,
)


def create_record(
    request_id: str,
    *,
    received: float,
    admitted: float | None,
    prefill_start: float | None,
    prefill_end: float | None,
    token_times: tuple[float, ...],
    finished: float,
    outcome: str = "completed",
) -> RequestMetrics:
    if outcome == "completed":
        completion_reason = "max_new_tokens"
    else:
        completion_reason = None

    return RequestMetrics(
        request_id=request_id,
        request_received=received,
        request_admitted=admitted,
        prefill_start=prefill_start,
        prefill_end=prefill_end,
        token_produced_times=token_times,
        request_finished=finished,
        outcome=outcome,  # type: ignore[arg-type]
        completion_reason=completion_reason,
    )


def test_known_linear_percentiles() -> None:
    samples = (
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    )

    assert percentile(samples, 0.50) == 3.0
    assert percentile(samples, 0.95) == pytest.approx(4.8)
    assert percentile(samples, 0.99) == pytest.approx(4.96)


def test_percentile_is_independent_of_input_order() -> None:
    samples = (
        4.0,
        1.0,
        3.0,
        2.0,
    )

    assert percentile(samples, 0.50) == 2.5


def test_empty_and_single_sample_distributions() -> None:
    empty = summarize(())
    single = summarize((7.0,))

    assert empty.sample_count == 0
    assert empty.p50_seconds is None
    assert empty.p95_seconds is None
    assert empty.p99_seconds is None

    assert single.sample_count == 1
    assert single.p50_seconds == 7.0
    assert single.p95_seconds == 7.0
    assert single.p99_seconds == 7.0


def test_invalid_quantile_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="quantile must be",
    ):
        percentile((1.0,), 1.1)


def test_aggregate_samples_and_throughput() -> None:
    records = (
        create_record(
            "A",
            received=0.0,
            admitted=1.0,
            prefill_start=2.0,
            prefill_end=3.0,
            token_times=(4.0, 6.0),
            finished=8.0,
        ),
        create_record(
            "B",
            received=2.0,
            admitted=3.0,
            prefill_start=4.0,
            prefill_end=5.0,
            token_times=(7.0,),
            finished=10.0,
        ),
    )

    result = aggregate_request_metrics(records)

    assert result.request_latency.sample_count == 2
    assert result.request_latency.p50_seconds == 8.0
    assert result.ttft.sample_count == 2
    assert result.queue_delay.sample_count == 2
    assert result.prefill_duration.sample_count == 2
    assert result.inter_token_latency.sample_count == 1
    assert result.inter_token_latency.p50_seconds == 2.0
    assert result.window_terminal_requests == 2
    assert result.window_completed_requests == 2
    assert result.window_generated_output_tokens == 3
    assert result.window_duration_seconds == 10.0
    assert result.aggregate_output_tokens_per_second == 0.3
    assert result.success_rate == 1.0


def test_outcomes_reconcile_and_zero_token_ttft_is_absent() -> None:
    records = (
        create_record(
            "completed",
            received=0.0,
            admitted=1.0,
            prefill_start=None,
            prefill_end=None,
            token_times=(),
            finished=2.0,
        ),
        create_record(
            "failed",
            received=2.0,
            admitted=None,
            prefill_start=None,
            prefill_end=None,
            token_times=(),
            finished=3.0,
            outcome="failed",
        ),
        create_record(
            "cancelled",
            received=3.0,
            admitted=4.0,
            prefill_start=None,
            prefill_end=None,
            token_times=(),
            finished=5.0,
            outcome="cancelled",
        ),
    )

    result = aggregate_request_metrics(records)

    assert result.window_terminal_requests == 3
    assert result.window_completed_requests == 1
    assert result.window_failed_requests == 1
    assert result.window_cancelled_requests == 1
    assert result.request_latency.sample_count == 3
    assert result.ttft.sample_count == 0
    assert result.inter_token_latency.sample_count == 0
    assert result.success_rate == pytest.approx(1.0 / 3.0)


def test_empty_aggregate_has_unavailable_rates() -> None:
    result = aggregate_request_metrics(())

    assert result.window_terminal_requests == 0
    assert result.window_duration_seconds is None
    assert result.aggregate_output_tokens_per_second is None
    assert result.success_rate is None


def test_zero_duration_window_has_no_throughput() -> None:
    record = create_record(
        "A",
        received=1.0,
        admitted=1.0,
        prefill_start=1.0,
        prefill_end=1.0,
        token_times=(1.0,),
        finished=1.0,
    )

    result = aggregate_request_metrics((record,))

    assert result.window_duration_seconds == 0.0
    assert result.aggregate_output_tokens_per_second is None
