"""Execute planned prefill and decode work."""

from collections.abc import Sequence

from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.inference import (
    decode_paged,
    prefill_paged,
)
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.generation.sampling import select_next_token
from tokserve.metrics.request_metrics import (
    RequestMetricsCollector,
)
from tokserve.scheduler.request import (
    RequestState,
    ScheduledRequest,
)


def execute_prefill(
    model: LlamaModel,
    manager: PagedKVCacheManager,
    requests: Sequence[ScheduledRequest],
    *,
    metrics: RequestMetricsCollector | None = None,
) -> tuple[str, ...]:
    """Prefill requests and select their first output token."""

    _validate_prefill_requests(requests)
    completed_ids: list[str] = []

    for request in requests:
        if request.generation_config.max_new_tokens == 0:
            request.mark_prefill_complete()
            completed_ids.append(request.request_id)
            continue

        if metrics is not None:
            metrics.prefill_started(request.request_id)

        result = prefill_paged(
            model,
            request.input_ids,
            manager,
            sequence_id=request.request_id,
        )

        if metrics is not None:
            metrics.prefill_finished(request.request_id)

        request.mark_prefill_complete()
        next_token = select_next_token(
            result.logits[:, -1, :],
            request.generation_config,
            generator=request.sampling_generator,
        )
        request.append_generated_token(next_token)

        if metrics is not None:
            metrics.token_produced(request.request_id)

        if request.completion_reason is not None:
            completed_ids.append(request.request_id)

    return tuple(completed_ids)


def execute_decode(
    model: LlamaModel,
    manager: PagedKVCacheManager,
    requests: Sequence[ScheduledRequest],
    *,
    metrics: RequestMetricsCollector | None = None,
) -> tuple[str, ...]:
    """Advance decode-ready requests by one token step."""

    _validate_decode_requests(requests)
    completed_ids: list[str] = []

    for request in requests:
        cache = manager.get_sequence(request.request_id)
        logits = decode_paged(
            model,
            request.next_input_token,
            cache,
        )
        next_token = select_next_token(
            logits[:, -1, :],
            request.generation_config,
            generator=request.sampling_generator,
        )
        request.append_generated_token(next_token)

        if metrics is not None:
            metrics.token_produced(request.request_id)

        if request.completion_reason is not None:
            completed_ids.append(request.request_id)

    return tuple(completed_ids)


def _validate_decode_requests(
    requests: Sequence[ScheduledRequest],
) -> None:
    """Validate a decode work group before mutating it."""

    seen_ids: set[str] = set()

    for request in requests:
        if request.request_id in seen_ids:
            raise ValueError("decode batch contains a duplicate request_id")

        seen_ids.add(request.request_id)

        if request.state is not RequestState.RUNNING:
            raise ValueError("decode requests must be running")

        if not request.decode_ready:
            raise ValueError("decode requests must be decode-ready")


def _validate_prefill_requests(
    requests: Sequence[ScheduledRequest],
) -> None:
    """Validate a prefill work group before mutating it."""

    seen_ids: set[str] = set()

    for request in requests:
        if request.request_id in seen_ids:
            raise ValueError("prefill batch contains a duplicate request_id")

        seen_ids.add(request.request_id)

        if request.state is not RequestState.RUNNING:
            raise ValueError("prefill requests must be running")

        if request.prefill_complete:
            raise ValueError("request prefill is already complete")

        if request.completion_reason is not None:
            raise ValueError("completed request cannot be prefetched")
