"""Iteration-level prefill and decode execution primitives."""

from collections.abc import Sequence

import torch

from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.inference import decode_paged, prefill_paged
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.generation.sampling import select_next_token
from tokserve.scheduler.request import RequestState, ScheduledRequest


@torch.inference_mode()
def execute_prefill(
    model: LlamaModel,
    manager: PagedKVCacheManager,
    requests: Sequence[ScheduledRequest],
) -> tuple[str, ...]:
    """Prefill each eligible request once during one engine iteration.

    The current paged model interface owns one cache per batch-size-one
    sequence. Requests therefore share an engine iteration while retaining
    separate model calls and separate paged-cache ownership.
    """

    _validate_prefill_requests(requests)
    completed_ids: list[str] = []

    for request in requests:
        if request.generation_config.max_new_tokens == 0:
            request.mark_prefill_complete()
            completed_ids.append(request.request_id)
            continue

        prefill_result = prefill_paged(
            model,
            request.input_ids,
            manager,
            sequence_id=request.request_id,
        )
        next_token_logits = prefill_result.logits[:, -1, :]
        next_token = select_next_token(
            next_token_logits,
            request.generation_config,
            generator=request.sampling_generator,
        )

        request.mark_prefill_complete()
        request.append_generated_token(next_token)

        if request.completion_reason is not None:
            completed_ids.append(request.request_id)

    return tuple(completed_ids)


@torch.inference_mode()
def execute_decode(
    model: LlamaModel,
    manager: PagedKVCacheManager,
    requests: Sequence[ScheduledRequest],
) -> tuple[str, ...]:
    """Advance every decode-ready request by exactly one token step."""

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

        if request.completion_reason is not None:
            completed_ids.append(request.request_id)

    return tuple(completed_ids)


def _validate_decode_requests(
    requests: Sequence[ScheduledRequest],
) -> None:
    """Validate a complete decode work group before mutating it."""

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
    """Validate a complete prefill work group before mutating it."""

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
