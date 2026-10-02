import pytest
import torch

from tokserve.batching.plan import build_iteration_plan
from tokserve.generation.types import GenerationConfig
from tokserve.scheduler.request import ScheduledRequest


def create_request(
    request_id: str,
    *,
    running: bool = True,
) -> ScheduledRequest:
    request = ScheduledRequest(
        request_id,
        torch.tensor([[1, 2]], dtype=torch.int64),
        GenerationConfig(max_new_tokens=3),
    )

    if running:
        request.mark_running()

    return request


def make_decode_ready(request: ScheduledRequest, token_id: int) -> None:
    request.mark_prefill_complete()
    request.append_generated_token(torch.tensor([[token_id]], dtype=torch.int64))


def test_empty_plan() -> None:
    plan = build_iteration_plan(())

    assert plan.prefill_requests == ()
    assert plan.decode_requests == ()
    assert plan.request_ids == ()
    assert plan.is_empty is True


def test_all_prefill_requests_preserve_order() -> None:
    requests = (
        create_request("A"),
        create_request("B"),
        create_request("C"),
    )

    plan = build_iteration_plan(requests)

    assert tuple(request.request_id for request in plan.prefill_requests) == (
        "A",
        "B",
        "C",
    )
    assert plan.decode_requests == ()
    assert plan.is_empty is False


def test_all_decode_requests_preserve_order() -> None:
    first = create_request("A")
    second = create_request("B")
    make_decode_ready(first, 3)
    make_decode_ready(second, 4)

    plan = build_iteration_plan((first, second))

    assert plan.prefill_requests == ()
    assert tuple(request.request_id for request in plan.decode_requests) == ("A", "B")


def test_mixed_prefill_and_decode_are_separated() -> None:
    prefill = create_request("prefill")
    decode = create_request("decode")
    make_decode_ready(decode, 3)

    plan = build_iteration_plan((prefill, decode))

    assert tuple(request.request_id for request in plan.prefill_requests) == (
        "prefill",
    )
    assert tuple(request.request_id for request in plan.decode_requests) == ("decode",)
    assert plan.request_ids == ("prefill", "decode")


def test_waiting_finished_and_cancelled_requests_are_excluded() -> None:
    waiting = create_request("waiting", running=False)

    finished = create_request("finished")
    finished.mark_prefill_complete()
    finished.append_generated_token(torch.tensor([[3]], dtype=torch.int64))
    finished.mark_finished()

    cancelled = create_request("cancelled")
    cancelled.cancel()

    plan = build_iteration_plan((waiting, finished, cancelled))

    assert plan.is_empty is True


def test_request_at_completion_condition_is_excluded() -> None:
    request = ScheduledRequest(
        "complete",
        torch.tensor([[1]], dtype=torch.int64),
        GenerationConfig(max_new_tokens=1),
    )
    request.mark_running()
    request.mark_prefill_complete()
    request.append_generated_token(torch.tensor([[2]], dtype=torch.int64))

    plan = build_iteration_plan((request,))

    assert request.completion_reason == "max_new_tokens"
    assert plan.is_empty is True


def test_duplicate_request_is_rejected() -> None:
    request = create_request("A")

    with pytest.raises(ValueError, match="duplicate request_id"):
        build_iteration_plan((request, request))
