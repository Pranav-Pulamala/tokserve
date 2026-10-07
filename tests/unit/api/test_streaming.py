import asyncio

import pytest
import torch

from tokserve.api.streaming import (
    CancelledEvent,
    DoneEvent,
    FailureEvent,
    RequestEventBroker,
    StreamBackpressureError,
    TokenEvent,
)
from tokserve.generation.types import GenerationConfig
from tokserve.scheduler.request import ScheduledRequest


def create_running_request(
    request_id: str,
    *,
    max_new_tokens: int = 3,
) -> ScheduledRequest:
    request = ScheduledRequest(
        request_id,
        torch.tensor([[1]], dtype=torch.int64),
        GenerationConfig(max_new_tokens=max_new_tokens),
    )
    request.mark_running()
    request.mark_prefill_complete()
    return request


def receive(stream: object) -> object:
    async def run() -> object:
        return await stream.receive()  # type: ignore[attr-defined]

    return asyncio.run(run())


def test_token_is_available_while_request_is_still_running() -> None:
    broker = RequestEventBroker()
    stream = broker.open_stream("A")
    request = create_running_request("A")
    request.append_generated_token(torch.tensor([[2]], dtype=torch.int64))

    broker.publish_progress(request)
    event = receive(stream)

    assert request.result is None
    assert request.state.value == "running"
    assert event == TokenEvent(token_id=2, token_ids=(2,))


def test_token_and_completion_events_are_ordered() -> None:
    broker = RequestEventBroker()
    stream = broker.open_stream("A")
    request = create_running_request("A", max_new_tokens=2)

    request.append_generated_token(torch.tensor([[2]], dtype=torch.int64))
    broker.publish_progress(request)

    request.append_generated_token(torch.tensor([[3]], dtype=torch.int64))
    result = request.build_result()
    request.mark_finished(result)
    broker.publish_progress(request)

    assert receive(stream) == TokenEvent(
        token_id=2,
        token_ids=(2,),
    )
    assert receive(stream) == TokenEvent(
        token_id=3,
        token_ids=(2, 3),
    )
    assert receive(stream) == DoneEvent(finish_reason="max_new_tokens")


def test_publish_progress_does_not_duplicate_tokens() -> None:
    broker = RequestEventBroker()
    stream = broker.open_stream("A")
    request = create_running_request("A")
    request.append_generated_token(torch.tensor([[2]], dtype=torch.int64))

    broker.publish_progress(request)
    broker.publish_progress(request)

    assert receive(stream) == TokenEvent(
        token_id=2,
        token_ids=(2,),
    )


def test_request_streams_are_isolated() -> None:
    broker = RequestEventBroker()
    first_stream = broker.open_stream("A")
    second_stream = broker.open_stream("B")
    first = create_running_request("A")
    second = create_running_request("B")
    first.append_generated_token(torch.tensor([[2]], dtype=torch.int64))
    second.append_generated_token(torch.tensor([[7]], dtype=torch.int64))

    broker.publish_progress(first)
    broker.publish_progress(second)

    assert receive(first_stream) == TokenEvent(
        token_id=2,
        token_ids=(2,),
    )
    assert receive(second_stream) == TokenEvent(
        token_id=7,
        token_ids=(7,),
    )


def test_cancellation_event_and_cleanup() -> None:
    broker = RequestEventBroker()
    stream = broker.open_stream("A")
    request = create_running_request("A")
    request.cancel()

    broker.publish_progress(request)

    assert receive(stream) == CancelledEvent()
    assert stream.terminal_published is True

    broker.remove_stream("A")

    assert broker.active_request_ids == ()


def test_failure_event_is_sanitized() -> None:
    broker = RequestEventBroker()
    stream = broker.open_stream("A")

    broker.publish_failure("A")

    assert receive(stream) == FailureEvent(message="generation failed")


def test_bounded_buffer_rejects_unlimited_progress() -> None:
    broker = RequestEventBroker(buffer_size=1)
    broker.open_stream("A")
    request = create_running_request("A", max_new_tokens=3)
    request.append_generated_token(torch.tensor([[2]], dtype=torch.int64))
    request.append_generated_token(torch.tensor([[3]], dtype=torch.int64))

    with pytest.raises(
        StreamBackpressureError,
        match="buffer is full",
    ):
        broker.publish_progress(request)


def test_duplicate_and_unknown_streams_are_rejected() -> None:
    broker = RequestEventBroker()
    broker.open_stream("A")

    with pytest.raises(ValueError, match="already registered"):
        broker.open_stream("A")

    with pytest.raises(KeyError, match="unknown request stream"):
        broker.publish_failure("missing")


def test_nonterminal_stream_cannot_be_removed() -> None:
    broker = RequestEventBroker()
    broker.open_stream("A")

    with pytest.raises(RuntimeError, match="nonterminal"):
        broker.remove_stream("A")
