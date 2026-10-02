import pytest
import torch

from tokserve.generation.types import GenerationConfig
from tokserve.scheduler.request import RequestState, ScheduledRequest


def create_request() -> ScheduledRequest:
    return ScheduledRequest(
        "request-1",
        torch.tensor([[1, 2, 3]], dtype=torch.int64),
        GenerationConfig(max_new_tokens=2),
    )


def test_request_starts_waiting() -> None:
    request = create_request()

    assert request.request_id == "request-1"
    assert request.state is RequestState.WAITING
    assert request.generated_token_ids.shape == (1, 0)


def test_request_transitions_from_waiting_to_running_to_finished() -> None:
    request = create_request()

    request.mark_running()
    request.mark_prefill_complete()
    request.append_generated_token(torch.tensor([[4]], dtype=torch.int64))
    request.mark_finished()

    assert request.state is RequestState.FINISHED
    torch.testing.assert_close(
        request.generated_token_ids,
        torch.tensor([[4]], dtype=torch.int64),
    )


def test_invalid_transition_is_rejected() -> None:
    request = create_request()

    with pytest.raises(RuntimeError, match="must be running"):
        request.mark_finished()

    request.mark_running()
    request.mark_finished()

    with pytest.raises(RuntimeError, match="must be waiting"):
        request.mark_running()


def test_waiting_and_running_requests_can_be_cancelled() -> None:
    waiting = create_request()
    waiting.cancel()

    running = ScheduledRequest(
        "request-2",
        torch.tensor([[1]], dtype=torch.int64),
        GenerationConfig(max_new_tokens=1),
    )
    running.mark_running()
    running.cancel()

    assert waiting.state is RequestState.CANCELLED
    assert running.state is RequestState.CANCELLED


@pytest.mark.parametrize(
    ("request_id", "input_ids", "message"),
    [
        (
            "",
            torch.tensor([[1]], dtype=torch.int64),
            "request_id must be nonempty",
        ),
        (
            "bad-rank",
            torch.tensor([1], dtype=torch.int64),
            r"input_ids must have shape \(1, T\)",
        ),
        (
            "empty",
            torch.empty((1, 0), dtype=torch.int64),
            "input_ids must contain at least one token",
        ),
        (
            "float",
            torch.tensor([[1.0]], dtype=torch.float32),
            "input_ids must contain integers",
        ),
    ],
)
def test_invalid_requests_are_rejected(
    request_id: str,
    input_ids: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        ScheduledRequest(
            request_id,
            input_ids,
            GenerationConfig(max_new_tokens=1),
        )
