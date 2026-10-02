import pytest
import torch

from tokserve.generation.types import GenerationConfig
from tokserve.scheduler.request import RequestState, ScheduledRequest


def create_request(
    request_id: str = "request",
    *,
    max_new_tokens: int = 3,
    eos_token_id: int | None = None,
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id,
        torch.tensor([[1, 2]], dtype=torch.int64),
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        ),
    )


def test_newly_running_request_requires_prefill() -> None:
    request = create_request()
    request.mark_running()

    assert request.state is RequestState.RUNNING
    assert request.prefill_complete is False
    assert request.decode_ready is False
    assert request.sequence_length == 2
    assert request.num_generated_tokens == 0


def test_prefill_and_first_token_make_request_decode_ready() -> None:
    request = create_request()
    request.mark_running()
    request.mark_prefill_complete()
    request.append_generated_token(torch.tensor([[3]], dtype=torch.int64))

    assert request.prefill_complete is True
    assert request.decode_ready is True
    assert request.sequence_length == 3
    assert request.num_generated_tokens == 1
    torch.testing.assert_close(
        request.next_input_token,
        torch.tensor([[3]], dtype=torch.int64),
    )


def test_max_new_tokens_sets_completion_reason() -> None:
    request = create_request(max_new_tokens=2)
    request.mark_running()
    request.mark_prefill_complete()
    request.append_generated_token(torch.tensor([[3]], dtype=torch.int64))
    request.append_generated_token(torch.tensor([[4]], dtype=torch.int64))

    assert request.completion_reason == "max_new_tokens"
    assert request.decode_ready is False

    result = request.build_result()

    torch.testing.assert_close(
        result.token_ids,
        torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
    )
    assert result.stop_reason == "max_new_tokens"


def test_eos_sets_completion_reason_early() -> None:
    request = create_request(max_new_tokens=5, eos_token_id=3)
    request.mark_running()
    request.mark_prefill_complete()
    request.append_generated_token(torch.tensor([[3]], dtype=torch.int64))

    assert request.completion_reason == "eos"
    assert request.num_generated_tokens == 1
    assert request.decode_ready is False
    assert request.build_result().stop_reason == "eos"


def test_zero_token_request_completes_after_prefill_transition() -> None:
    request = create_request(max_new_tokens=0)
    request.mark_running()
    request.mark_prefill_complete()

    assert request.completion_reason == "max_new_tokens"
    assert request.generated_token_ids.shape == (1, 0)
    assert request.build_result().stop_reason == "max_new_tokens"


def test_multiple_requests_keep_independent_progress() -> None:
    first = create_request("A", max_new_tokens=3)
    second = create_request("B", max_new_tokens=3)

    first.mark_running()
    second.mark_running()
    first.mark_prefill_complete()
    second.mark_prefill_complete()
    first.append_generated_token(torch.tensor([[5]], dtype=torch.int64))
    second.append_generated_token(torch.tensor([[7]], dtype=torch.int64))

    torch.testing.assert_close(
        first.generated_token_ids,
        torch.tensor([[5]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        second.generated_token_ids,
        torch.tensor([[7]], dtype=torch.int64),
    )


def test_invalid_iterative_transitions_are_rejected() -> None:
    request = create_request()
    request.mark_running()

    with pytest.raises(
        RuntimeError,
        match="prefill must complete",
    ):
        request.append_generated_token(torch.tensor([[3]], dtype=torch.int64))

    request.mark_prefill_complete()

    with pytest.raises(RuntimeError, match="already complete"):
        request.mark_prefill_complete()

    with pytest.raises(RuntimeError, match="not decode-ready"):
        create_request().next_input_token
