from unittest.mock import patch

import pytest
import torch

from tokserve.batching.execution import execute_decode, execute_prefill
from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.generation.generate import generate
from tokserve.generation.types import GenerationConfig
from tokserve.reference.llama.config import LlamaConfig
from tokserve.scheduler.request import ScheduledRequest


def create_model() -> LlamaModel:
    torch.manual_seed(121)
    model = LlamaModel(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
    )
    model.eval()
    return model


def create_manager(model: LlamaModel) -> PagedKVCacheManager:
    parameter = model.embed_tokens.weight
    return PagedKVCacheManager(
        num_layers=model.config.num_hidden_layers,
        num_blocks=24,
        num_key_value_heads=model.config.num_key_value_heads,
        block_size=2,
        head_dim=model.config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )


def create_request(
    request_id: str,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int = 3,
    eos_token_id: int | None = None,
) -> ScheduledRequest:
    request = ScheduledRequest(
        request_id,
        prompt,
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        ),
    )
    request.mark_running()
    return request


def test_decode_step_matches_independent_generation() -> None:
    model = create_model()
    manager = create_manager(model)
    prompts = (
        torch.tensor([[1, 2]], dtype=torch.int64),
        torch.tensor([[3, 4, 5, 6]], dtype=torch.int64),
    )
    requests = (
        create_request("A", prompts[0]),
        create_request("B", prompts[1]),
    )

    execute_prefill(model, manager, requests)
    completed = execute_decode(model, manager, requests)

    assert completed == ()

    for request, prompt in zip(requests, prompts, strict=True):
        expected = generate(
            model,
            prompt,
            GenerationConfig(max_new_tokens=2),
        )

        torch.testing.assert_close(
            request.generated_token_ids,
            expected.generated_token_ids,
        )
        assert request.num_generated_tokens == 2
        assert request.decode_ready is True


def test_decode_updates_each_cache_independently() -> None:
    model = create_model()
    manager = create_manager(model)
    first = create_request(
        "A",
        torch.tensor([[1, 2]], dtype=torch.int64),
    )
    second = create_request(
        "B",
        torch.tensor([[3, 4, 5]], dtype=torch.int64),
    )

    execute_prefill(model, manager, (first, second))

    first_blocks_before = manager.get_sequence("A").block_ids
    second_blocks_before = manager.get_sequence("B").block_ids

    execute_decode(model, manager, (first, second))

    first_cache = manager.get_sequence("A")
    second_cache = manager.get_sequence("B")

    assert first_cache.current_length == 3
    assert second_cache.current_length == 4
    assert set(first_cache.block_ids).isdisjoint(second_cache.block_ids)
    assert set(first_blocks_before).issubset(first_cache.block_ids)
    assert set(second_blocks_before).issubset(second_cache.block_ids)


def test_one_request_can_finish_while_another_continues() -> None:
    model = create_model()
    manager = create_manager(model)
    first = create_request(
        "A",
        torch.tensor([[1]], dtype=torch.int64),
        max_new_tokens=2,
    )
    second = create_request(
        "B",
        torch.tensor([[2]], dtype=torch.int64),
        max_new_tokens=3,
    )

    execute_prefill(model, manager, (first, second))
    completed = execute_decode(model, manager, (first, second))

    assert completed == ("A",)
    assert first.completion_reason == "max_new_tokens"
    assert first.decode_ready is False
    assert second.completion_reason is None
    assert second.decode_ready is True


def test_eos_can_finish_one_request_during_decode() -> None:
    model = create_model()
    manager = create_manager(model)
    first = create_request(
        "A",
        torch.tensor([[1]], dtype=torch.int64),
        max_new_tokens=4,
        eos_token_id=0,
    )
    second = create_request(
        "B",
        torch.tensor([[2]], dtype=torch.int64),
        max_new_tokens=4,
    )
    selected_tokens = iter(
        (
            torch.tensor([[1]], dtype=torch.int64),
            torch.tensor([[1]], dtype=torch.int64),
            torch.tensor([[0]], dtype=torch.int64),
            torch.tensor([[2]], dtype=torch.int64),
        )
    )

    with patch(
        "tokserve.batching.execution.select_next_token",
        side_effect=lambda *_args, **_kwargs: next(selected_tokens),
    ):
        execute_prefill(model, manager, (first, second))
        completed = execute_decode(model, manager, (first, second))

    assert completed == ("A",)
    assert first.completion_reason == "eos"
    assert first.decode_ready is False
    assert second.completion_reason is None
    assert second.decode_ready is True


def test_adding_request_does_not_change_first_greedy_output() -> None:
    model = create_model()
    prompt = torch.tensor([[1, 2]], dtype=torch.int64)

    solo_manager = create_manager(model)
    solo = create_request("solo", prompt)
    execute_prefill(model, solo_manager, (solo,))
    execute_decode(model, solo_manager, (solo,))

    batch_manager = create_manager(model)
    batched = create_request("A", prompt)
    companion = create_request(
        "B",
        torch.tensor([[7, 8, 9]], dtype=torch.int64),
    )
    execute_prefill(model, batch_manager, (batched, companion))
    execute_decode(model, batch_manager, (batched, companion))

    torch.testing.assert_close(
        batched.generated_token_ids,
        solo.generated_token_ids,
    )


def test_invalid_decode_group_is_rejected_before_execution() -> None:
    model = create_model()
    manager = create_manager(model)
    request = create_request(
        "A",
        torch.tensor([[1]], dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="decode-ready"):
        execute_decode(model, manager, (request,))

    execute_prefill(model, manager, (request,))

    with pytest.raises(ValueError, match="duplicate request_id"):
        execute_decode(model, manager, (request, request))

    assert manager.get_sequence("A").current_length == 1
