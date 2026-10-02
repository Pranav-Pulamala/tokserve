import pytest
import torch

from tokserve.batching.execution import execute_prefill
from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.generation.generate import generate
from tokserve.generation.types import GenerationConfig
from tokserve.reference.llama.config import LlamaConfig
from tokserve.scheduler.request import ScheduledRequest


def create_model() -> LlamaModel:
    torch.manual_seed(120)
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
        num_blocks=16,
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


def test_equal_length_prefills_match_independent_generation() -> None:
    model = create_model()
    manager = create_manager(model)
    prompts = (
        torch.tensor([[1, 2]], dtype=torch.int64),
        torch.tensor([[3, 4]], dtype=torch.int64),
    )
    requests = (
        create_request("A", prompts[0]),
        create_request("B", prompts[1]),
    )

    execute_prefill(model, manager, requests)

    for request, prompt in zip(requests, prompts, strict=True):
        expected = generate(
            model,
            prompt,
            GenerationConfig(max_new_tokens=1),
        )

        torch.testing.assert_close(
            request.generated_token_ids,
            expected.generated_token_ids,
        )
        assert request.prefill_complete is True
        assert request.decode_ready is True


def test_unequal_prompts_retain_independent_cache_state() -> None:
    model = create_model()
    manager = create_manager(model)
    first = create_request(
        "A",
        torch.tensor([[1, 2]], dtype=torch.int64),
    )
    second = create_request(
        "B",
        torch.tensor([[3, 4, 5, 6]], dtype=torch.int64),
    )

    execute_prefill(model, manager, (first, second))

    first_cache = manager.get_sequence("A")
    second_cache = manager.get_sequence("B")

    assert manager.active_sequence_ids == ("A", "B")
    assert first_cache.current_length == 2
    assert second_cache.current_length == 4
    assert set(first_cache.block_ids).isdisjoint(second_cache.block_ids)
    assert first.sequence_length == 3
    assert second.sequence_length == 5


def test_max_one_token_finishes_during_prefill() -> None:
    model = create_model()
    manager = create_manager(model)
    request = create_request(
        "A",
        torch.tensor([[1, 2]], dtype=torch.int64),
        max_new_tokens=1,
    )

    completed = execute_prefill(model, manager, (request,))

    assert completed == ("A",)
    assert request.completion_reason == "max_new_tokens"
    assert request.num_generated_tokens == 1
    assert request.decode_ready is False


def test_zero_tokens_complete_without_allocating_cache() -> None:
    model = create_model()
    manager = create_manager(model)
    request = create_request(
        "zero",
        torch.tensor([[1, 2]], dtype=torch.int64),
        max_new_tokens=0,
    )

    completed = execute_prefill(model, manager, (request,))

    assert completed == ("zero",)
    assert request.completion_reason == "max_new_tokens"
    assert request.generated_token_ids.shape == (1, 0)
    assert manager.active_sequence_ids == ()
    assert manager.allocator.allocated_count == 0


def test_eos_can_finish_during_prefill() -> None:
    model = create_model()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    manager = create_manager(model)
    request = create_request(
        "eos",
        torch.tensor([[1]], dtype=torch.int64),
        max_new_tokens=4,
        eos_token_id=0,
    )

    completed = execute_prefill(model, manager, (request,))

    assert completed == ("eos",)
    assert request.completion_reason == "eos"
    torch.testing.assert_close(
        request.generated_token_ids,
        torch.tensor([[0]], dtype=torch.int64),
    )


def test_invalid_prefill_group_is_rejected_before_execution() -> None:
    model = create_model()
    manager = create_manager(model)
    waiting = ScheduledRequest(
        "waiting",
        torch.tensor([[1]], dtype=torch.int64),
        GenerationConfig(max_new_tokens=1),
    )

    with pytest.raises(ValueError, match="must be running"):
        execute_prefill(model, manager, (waiting,))

    running = create_request(
        "duplicate",
        torch.tensor([[1]], dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="duplicate request_id"):
        execute_prefill(model, manager, (running, running))

    assert manager.active_sequence_ids == ()
