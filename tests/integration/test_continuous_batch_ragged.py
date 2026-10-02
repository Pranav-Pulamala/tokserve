from unittest.mock import patch

import torch

from tokserve.batching.engine import ContinuousBatchEngine
from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.generation.generate import generate
from tokserve.generation.types import GenerationConfig
from tokserve.reference.llama.config import LlamaConfig
from tokserve.scheduler.request import RequestState, ScheduledRequest
from tokserve.scheduler.scheduler import RequestScheduler


def create_model() -> LlamaModel:
    torch.manual_seed(123)
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
        num_blocks=8,
        num_key_value_heads=model.config.num_key_value_heads,
        block_size=4,
        head_dim=model.config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )


def create_request(
    request_id: str,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id,
        prompt,
        GenerationConfig(max_new_tokens=max_new_tokens),
    )


def test_ragged_requests_match_independent_generation_and_reuse_blocks() -> None:
    model = create_model()
    manager = create_manager(model)
    scheduler = RequestScheduler(
        max_running_requests=3,
        cache_manager=manager,
    )
    specifications = (
        (
            "A",
            torch.tensor([[1, 2, 3]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=3),
        ),
        (
            "B",
            torch.tensor([[4, 5, 6, 7]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=2),
        ),
        (
            "C",
            torch.tensor(
                [[8, 9, 10, 11, 12, 13, 14]],
                dtype=torch.int64,
            ),
            GenerationConfig(max_new_tokens=1),
        ),
        (
            "D",
            torch.tensor([[15]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=2),
        ),
    )
    expected = {
        request_id: generate(model, prompt, config)
        for request_id, prompt, config in specifications
    }

    for request_id, prompt, config in specifications:
        scheduler.submit(ScheduledRequest(request_id, prompt, config))

    released_blocks: dict[str, tuple[int, ...]] = {}
    original_release = manager.release_sequence

    def record_release(sequence_id: str) -> None:
        released_blocks[sequence_id] = manager.get_sequence(sequence_id).block_ids
        original_release(sequence_id)

    engine = ContinuousBatchEngine(model, scheduler, manager)

    with patch.object(
        manager,
        "release_sequence",
        side_effect=record_release,
    ):
        first = engine.step()

        assert first.prefill_ids == ("A", "B", "C")
        assert first.completed_ids == ("C",)
        assert scheduler.get_request("D").state is RequestState.WAITING
        assert released_blocks["C"] == (2, 3)

        first_cache = manager.get_sequence("A")
        second_cache = manager.get_sequence("B")

        assert first_cache.current_length == 3
        assert second_cache.current_length == 4
        assert first_cache.block_ids == (0,)
        assert second_cache.block_ids == (1,)
        assert set(first_cache.block_ids).isdisjoint(second_cache.block_ids)

        second = engine.step()

        assert second.admitted_ids == ("D",)
        assert second.prefill_ids == ("D",)
        assert second.decode_ids == ("A", "B")
        assert second.completed_ids == ("B",)

        fourth_cache = manager.get_sequence("D")

        assert fourth_cache.block_ids == (2,)
        assert 2 in released_blocks["C"]
        assert len(released_blocks["B"]) == 2

        third = engine.step()

        assert third.prefill_ids == ()
        assert third.decode_ids == ("A", "D")
        assert third.completed_ids == ("A", "D")
        assert third.active_ids == ()

    for request_id, _prompt, _config in specifications:
        request = scheduler.get_request(request_id)
        result = request.result

        assert request.state is RequestState.FINISHED
        assert result is not None
        torch.testing.assert_close(
            result.token_ids,
            expected[request_id].token_ids,
        )
        torch.testing.assert_close(
            result.generated_token_ids,
            expected[request_id].generated_token_ids,
        )

    assert released_blocks["C"] == (2, 3)
    assert released_blocks["D"] == (2,)
    assert len(released_blocks["A"]) == 2
    assert len(released_blocks["B"]) == 2
    assert manager.active_sequence_ids == ()
    assert manager.allocator.allocated_count == 0
    assert manager.allocator.free_count == manager.allocator.num_blocks


def test_ragged_iteration_contains_prefill_and_decode_without_overlap() -> None:
    model = create_model()
    manager = create_manager(model)
    scheduler = RequestScheduler(
        max_running_requests=2,
        cache_manager=manager,
    )
    scheduler.submit(
        create_request(
            "short",
            torch.tensor([[1]], dtype=torch.int64),
            max_new_tokens=3,
        )
    )
    scheduler.submit(
        create_request(
            "early",
            torch.tensor([[2, 3, 4]], dtype=torch.int64),
            max_new_tokens=1,
        )
    )
    scheduler.submit(
        create_request(
            "late",
            torch.tensor([[5, 6]], dtype=torch.int64),
            max_new_tokens=2,
        )
    )
    engine = ContinuousBatchEngine(model, scheduler, manager)

    first = engine.step()
    second = engine.step()

    assert first.prefill_ids == ("short", "early")
    assert first.decode_ids == ()
    assert first.completed_ids == ("early",)

    assert second.prefill_ids == ("late",)
    assert second.decode_ids == ("short",)
    assert set(second.prefill_ids).isdisjoint(second.decode_ids)
    assert scheduler.get_request("late").num_generated_tokens == 1
    assert scheduler.get_request("short").num_generated_tokens == 2
