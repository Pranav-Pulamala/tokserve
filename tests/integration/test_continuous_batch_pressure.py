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
    torch.manual_seed(124)
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


def create_manager(
    model: LlamaModel,
    *,
    num_blocks: int,
) -> PagedKVCacheManager:
    parameter = model.embed_tokens.weight
    return PagedKVCacheManager(
        num_layers=model.config.num_hidden_layers,
        num_blocks=num_blocks,
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
    max_new_tokens: int,
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id,
        prompt,
        GenerationConfig(max_new_tokens=max_new_tokens),
    )


def test_arrivals_preserve_fifo_across_iteration_boundaries() -> None:
    model = create_model()
    manager = create_manager(model, num_blocks=16)
    scheduler = RequestScheduler(
        max_running_requests=2,
        cache_manager=manager,
    )
    first = create_request(
        "A",
        torch.tensor([[1, 2]], dtype=torch.int64),
        max_new_tokens=4,
    )
    second = create_request(
        "B",
        torch.tensor([[3]], dtype=torch.int64),
        max_new_tokens=2,
    )
    third = create_request(
        "C",
        torch.tensor([[4, 5, 6]], dtype=torch.int64),
        max_new_tokens=2,
    )
    fourth = create_request(
        "D",
        torch.tensor([[7]], dtype=torch.int64),
        max_new_tokens=1,
    )

    scheduler.submit(first)
    scheduler.submit(second)
    engine = ContinuousBatchEngine(model, scheduler, manager)

    iteration_zero = engine.step()

    assert iteration_zero.admitted_ids == ("A", "B")
    assert iteration_zero.prefill_ids == ("A", "B")

    scheduler.submit(third)
    iteration_one = engine.step()

    assert iteration_one.admitted_ids == ()
    assert iteration_one.decode_ids == ("A", "B")
    assert iteration_one.completed_ids == ("B",)
    assert third.state is RequestState.WAITING

    scheduler.submit(fourth)
    iteration_two = engine.step()

    assert iteration_two.admitted_ids == ("C",)
    assert iteration_two.prefill_ids == ("C",)
    assert iteration_two.decode_ids == ("A",)
    assert fourth.state is RequestState.WAITING

    iteration_three = engine.step()

    assert iteration_three.decode_ids == ("A", "C")
    assert iteration_three.completed_ids == ("A", "C")
    assert fourth.state is RequestState.WAITING

    iteration_four = engine.step()

    assert iteration_four.admitted_ids == ("D",)
    assert iteration_four.prefill_ids == ("D",)
    assert iteration_four.completed_ids == ("D",)

    assert scheduler.waiting_count == 0
    assert scheduler.running_count == 0
    assert all(
        request.state is RequestState.FINISHED
        for request in (first, second, third, fourth)
    )


def test_cache_pressure_keeps_request_waiting_until_capacity_returns() -> None:
    model = create_model()
    manager = create_manager(model, num_blocks=3)
    scheduler = RequestScheduler(
        max_running_requests=2,
        cache_manager=manager,
    )
    first_prompt = torch.tensor([[1, 2]], dtype=torch.int64)
    second_prompt = torch.tensor([[3]], dtype=torch.int64)
    first = create_request(
        "A",
        first_prompt,
        max_new_tokens=3,
    )
    second = create_request(
        "B",
        second_prompt,
        max_new_tokens=1,
    )
    scheduler.submit(first)
    scheduler.submit(second)
    engine = ContinuousBatchEngine(model, scheduler, manager)

    iteration_zero = engine.step()

    assert iteration_zero.admitted_ids == ("A",)
    assert iteration_zero.prefill_ids == ("A",)
    assert second.state is RequestState.WAITING
    assert scheduler.waiting_count == 1

    iteration_one = engine.step()

    assert iteration_one.admitted_ids == ()
    assert iteration_one.decode_ids == ("A",)
    assert second.state is RequestState.WAITING

    iteration_two = engine.step()

    assert iteration_two.decode_ids == ("A",)
    assert iteration_two.completed_ids == ("A",)
    assert second.state is RequestState.WAITING
    assert manager.allocator.allocated_count == 0

    iteration_three = engine.step()

    assert iteration_three.admitted_ids == ("B",)
    assert iteration_three.prefill_ids == ("B",)
    assert iteration_three.completed_ids == ("B",)
    assert second.state is RequestState.FINISHED

    expected_first = generate(
        model,
        first_prompt,
        GenerationConfig(max_new_tokens=3),
    )
    expected_second = generate(
        model,
        second_prompt,
        GenerationConfig(max_new_tokens=1),
    )

    assert first.result is not None
    assert second.result is not None
    torch.testing.assert_close(
        first.result.token_ids,
        expected_first.token_ids,
    )
    torch.testing.assert_close(
        second.result.token_ids,
        expected_second.token_ids,
    )

    assert scheduler.waiting_count == 0
    assert scheduler.running_count == 0
    assert manager.active_sequence_ids == ()
    assert manager.allocator.allocated_count == 0
    assert manager.allocator.free_count == 3


def test_waiting_order_is_unchanged_during_resource_pressure() -> None:
    model = create_model()
    manager = create_manager(model, num_blocks=3)
    scheduler = RequestScheduler(
        max_running_requests=3,
        cache_manager=manager,
    )
    first = create_request(
        "A",
        torch.tensor([[1, 2]], dtype=torch.int64),
        max_new_tokens=3,
    )
    second = create_request(
        "B",
        torch.tensor([[3]], dtype=torch.int64),
        max_new_tokens=1,
    )
    third = create_request(
        "C",
        torch.tensor([[4]], dtype=torch.int64),
        max_new_tokens=1,
    )

    scheduler.submit(first)
    scheduler.submit(second)
    scheduler.submit(third)
    engine = ContinuousBatchEngine(model, scheduler, manager)

    engine.step()

    assert tuple(request.request_id for request in scheduler.waiting_requests) == (
        "B",
        "C",
    )

    engine.step()

    assert tuple(request.request_id for request in scheduler.waiting_requests) == (
        "B",
        "C",
    )

    engine.step()
    next_iteration = engine.step()

    assert next_iteration.admitted_ids == ("B", "C")
    assert next_iteration.prefill_ids == ("B", "C")
    assert next_iteration.completed_ids == ("B", "C")
