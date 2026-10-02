import pytest
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
    torch.manual_seed(122)
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
    max_new_tokens: int,
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id,
        prompt,
        GenerationConfig(max_new_tokens=max_new_tokens),
    )


def test_batch_membership_changes_between_iterations() -> None:
    model = create_model()
    manager = create_manager(model)
    scheduler = RequestScheduler(
        max_running_requests=2,
        cache_manager=manager,
    )
    requests = (
        create_request(
            "A",
            torch.tensor([[1, 2]], dtype=torch.int64),
            max_new_tokens=3,
        ),
        create_request(
            "B",
            torch.tensor([[3, 4, 5]], dtype=torch.int64),
            max_new_tokens=1,
        ),
        create_request(
            "C",
            torch.tensor([[6]], dtype=torch.int64),
            max_new_tokens=2,
        ),
    )

    for request in requests:
        scheduler.submit(request)

    engine = ContinuousBatchEngine(model, scheduler, manager)

    first = engine.step()

    assert first.admitted_ids == ("A", "B")
    assert first.prefill_ids == ("A", "B")
    assert first.decode_ids == ()
    assert first.completed_ids == ("B",)
    assert first.active_ids == ("A",)
    assert scheduler.get_request("B").state is RequestState.FINISHED
    assert scheduler.get_request("C").state is RequestState.WAITING

    second = engine.step()

    assert second.admitted_ids == ("C",)
    assert second.prefill_ids == ("C",)
    assert second.decode_ids == ("A",)
    assert second.completed_ids == ()
    assert second.active_ids == ("A", "C")
    assert scheduler.get_request("C").num_generated_tokens == 1

    third = engine.step()

    assert third.admitted_ids == ()
    assert third.prefill_ids == ()
    assert third.decode_ids == ("A", "C")
    assert third.completed_ids == ("A", "C")
    assert third.active_ids == ()
    assert scheduler.waiting_count == 0
    assert scheduler.running_count == 0


def test_continuous_results_match_independent_generation() -> None:
    model = create_model()
    manager = create_manager(model)
    scheduler = RequestScheduler(
        max_running_requests=2,
        cache_manager=manager,
    )
    specifications = (
        (
            "A",
            torch.tensor([[1, 2]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=3),
        ),
        (
            "B",
            torch.tensor([[3, 4, 5]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=1),
        ),
        (
            "C",
            torch.tensor([[6]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=2),
        ),
    )

    expected = {
        request_id: generate(model, prompt, config)
        for request_id, prompt, config in specifications
    }

    for request_id, prompt, config in specifications:
        scheduler.submit(ScheduledRequest(request_id, prompt, config))

    engine = ContinuousBatchEngine(model, scheduler, manager)
    engine.run_until_idle()

    for request_id, _prompt, _config in specifications:
        request = scheduler.get_request(request_id)
        result = request.result

        assert request.state is RequestState.FINISHED
        assert result is not None
        torch.testing.assert_close(
            result.token_ids,
            expected[request_id].token_ids,
        )

    assert manager.active_sequence_ids == ()
    assert manager.allocator.allocated_count == 0
    assert manager.allocator.free_count == manager.allocator.num_blocks


def test_newly_prefilled_request_waits_until_next_decode_iteration() -> None:
    model = create_model()
    manager = create_manager(model)
    scheduler = RequestScheduler(
        max_running_requests=1,
        cache_manager=manager,
    )
    request = create_request(
        "A",
        torch.tensor([[1, 2]], dtype=torch.int64),
        max_new_tokens=3,
    )
    scheduler.submit(request)
    engine = ContinuousBatchEngine(model, scheduler, manager)

    first = engine.step()

    assert first.prefill_ids == ("A",)
    assert first.decode_ids == ()
    assert request.num_generated_tokens == 1

    second = engine.step()

    assert second.prefill_ids == ()
    assert second.decode_ids == ("A",)
    assert request.num_generated_tokens == 2


def test_completed_requests_never_execute_again() -> None:
    model = create_model()
    manager = create_manager(model)
    scheduler = RequestScheduler(
        max_running_requests=1,
        cache_manager=manager,
    )
    scheduler.submit(
        create_request(
            "A",
            torch.tensor([[1]], dtype=torch.int64),
            max_new_tokens=1,
        )
    )
    engine = ContinuousBatchEngine(model, scheduler, manager)

    first = engine.step()
    second = engine.step()

    assert first.completed_ids == ("A",)
    assert second.prefill_ids == ()
    assert second.decode_ids == ()
    assert second.completed_ids == ()


def test_engine_rejects_different_cache_manager() -> None:
    model = create_model()
    scheduler_manager = create_manager(model)
    other_manager = create_manager(model)
    scheduler = RequestScheduler(
        max_running_requests=1,
        cache_manager=scheduler_manager,
    )

    with pytest.raises(ValueError, match="share the cache manager"):
        ContinuousBatchEngine(model, scheduler, other_manager)
