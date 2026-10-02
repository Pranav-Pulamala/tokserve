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
    torch.manual_seed(125)
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


def test_continuous_batching_end_to_end() -> None:
    model = create_model()
    manager = create_manager(model)
    scheduler = RequestScheduler(
        max_running_requests=2,
        cache_manager=manager,
    )
    specifications = {
        "A": (
            torch.tensor([[1, 2]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=4),
        ),
        "B": (
            torch.tensor([[3, 4, 5]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=2),
        ),
        "C": (
            torch.tensor([[6]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=3),
        ),
        "D": (
            torch.tensor([[7, 8, 9, 10]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=1),
        ),
    }
    requests = {
        request_id: ScheduledRequest(request_id, prompt, config)
        for request_id, (prompt, config) in specifications.items()
    }
    expected = {
        request_id: generate(model, prompt, config)
        for request_id, (prompt, config) in specifications.items()
    }

    scheduler.submit(requests["A"])
    scheduler.submit(requests["B"])

    engine = ContinuousBatchEngine(model, scheduler, manager)
    released_blocks: dict[str, tuple[int, ...]] = {}
    original_release = manager.release_sequence

    def record_release(sequence_id: str) -> None:
        released_blocks[sequence_id] = manager.get_sequence(sequence_id).block_ids
        original_release(sequence_id)

    with patch.object(
        manager,
        "release_sequence",
        side_effect=record_release,
    ):
        iteration_zero = engine.step()

        assert iteration_zero.iteration == 0
        assert iteration_zero.admitted_ids == ("A", "B")
        assert iteration_zero.prefill_ids == ("A", "B")
        assert iteration_zero.decode_ids == ()
        assert iteration_zero.completed_ids == ()
        assert iteration_zero.active_ids == ("A", "B")

        iteration_one = engine.step()

        assert iteration_one.iteration == 1
        assert iteration_one.admitted_ids == ()
        assert iteration_one.prefill_ids == ()
        assert iteration_one.decode_ids == ("A", "B")
        assert iteration_one.completed_ids == ("B",)
        assert iteration_one.active_ids == ("A",)

        scheduler.submit(requests["C"])
        iteration_two = engine.step()

        assert iteration_two.iteration == 2
        assert iteration_two.admitted_ids == ("C",)
        assert iteration_two.prefill_ids == ("C",)
        assert iteration_two.decode_ids == ("A",)
        assert iteration_two.completed_ids == ()
        assert iteration_two.active_ids == ("A", "C")

        scheduler.submit(requests["D"])
        iteration_three = engine.step()

        assert iteration_three.iteration == 3
        assert iteration_three.admitted_ids == ()
        assert iteration_three.prefill_ids == ()
        assert iteration_three.decode_ids == ("A", "C")
        assert iteration_three.completed_ids == ("A",)
        assert iteration_three.active_ids == ("C",)
        assert requests["D"].state is RequestState.WAITING

        iteration_four = engine.step()

        assert iteration_four.iteration == 4
        assert iteration_four.admitted_ids == ("D",)
        assert iteration_four.prefill_ids == ("D",)
        assert iteration_four.decode_ids == ("C",)
        assert iteration_four.completed_ids == ("C", "D")
        assert iteration_four.active_ids == ()

    assert engine.history == (
        iteration_zero,
        iteration_one,
        iteration_two,
        iteration_three,
        iteration_four,
    )

    for request_id, request in requests.items():
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
        assert result.stop_reason == expected[request_id].stop_reason

    assert set(released_blocks) == {"A", "B", "C", "D"}
    assert set(released_blocks["D"]).intersection(
        (*released_blocks["A"], *released_blocks["B"])
    )
    assert manager.active_sequence_ids == ()
    assert manager.allocator.allocated_count == 0
    assert manager.allocator.free_count == manager.allocator.num_blocks
    assert scheduler.waiting_count == 0
    assert scheduler.running_count == 0


def test_run_until_idle_records_every_iteration() -> None:
    model = create_model()
    manager = create_manager(model)
    scheduler = RequestScheduler(
        max_running_requests=2,
        cache_manager=manager,
    )
    scheduler.submit(
        create_request(
            "A",
            torch.tensor([[1]], dtype=torch.int64),
            max_new_tokens=2,
        )
    )
    scheduler.submit(
        create_request(
            "B",
            torch.tensor([[2, 3]], dtype=torch.int64),
            max_new_tokens=3,
        )
    )
    engine = ContinuousBatchEngine(model, scheduler, manager)

    results = engine.run_until_idle()

    assert len(results) == 2
    assert len(engine.history) == 3
    assert engine.history[0].prefill_ids == ("A", "B")
    assert engine.history[1].decode_ids == ("A", "B")
    assert engine.history[1].completed_ids == ("A",)
    assert engine.history[2].decode_ids == ("B",)
    assert engine.history[2].completed_ids == ("B",)
    assert scheduler.waiting_count == 0
    assert scheduler.running_count == 0
    assert manager.allocator.allocated_count == 0
