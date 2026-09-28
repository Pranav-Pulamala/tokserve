import torch

from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.generation.generate import generate
from tokserve.generation.types import GenerationConfig
from tokserve.reference.llama.config import LlamaConfig
from tokserve.scheduler.executor import ScheduledGenerationExecutor
from tokserve.scheduler.request import RequestState, ScheduledRequest
from tokserve.scheduler.scheduler import RequestScheduler


def create_model() -> LlamaModel:
    torch.manual_seed(110)
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


def test_scheduled_generation_matches_direct_generation_in_fifo_order() -> None:
    model = create_model()
    manager = create_manager(model)
    scheduler = RequestScheduler(
        max_running_requests=1,
        cache_manager=manager,
    )
    prompts = (
        torch.tensor([[1, 2]], dtype=torch.int64),
        torch.tensor([[3, 4, 5]], dtype=torch.int64),
    )
    config = GenerationConfig(max_new_tokens=3)

    scheduler.submit(ScheduledRequest("A", prompts[0], config))
    scheduler.submit(ScheduledRequest("B", prompts[1], config))

    expected = tuple(generate(model, prompt, config) for prompt in prompts)
    executor = ScheduledGenerationExecutor(model, scheduler, manager)
    actual = executor.run_until_idle()

    assert len(actual) == 2
    torch.testing.assert_close(actual[0].token_ids, expected[0].token_ids)
    torch.testing.assert_close(actual[1].token_ids, expected[1].token_ids)
    assert scheduler.get_request("A").state is RequestState.FINISHED
    assert scheduler.get_request("B").state is RequestState.FINISHED
    assert scheduler.waiting_count == 0
    assert scheduler.running_count == 0
    assert manager.active_sequence_ids == ()
    assert manager.allocator.allocated_count == 0


def test_scheduled_generation_respects_eos() -> None:
    model = create_model()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    manager = create_manager(model)
    scheduler = RequestScheduler(
        max_running_requests=1,
        cache_manager=manager,
    )
    scheduler.submit(
        ScheduledRequest(
            "eos",
            torch.tensor([[1]], dtype=torch.int64),
            GenerationConfig(max_new_tokens=5, eos_token_id=0),
        )
    )

    executor = ScheduledGenerationExecutor(model, scheduler, manager)
    result = executor.run_next()

    assert result is not None
    assert result.num_generated_tokens == 1
    assert result.stop_reason == "eos"
    assert scheduler.get_request("eos").state is RequestState.FINISHED
    assert manager.allocator.allocated_count == 0
