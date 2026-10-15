import asyncio

import torch

from tokserve.api.schemas import GenerateRequest
from tokserve.api.service import GenerationService
from tokserve.api.streaming import TokenEvent
from tokserve.batching.engine import (
    ContinuousBatchEngine,
)
from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.manager import (
    PagedKVCacheManager,
)
from tokserve.metrics.request_metrics import (
    RequestMetricsCollector,
)
from tokserve.reference.llama.config import (
    LlamaConfig,
)
from tokserve.scheduler.request import RequestState
from tokserve.scheduler.scheduler import (
    RequestScheduler,
)


class IntegerTokenizer:
    """Encode and decode deterministic integer IDs."""

    def encode(self, text: str) -> list[int]:
        return [int(piece) for piece in text.split()]

    def decode(
        self,
        token_ids: list[int],
    ) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def create_service(
    *,
    zero_model: bool = False,
) -> GenerationService:
    torch.manual_seed(140)
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

    if zero_model:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()

    parameter = model.embed_tokens.weight
    manager = PagedKVCacheManager(
        num_layers=(model.config.num_hidden_layers),
        num_blocks=24,
        num_key_value_heads=(model.config.num_key_value_heads),
        block_size=2,
        head_dim=model.config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    scheduler = RequestScheduler(
        max_running_requests=4,
        cache_manager=manager,
    )
    metrics = RequestMetricsCollector(history_size=16)
    engine = ContinuousBatchEngine(
        model,
        scheduler,
        manager,
        metrics=metrics,
    )
    return GenerationService(
        engine,
        IntegerTokenizer(),
        metrics=metrics,
    )


def test_non_stream_request_records_lifecycle() -> None:
    service = create_service()

    response = asyncio.run(
        service.generate(
            GenerateRequest(
                prompt="1 2",
                max_new_tokens=3,
            )
        )
    )
    snapshot = service.metrics.snapshot()

    assert response.generated_tokens == 3
    assert snapshot.requests_received == 1
    assert snapshot.requests_admitted == 1
    assert snapshot.requests_completed == 1
    assert snapshot.requests_failed == 0
    assert snapshot.requests_cancelled == 0
    assert snapshot.generated_output_tokens == 3
    assert snapshot.active_requests == 0
    assert snapshot.prefill_requests == 1
    assert snapshot.decode_requests == 2

    record = snapshot.completed_records[0]

    assert record.outcome == "completed"
    assert record.completion_reason == "max_new_tokens"
    assert record.generated_tokens == 3
    assert record.queue_delay_seconds is not None
    assert record.prefill_duration_seconds is not None
    assert record.ttft_seconds is not None
    assert record.request_latency_seconds >= 0.0
    assert all(latency >= 0.0 for latency in (record.inter_token_latencies_seconds))


def test_two_requests_share_iteration_metrics() -> None:
    service = create_service()

    async def run() -> None:
        await asyncio.gather(
            service.generate(
                GenerateRequest(
                    prompt="1",
                    max_new_tokens=2,
                )
            ),
            service.generate(
                GenerateRequest(
                    prompt="2 3",
                    max_new_tokens=3,
                )
            ),
        )

    asyncio.run(run())
    snapshot = service.metrics.snapshot()

    assert snapshot.requests_received == 2
    assert snapshot.requests_admitted == 2
    assert snapshot.requests_completed == 2
    assert snapshot.generated_output_tokens == 5
    assert snapshot.active_requests == 0
    assert snapshot.maximum_batch_size >= 1
    assert snapshot.terminal_requests == 2


def test_eos_completion_is_recorded_once() -> None:
    service = create_service(zero_model=True)

    response = asyncio.run(
        service.generate(
            GenerateRequest(
                prompt="1",
                max_new_tokens=4,
                eos_token_id=0,
            )
        )
    )
    snapshot = service.metrics.snapshot()
    record = snapshot.completed_records[0]

    assert response.finish_reason == "eos"
    assert response.generated_tokens == 1
    assert snapshot.requests_completed == 1
    assert snapshot.generated_output_tokens == 1
    assert record.completion_reason == "eos"


def test_stream_disconnect_records_cancellation() -> None:
    service = create_service()

    async def run() -> RequestState:
        session = await service.start_stream(
            GenerateRequest(
                prompt="1",
                max_new_tokens=3,
                stream=True,
            )
        )
        events = service.stream_events(session)
        first = await anext(events)

        assert isinstance(
            first,
            TokenEvent,
        )

        await events.aclose()
        return session.request.state

    state = asyncio.run(run())
    snapshot = service.metrics.snapshot()

    assert state is RequestState.CANCELLED
    assert snapshot.requests_cancelled == 1
    assert snapshot.requests_completed == 0
    assert snapshot.active_requests == 0
    assert snapshot.terminal_requests == 1
    assert service.engine.cache_manager.allocator.allocated_count == 0
