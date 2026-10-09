import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import torch
from fastapi.testclient import TestClient

from tokserve.api.app import create_app
from tokserve.api.schemas import GenerateRequest
from tokserve.api.service import (
    GenerationService,
    GenerationServiceError,
)
from tokserve.api.streaming import (
    DoneEvent,
    GenerationEvent,
    TokenEvent,
)
from tokserve.batching.engine import ContinuousBatchEngine
from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.generation.generate import generate
from tokserve.generation.types import GenerationConfig
from tokserve.reference.llama.config import LlamaConfig
from tokserve.scheduler.request import RequestState
from tokserve.scheduler.scheduler import RequestScheduler


class IntegerTokenizer:
    """Encode space-separated integers and decode them deterministically."""

    def encode(self, text: str) -> list[int]:
        return [int(piece) for piece in text.split()]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def create_model() -> LlamaModel:
    torch.manual_seed(130)
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


def create_service(
    model: LlamaModel | None = None,
) -> GenerationService:
    if model is None:
        model = create_model()

    parameter = model.embed_tokens.weight
    manager = PagedKVCacheManager(
        num_layers=model.config.num_hidden_layers,
        num_blocks=24,
        num_key_value_heads=model.config.num_key_value_heads,
        block_size=2,
        head_dim=model.config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    scheduler = RequestScheduler(
        max_running_requests=4,
        cache_manager=manager,
    )
    engine = ContinuousBatchEngine(
        model,
        scheduler,
        manager,
    )
    return GenerationService(engine, IntegerTokenizer())


def test_non_streaming_generation_matches_tokserve_reference() -> None:
    model = create_model()
    service = create_service(model)
    application = create_app(service)
    prompt = torch.tensor([[1, 2]], dtype=torch.int64)
    expected = generate(
        model,
        prompt,
        GenerationConfig(max_new_tokens=3),
    )

    with TestClient(application) as client:
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "1 2",
                "max_new_tokens": 3,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "request-1"
    assert body["generated_tokens"] == 3
    assert body["finish_reason"] == "max_new_tokens"
    assert body["generated_text"] == " ".join(
        str(token_id) for token_id in expected.generated_token_ids[0].tolist()
    )
    assert service.engine.scheduler.waiting_count == 0
    assert service.engine.scheduler.running_count == 0
    assert service.engine.cache_manager.allocator.allocated_count == 0


def test_service_is_reused_across_http_requests() -> None:
    service = create_service()
    engine_identity = id(service.engine)
    application = create_app(service)

    with TestClient(application) as client:
        first = client.post(
            "/v1/generate",
            json={
                "prompt": "1",
                "max_new_tokens": 1,
            },
        )
        second = client.post(
            "/v1/generate",
            json={
                "prompt": "2",
                "max_new_tokens": 1,
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["request_id"] == "request-1"
    assert second.json()["request_id"] == "request-2"
    assert id(service.engine) == engine_identity


def test_eos_response() -> None:
    model = create_model()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    service = create_service(model)
    application = create_app(service)

    with TestClient(application) as client:
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "1",
                "max_new_tokens": 4,
                "eos_token_id": 0,
            },
        )

    assert response.status_code == 200
    assert response.json()["generated_text"] == "0"
    assert response.json()["generated_tokens"] == 1
    assert response.json()["finish_reason"] == "eos"


def test_invalid_body_returns_validation_error() -> None:
    application = create_app(create_service())

    with TestClient(application) as client:
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "",
                "max_new_tokens": 0,
            },
        )

    assert response.status_code == 422


def test_request_exceeding_model_context_is_rejected() -> None:
    application = create_app(create_service())

    with TestClient(application) as client:
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "1 2 3 4 5 6 7 8 9 10",
                "max_new_tokens": 10,
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "prompt and generation exceed model context length"
    }


def test_streaming_generation_emits_tokens_before_completion() -> None:
    application = create_app(create_service())

    with TestClient(application) as client:
        with client.stream(
            "POST",
            "/v1/generate",
            json={
                "prompt": "1",
                "max_new_tokens": 3,
                "stream": True,
            },
        ) as response:
            lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert lines[0] == "event: token"
    assert lines[-2] == "event: done"
    assert '"finish_reason":"max_new_tokens"' in lines[-1]
    assert sum(line == "event: token" for line in lines) == 3


def test_streaming_and_non_streaming_text_match() -> None:
    service = create_service()
    application = create_app(service)

    with TestClient(application) as client:
        complete = client.post(
            "/v1/generate",
            json={
                "prompt": "1 2",
                "max_new_tokens": 3,
            },
        )

        with client.stream(
            "POST",
            "/v1/generate",
            json={
                "prompt": "1 2",
                "max_new_tokens": 3,
                "stream": True,
            },
        ) as streamed:
            data_lines = [
                line.removeprefix("data: ")
                for line in streamed.iter_lines()
                if line.startswith("data: ")
            ]

    token_payloads = [json.loads(line) for line in data_lines[:-1]]

    assert complete.status_code == 200
    assert streamed.status_code == 200
    assert token_payloads[-1]["generated_text"] == complete.json()["generated_text"]


def test_missing_generation_service_returns_unavailable() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "1",
                "max_new_tokens": 1,
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "generation service is unavailable"}


def test_engine_failure_returns_generic_error() -> None:
    service = create_service()
    service.generate = AsyncMock(side_effect=GenerationServiceError("private failure"))
    application = create_app(service)

    with TestClient(
        application,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "1",
                "max_new_tokens": 1,
            },
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "generation failed"}
    assert "private failure" not in response.text


def test_schema_conversion_used_by_service() -> None:
    payload = GenerateRequest(
        prompt="1",
        max_new_tokens=2,
        temperature=0.5,
    )

    assert payload.to_generation_config().max_new_tokens == 2


def test_closing_stream_cancels_request_and_releases_resources() -> None:
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

        first_event = await anext(events)
        assert isinstance(first_event, TokenEvent)

        await events.aclose()
        return session.request.state

    state = asyncio.run(run())

    assert state is RequestState.CANCELLED
    assert service.engine.scheduler.waiting_count == 0
    assert service.engine.scheduler.running_count == 0
    assert service.engine.cache_manager.allocator.allocated_count == 0
    assert service.active_stream_request_ids == ()


def test_shutdown_cancels_waiting_streams() -> None:
    service = create_service()

    async def run() -> RequestState:
        session = await service.start_stream(
            GenerateRequest(
                prompt="1",
                max_new_tokens=3,
                stream=True,
            )
        )
        await service.shutdown()
        return session.request.state

    state = asyncio.run(run())

    assert state is RequestState.CANCELLED
    assert service.engine.scheduler.waiting_count == 0
    assert service.engine.scheduler.running_count == 0
    assert service.active_stream_request_ids == ()


def test_multiple_streams_remain_isolated() -> None:
    service = create_service()

    async def collect(
        session_id: str,
        events: list[GenerationEvent],
    ) -> tuple[str, list[GenerationEvent]]:
        return session_id, events

    async def run() -> tuple[
        tuple[str, list[GenerationEvent]],
        tuple[str, list[GenerationEvent]],
    ]:
        first_session = await service.start_stream(
            GenerateRequest(
                prompt="1",
                max_new_tokens=3,
                stream=True,
            )
        )
        second_session = await service.start_stream(
            GenerateRequest(
                prompt="2",
                max_new_tokens=2,
                stream=True,
            )
        )

        async def consume(
            session_id: str,
            session_events: AsyncIterator[GenerationEvent],
        ) -> tuple[str, list[GenerationEvent]]:
            collected = [event async for event in session_events]
            return await collect(session_id, collected)

        return await asyncio.gather(
            consume(
                first_session.request_id,
                service.stream_events(first_session),
            ),
            consume(
                second_session.request_id,
                service.stream_events(second_session),
            ),
        )

    first, second = asyncio.run(run())

    first_id, first_events = first
    second_id, second_events = second

    assert first_id != second_id
    assert sum(isinstance(event, TokenEvent) for event in first_events) == 3
    assert sum(isinstance(event, TokenEvent) for event in second_events) == 2
    assert isinstance(first_events[-1], DoneEvent)
    assert isinstance(second_events[-1], DoneEvent)
    assert service.active_stream_request_ids == ()
    assert service.engine.scheduler.waiting_count == 0
    assert service.engine.scheduler.running_count == 0
    assert service.engine.cache_manager.allocator.allocated_count == 0


def test_invalid_streaming_request_is_rejected_before_sse_starts() -> None:
    application = create_app(create_service())

    with TestClient(application) as client:
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "1 2 3 4 5 6 7 8 9 10",
                "max_new_tokens": 10,
                "stream": True,
            },
        )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "detail": ("prompt and generation exceed model context length")
    }


def test_streaming_response_uses_security_headers() -> None:
    application = create_app(create_service())

    with TestClient(application) as client:
        with client.stream(
            "POST",
            "/v1/generate",
            json={
                "prompt": "1",
                "max_new_tokens": 1,
                "stream": True,
            },
        ) as response:
            list(response.iter_lines())

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-type"].startswith("text/event-stream")
