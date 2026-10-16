import asyncio

import httpx
import torch
from fastapi.testclient import TestClient

from benchmarks.load_test import (
    LoadTestConfig,
    run_load_test,
)
from tokserve.api.app import create_app
from tokserve.api.service import GenerationService
from tokserve.batching.engine import ContinuousBatchEngine
from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.manager import (
    PagedKVCacheManager,
)
from tokserve.metrics.request_metrics import (
    RequestMetricsCollector,
)
from tokserve.reference.llama.config import LlamaConfig
from tokserve.scheduler.scheduler import RequestScheduler


class IntegerTokenizer:
    """Encode and decode space-separated integer IDs."""

    def encode(self, text: str) -> list[int]:
        return [int(piece) for piece in text.split()]

    def decode(
        self,
        token_ids: list[int],
    ) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def create_service(
    *,
    max_running_requests: int = 4,
    history_size: int = 16,
    zero_model: bool = False,
) -> GenerationService:
    torch.manual_seed(146)

    model = LlamaModel(
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )
    model.eval()

    if zero_model:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()

    parameter = model.embed_tokens.weight
    manager = PagedKVCacheManager(
        num_layers=model.config.num_hidden_layers,
        num_blocks=32,
        num_key_value_heads=(model.config.num_key_value_heads),
        block_size=2,
        head_dim=model.config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    scheduler = RequestScheduler(
        max_running_requests=max_running_requests,
        cache_manager=manager,
    )
    metrics = RequestMetricsCollector(history_size=history_size)
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


def create_load_config(
    *,
    requests: int,
    concurrency: int,
    stream: bool,
    max_new_tokens: int,
) -> LoadTestConfig:
    return LoadTestConfig(
        base_url="http://tokserve.test",
        prompts=(
            "1",
            "2 3",
            "4 5 6",
        ),
        requests=requests,
        concurrency=concurrency,
        max_new_tokens=max_new_tokens,
        stream=stream,
        warmup_requests=0,
        timeout_seconds=10.0,
    )


def test_non_stream_load_reconciles_with_server_metrics() -> None:
    service = create_service()
    application = create_app(service)
    transport = httpx.ASGITransport(app=application)

    result = asyncio.run(
        run_load_test(
            create_load_config(
                requests=5,
                concurrency=3,
                stream=False,
                max_new_tokens=3,
            ),
            transport=transport,
        )
    )
    snapshot = service.metrics.snapshot()

    assert result.summary.requested == 5
    assert result.summary.succeeded == 5
    assert result.summary.failed == 0
    assert result.summary.generated_output_tokens == 15

    assert snapshot.requests_received == 5
    assert snapshot.requests_admitted == 5
    assert snapshot.requests_completed == 5
    assert snapshot.requests_failed == 0
    assert snapshot.requests_cancelled == 0
    assert snapshot.generated_output_tokens == 15
    assert snapshot.active_requests == 0
    assert snapshot.terminal_requests == 5

    assert service.engine.scheduler.waiting_count == 0
    assert service.engine.scheduler.running_count == 0
    assert service.engine.cache_manager.allocator.allocated_count == 0


def test_stream_load_reconciles_token_events() -> None:
    service = create_service()
    application = create_app(service)
    transport = httpx.ASGITransport(app=application)

    result = asyncio.run(
        run_load_test(
            create_load_config(
                requests=4,
                concurrency=2,
                stream=True,
                max_new_tokens=2,
            ),
            transport=transport,
        )
    )
    snapshot = service.metrics.snapshot()

    assert result.summary.requested == 4
    assert result.summary.succeeded == 4
    assert result.summary.failed == 0
    assert result.summary.generated_output_tokens == 8
    assert snapshot.requests_completed == 4
    assert snapshot.generated_output_tokens == 8
    assert snapshot.terminal_requests == 4
    assert snapshot.active_requests == 0

    assert all(
        observation.first_token_seconds is not None
        for observation in result.observations
    )
    assert all(observation.generated_tokens == 2 for observation in result.observations)
    assert service.engine.cache_manager.allocator.allocated_count == 0


def test_eos_and_max_tokens_reconcile() -> None:
    service = create_service(zero_model=True)
    application = create_app(service)

    with TestClient(application) as client:
        eos_response = client.post(
            "/v1/generate",
            json={
                "prompt": "1",
                "max_new_tokens": 4,
                "eos_token_id": 0,
            },
        )
        maximum_response = client.post(
            "/v1/generate",
            json={
                "prompt": "2 3 4",
                "max_new_tokens": 3,
            },
        )
        metrics_response = client.get("/metrics")

    assert eos_response.status_code == 200
    assert maximum_response.status_code == 200
    assert metrics_response.status_code == 200

    assert eos_response.json()["finish_reason"] == "eos"
    assert eos_response.json()["generated_tokens"] == 1
    assert maximum_response.json()["finish_reason"] == "max_new_tokens"
    assert maximum_response.json()["generated_tokens"] == 3

    body = metrics_response.json()

    assert body["requests"]["completed_total"] == 2
    assert body["requests"]["failed_total"] == 0
    assert body["requests"]["cancelled_total"] == 0
    assert body["tokens"]["generated_output_tokens_total"] == 4
    assert body["kv_cache"]["allocated_blocks"] == 0
    assert body["kv_cache"]["free_blocks"] == 32
    assert body["observations"]["window_completed_requests"] == 2


def test_queue_pressure_preserves_all_requests() -> None:
    service = create_service(max_running_requests=1)
    application = create_app(service)
    transport = httpx.ASGITransport(app=application)

    result = asyncio.run(
        run_load_test(
            create_load_config(
                requests=6,
                concurrency=6,
                stream=False,
                max_new_tokens=2,
            ),
            transport=transport,
        )
    )
    snapshot = service.metrics.snapshot()

    assert result.summary.succeeded == 6
    assert result.summary.failed == 0
    assert snapshot.requests_received == 6
    assert snapshot.requests_admitted == 6
    assert snapshot.requests_completed == 6
    assert snapshot.terminal_requests == 6
    assert snapshot.maximum_batch_size == 1
    assert snapshot.active_requests == 0
    assert all(
        record.queue_delay_seconds is not None and record.queue_delay_seconds >= 0.0
        for record in snapshot.completed_records
    )
    assert service.engine.cache_manager.allocator.allocated_count == 0


def test_completed_history_remains_bounded() -> None:
    service = create_service(history_size=3)
    application = create_app(service)

    with TestClient(application) as client:
        for prompt in (
            "1",
            "2",
            "3",
            "4",
            "5",
        ):
            response = client.post(
                "/v1/generate",
                json={
                    "prompt": prompt,
                    "max_new_tokens": 1,
                },
            )
            assert response.status_code == 200

        metrics_response = client.get("/metrics")

    snapshot = service.metrics.snapshot()

    assert snapshot.requests_received == 5
    assert snapshot.requests_completed == 5
    assert snapshot.terminal_requests == 5
    assert len(snapshot.completed_records) == 3

    observations = metrics_response.json()["observations"]

    assert observations["window_terminal_requests"] == 3
    assert observations["window_completed_requests"] == 3


def test_all_observed_durations_are_nonnegative() -> None:
    service = create_service()
    application = create_app(service)

    with TestClient(application) as client:
        for max_new_tokens in (
            1,
            2,
            4,
        ):
            response = client.post(
                "/v1/generate",
                json={
                    "prompt": "1 2",
                    "max_new_tokens": (max_new_tokens),
                },
            )
            assert response.status_code == 200

    snapshot = service.metrics.snapshot()

    for record in snapshot.completed_records:
        assert record.request_latency_seconds >= 0.0

        if record.queue_delay_seconds is not None:
            assert record.queue_delay_seconds >= 0.0

        if record.prefill_duration_seconds is not None:
            assert record.prefill_duration_seconds >= 0.0

        if record.ttft_seconds is not None:
            assert record.ttft_seconds >= 0.0

        assert all(latency >= 0.0 for latency in (record.inter_token_latencies_seconds))


def test_health_generation_and_metrics_regression() -> None:
    service = create_service()
    application = create_app(service)

    with TestClient(application) as client:
        health = client.get("/health")
        generated = client.post(
            "/v1/generate",
            json={
                "prompt": "1 2",
                "max_new_tokens": 2,
            },
        )
        metrics = client.get("/metrics")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    assert generated.status_code == 200
    assert generated.json()["generated_tokens"] == 2

    assert metrics.status_code == 200
    assert metrics.json()["requests"]["completed_total"] == 1
    assert metrics.json()["kv_cache"]["allocated_blocks"] == 0
