import torch
from fastapi.testclient import TestClient

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


def create_service() -> GenerationService:
    torch.manual_seed(143)
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
    parameter = model.embed_tokens.weight
    manager = PagedKVCacheManager(
        num_layers=model.config.num_hidden_layers,
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
    metrics = RequestMetricsCollector()
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


def test_metrics_endpoint_reports_idle_state() -> None:
    application = create_app(create_service())

    with TestClient(application) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()

    assert body["schema_version"] == 1
    assert body["duration_unit"] == "seconds"
    assert body["requests"]["received_total"] == 0
    assert body["requests"]["waiting_current"] == 0
    assert body["requests"]["running_current"] == 0
    assert body["kv_cache"]["capacity_blocks"] == 24
    assert body["kv_cache"]["allocated_blocks"] == 0
    assert body["kv_cache"]["free_blocks"] == 24
    assert body["kv_cache"]["utilization_ratio"] == 0.0


def test_metrics_endpoint_reflects_completed_request() -> None:
    application = create_app(create_service())

    with TestClient(application) as client:
        generated = client.post(
            "/v1/generate",
            json={
                "prompt": "1 2",
                "max_new_tokens": 3,
            },
        )
        metrics = client.get("/metrics")

    assert generated.status_code == 200
    assert metrics.status_code == 200

    body = metrics.json()

    assert body["requests"]["received_total"] == 1
    assert body["requests"]["admitted_total"] == 1
    assert body["requests"]["completed_total"] == 1
    assert body["requests"]["failed_total"] == 0
    assert body["requests"]["cancelled_total"] == 0
    assert body["tokens"]["generated_output_tokens_total"] == 3
    assert body["batching"]["iterations_total"] == 3
    assert body["batching"]["prefill_requests_total"] == 1
    assert body["batching"]["decode_requests_total"] == 2
    assert body["kv_cache"]["allocated_blocks"] == 0
    assert body["kv_cache"]["free_blocks"] == 24

    serialized = metrics.text.lower()

    assert "prompt" not in serialized
    assert "generated_text" not in serialized
    assert "token_id" not in serialized
    assert "request-1" not in serialized


def test_metrics_without_service_returns_503() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/metrics")

    assert response.status_code == 503
    assert response.json() == {"detail": "generation service is unavailable"}


def test_health_behavior_is_unchanged() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
