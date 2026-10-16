from fastapi.testclient import TestClient

from tests.integration.test_metrics_endpoint import (
    create_service,
)
from tokserve.api.app import create_app


def test_endpoint_exposes_bounded_latency_aggregates() -> None:
    application = create_app(create_service())

    with TestClient(application) as client:
        generated = client.post(
            "/v1/generate",
            json={
                "prompt": "1 2",
                "max_new_tokens": 3,
            },
        )
        response = client.get("/metrics")

    assert generated.status_code == 200
    assert response.status_code == 200

    body = response.json()
    observations = body["observations"]

    assert body["percentile_method"] == "linear_interpolation"
    assert body["percentile_population"] == "bounded_terminal_request_history"
    assert observations["window_terminal_requests"] == 1
    assert observations["window_completed_requests"] == 1
    assert observations["window_generated_output_tokens"] == 3
    assert observations["success_rate"] == 1.0

    request_latency = observations["request_latency"]
    ttft = observations["ttft"]
    itl = observations["inter_token_latency"]

    assert request_latency["sample_count"] == 1
    assert request_latency["p50_seconds"] >= 0.0
    assert ttft["sample_count"] == 1
    assert ttft["p50_seconds"] >= 0.0
    assert itl["sample_count"] == 2
    assert itl["p50_seconds"] >= 0.0


def test_idle_endpoint_has_empty_observations() -> None:
    application = create_app(create_service())

    with TestClient(application) as client:
        response = client.get("/metrics")

    assert response.status_code == 200

    observations = response.json()["observations"]

    assert observations["window_terminal_requests"] == 0
    assert observations["request_latency"]["sample_count"] == 0
    assert observations["request_latency"]["p99_seconds"] is None
    assert observations["success_rate"] is None
    assert observations["aggregate_output_tokens_per_second"] is None
