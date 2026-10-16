from fastapi.testclient import TestClient

from tokserve.api.app import create_app
from tokserve.api.dev_app import (
    create_development_service,
)


def test_development_service_uses_real_components() -> None:
    service = create_development_service()

    assert service.engine.model is not None
    assert service.engine.scheduler is not None
    assert service.engine.cache_manager is not None
    assert service.engine.metrics is service.metrics


def test_development_service_is_deterministic() -> None:
    first_app = create_app(create_development_service())
    second_app = create_app(create_development_service())
    payload = {
        "prompt": "1 2 3",
        "max_new_tokens": 3,
    }

    with TestClient(first_app) as first_client:
        first = first_client.post(
            "/v1/generate",
            json=payload,
        )

    with TestClient(second_app) as second_client:
        second = second_client.post(
            "/v1/generate",
            json=payload,
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["generated_text"] == second.json()["generated_text"]


def test_development_app_exposes_real_metrics() -> None:
    application = create_app(create_development_service())

    with TestClient(application) as client:
        generated = client.post(
            "/v1/generate",
            json={
                "prompt": "1 2",
                "max_new_tokens": 2,
            },
        )
        metrics = client.get("/metrics")

    assert generated.status_code == 200
    assert metrics.status_code == 200
    assert metrics.json()["requests"]["completed_total"] == 1
    assert metrics.json()["tokens"]["generated_output_tokens_total"] == 2
