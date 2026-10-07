from fastapi import FastAPI
from fastapi.testclient import TestClient

from tokserve.api.app import app, create_app


def test_application_factory_returns_fastapi_app() -> None:
    application = create_app()

    assert isinstance(application, FastAPI)
    assert application.title == "TokServe"


def test_health_endpoint() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_route_returns_not_found() -> None:
    client = TestClient(create_app())

    response = client.get("/missing")

    assert response.status_code == 404


def test_module_app_does_not_initialize_inference_state() -> None:
    assert isinstance(app, FastAPI)
    assert not hasattr(app.state, "engine")
    assert not hasattr(app.state, "model")
    assert not hasattr(app.state, "tokenizer")
