import pytest
from pydantic import ValidationError

from tokserve.api.schemas import (
    MAX_NEW_TOKENS,
    MAX_PROMPT_CHARACTERS,
    GenerateRequest,
    GenerateResponse,
)


def test_request_defaults_match_generation_defaults() -> None:
    request = GenerateRequest(prompt="Hello")

    assert request.max_new_tokens == 32
    assert request.do_sample is False
    assert request.temperature == 1.0
    assert request.top_k is None
    assert request.top_p is None
    assert request.stream is False

    config = request.to_generation_config()

    assert config.max_new_tokens == 32
    assert config.do_sample is False
    assert config.temperature == 1.0


def test_supported_sampling_controls_are_preserved() -> None:
    request = GenerateRequest(
        prompt="Hello",
        max_new_tokens=8,
        do_sample=True,
        temperature=0.7,
        top_k=20,
        top_p=0.9,
        eos_token_id=2,
        seed=13,
    )

    config = request.to_generation_config()

    assert config.max_new_tokens == 8
    assert config.do_sample is True
    assert config.temperature == 0.7
    assert config.top_k == 20
    assert config.top_p == 0.9
    assert config.eos_token_id == 2
    assert config.seed == 13


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "   ",
        "\n\t",
        "x" * (MAX_PROMPT_CHARACTERS + 1),
    ],
)
def test_invalid_prompt_is_rejected(prompt: str) -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt=prompt)


@pytest.mark.parametrize(
    "max_new_tokens",
    [
        0,
        -1,
        MAX_NEW_TOKENS + 1,
    ],
)
def test_invalid_token_limit_is_rejected(
    max_new_tokens: int,
) -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(
            prompt="Hello",
            max_new_tokens=max_new_tokens,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.0),
        ("temperature", -1.0),
        ("temperature", 101.0),
        ("top_k", 0),
        ("top_p", 0.0),
        ("top_p", 1.1),
        ("eos_token_id", -1),
    ],
)
def test_invalid_sampling_parameter_is_rejected(
    field: str,
    value: int | float,
) -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(
            prompt="Hello",
            **{field: value},
        )


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(
            prompt="Hello",
            unexpected=True,
        )


def test_strict_schema_rejects_coerced_values() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(
            prompt="Hello",
            max_new_tokens="8",
        )


def test_response_serialization() -> None:
    response = GenerateResponse(
        request_id="request-1",
        generated_text=" world",
        generated_tokens=1,
        finish_reason="eos",
    )

    assert response.model_dump() == {
        "request_id": "request-1",
        "generated_text": " world",
        "generated_tokens": 1,
        "finish_reason": "eos",
    }
