import pytest
import torch

from tokserve.api.tokenization import (
    HuggingFaceTokenizerAdapter,
    decode_generated_tokens,
    encode_prompt,
)


class FakeTokenizer:
    """Small deterministic tokenizer requiring no network access."""

    def __init__(self) -> None:
        self._token_to_id = {
            "Hello": 1,
            "world": 2,
            "!": 3,
        }
        self._id_to_text = {
            1: "Hello",
            2: " world",
            3: "!",
        }

    def encode(self, text: str) -> list[int]:
        return [self._token_to_id[token] for token in text.split()]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self._id_to_text[token_id] for token_id in token_ids)


class FakeHuggingFaceTokenizer:
    """Record adapter options used for encoding and decoding."""

    def __init__(self) -> None:
        self.add_special_tokens: bool | None = None
        self.skip_special_tokens: bool | None = None

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        self.add_special_tokens = add_special_tokens
        return [1, len(text)]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        self.skip_special_tokens = skip_special_tokens
        return ",".join(str(token_id) for token_id in token_ids)


def test_prompt_tokenization_returns_batch_tensor() -> None:
    token_ids = encode_prompt(
        FakeTokenizer(),
        "Hello world",
    )

    torch.testing.assert_close(
        token_ids,
        torch.tensor([[1, 2]], dtype=torch.int64),
    )


def test_generated_tokens_decode_to_text() -> None:
    text = decode_generated_tokens(
        FakeTokenizer(),
        torch.tensor([[2, 3]], dtype=torch.int64),
    )

    assert text == " world!"


def test_empty_tokenization_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty prompt"):
        encode_prompt(FakeTokenizer(), "   ")


def test_negative_token_id_is_rejected() -> None:
    class NegativeTokenizer:
        def encode(self, _text: str) -> list[int]:
            return [-1]

        def decode(self, _token_ids: list[int]) -> str:
            return ""

    with pytest.raises(ValueError, match="negative"):
        encode_prompt(NegativeTokenizer(), "bad")


def test_noninteger_token_id_is_rejected() -> None:
    class InvalidTokenizer:
        def encode(self, _text: str) -> list[int]:
            return [True]

        def decode(self, _token_ids: list[int]) -> str:
            return ""

    with pytest.raises(TypeError, match="integer"):
        encode_prompt(InvalidTokenizer(), "bad")


def test_decode_requires_batch_size_one_tensor() -> None:
    with pytest.raises(ValueError, match=r"shape \(1, T\)"):
        decode_generated_tokens(
            FakeTokenizer(),
            torch.tensor([1, 2], dtype=torch.int64),
        )


def test_hugging_face_adapter_uses_safe_options() -> None:
    wrapped = FakeHuggingFaceTokenizer()
    adapter = HuggingFaceTokenizerAdapter(wrapped)

    assert adapter.encode("hello") == [1, 5]
    assert adapter.decode([1, 2]) == "1,2"
    assert wrapped.add_special_tokens is True
    assert wrapped.skip_special_tokens is True
