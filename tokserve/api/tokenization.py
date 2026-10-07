"""Text and token-ID conversion at the HTTP boundary."""

from typing import Protocol

import torch


class Tokenizer(Protocol):
    """Minimal tokenizer behavior required by TokServe's API."""

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs into text."""


class HuggingFaceTokenizerLike(Protocol):
    """Subset of a Hugging Face tokenizer used by the adapter."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        """Convert text into token IDs."""

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        """Convert token IDs into text."""


class HuggingFaceTokenizerAdapter:
    """Adapt a Hugging Face tokenizer without using HF model inference."""

    def __init__(self, tokenizer: HuggingFaceTokenizerLike) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        """Encode text while retaining tokenizer-defined special tokens."""

        return self._tokenizer.encode(
            text,
            add_special_tokens=True,
        )

    def decode(self, token_ids: list[int]) -> str:
        """Decode generated IDs while hiding tokenizer special tokens."""

        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )


def encode_prompt(
    tokenizer: Tokenizer,
    prompt: str,
    *,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Encode one prompt as a `(1, T)` int64 tensor."""

    token_ids = tokenizer.encode(prompt)

    if not token_ids:
        raise ValueError("tokenizer produced an empty prompt")

    if any(
        not isinstance(token_id, int) or isinstance(token_id, bool)
        for token_id in token_ids
    ):
        raise TypeError("tokenizer must produce integer token IDs")

    if any(token_id < 0 for token_id in token_ids):
        raise ValueError("tokenizer produced a negative token ID")

    return torch.tensor(
        [token_ids],
        dtype=torch.int64,
        device=device,
    )


def decode_generated_tokens(
    tokenizer: Tokenizer,
    token_ids: torch.Tensor,
) -> str:
    """Decode generated IDs from one batch-size-one tensor."""

    if token_ids.ndim != 2 or token_ids.shape[0] != 1:
        raise ValueError("token_ids must have shape (1, T)")

    if token_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("token_ids must contain integers")

    return tokenizer.decode([int(token_id) for token_id in token_ids[0].tolist()])
