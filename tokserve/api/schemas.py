"""Validated HTTP schemas for text generation."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tokserve.generation.types import GenerationConfig

MAX_PROMPT_CHARACTERS = 16_384
MAX_NEW_TOKENS = 2_048


class GenerateRequest(BaseModel):
    """Client-facing generation request."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    prompt: Annotated[
        str,
        Field(min_length=1, max_length=MAX_PROMPT_CHARACTERS),
    ]
    max_new_tokens: Annotated[
        int,
        Field(ge=1, le=MAX_NEW_TOKENS),
    ] = 32
    do_sample: bool = False
    temperature: Annotated[float, Field(gt=0.0, le=100.0)] = 1.0
    top_k: Annotated[int | None, Field(ge=1, le=100_000)] = None
    top_p: Annotated[float | None, Field(gt=0.0, le=1.0)] = None
    eos_token_id: Annotated[int | None, Field(ge=0)] = None
    seed: int | None = None
    stream: bool = False

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, prompt: str) -> str:
        """Reject prompts containing only whitespace."""

        if not prompt.strip():
            raise ValueError("prompt must contain non-whitespace text")

        return prompt

    def to_generation_config(self) -> GenerationConfig:
        """Convert validated HTTP controls to engine configuration."""

        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            eos_token_id=self.eos_token_id,
            seed=self.seed,
        )


class GenerateResponse(BaseModel):
    """Completed non-streaming generation response."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    generated_text: str
    generated_tokens: Annotated[int, Field(ge=0)]
    finish_reason: Literal["eos", "max_new_tokens"]
