"""Asynchronous adapter over TokServe's continuous-batching engine."""

import asyncio
from itertools import count

import torch

from tokserve.api.schemas import GenerateRequest, GenerateResponse
from tokserve.api.tokenization import (
    Tokenizer,
    decode_generated_tokens,
    encode_prompt,
)
from tokserve.batching.engine import ContinuousBatchEngine
from tokserve.scheduler.request import RequestState, ScheduledRequest


class RequestRejectedError(ValueError):
    """Raised when validated API input is incompatible with the engine."""


class GenerationServiceError(RuntimeError):
    """Raised when the serving engine cannot complete a request."""


class GenerationService:
    """Submit API requests to one shared continuous-batching engine."""

    def __init__(
        self,
        engine: ContinuousBatchEngine,
        tokenizer: Tokenizer,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self._iteration_lock = asyncio.Lock()
        self._request_numbers = count(1)

    async def generate(
        self,
        payload: GenerateRequest,
    ) -> GenerateResponse:
        """Submit and complete one non-streaming generation request."""

        request = await self._submit(payload)

        while request.state in (
            RequestState.WAITING,
            RequestState.RUNNING,
        ):
            try:
                async with self._iteration_lock:
                    if request.state in (
                        RequestState.WAITING,
                        RequestState.RUNNING,
                    ):
                        await asyncio.to_thread(self.engine.step)
            except Exception as error:
                await self._cancel_if_active(request)
                raise GenerationServiceError("generation engine failed") from error

            await asyncio.sleep(0)

        if request.state is not RequestState.FINISHED:
            raise GenerationServiceError(
                f"request ended in state {request.state.value}"
            )

        result = request.result

        if result is None:
            raise GenerationServiceError("finished request has no generation result")

        return GenerateResponse(
            request_id=request.request_id,
            generated_text=decode_generated_tokens(
                self.tokenizer,
                result.generated_token_ids,
            ),
            generated_tokens=result.num_generated_tokens,
            finish_reason=result.stop_reason,
        )

    async def _submit(
        self,
        payload: GenerateRequest,
    ) -> ScheduledRequest:
        """Tokenize, validate, and submit one request."""

        model = self.engine.model
        parameter = model.embed_tokens.weight
        input_ids = encode_prompt(
            self.tokenizer,
            payload.prompt,
            device=parameter.device,
        )
        total_length = input_ids.shape[1] + payload.max_new_tokens

        if total_length > model.config.max_position_embeddings:
            raise RequestRejectedError(
                "prompt and generation exceed model context length"
            )

        if torch.any(input_ids >= model.config.vocab_size):
            raise RequestRejectedError(
                "tokenizer produced an ID outside the model vocabulary"
            )

        block_size = self.engine.cache_manager.storage.block_size
        required_blocks = (total_length + block_size - 1) // block_size

        if required_blocks > self.engine.cache_manager.allocator.num_blocks:
            raise RequestRejectedError("request exceeds available KV-cache capacity")

        request = ScheduledRequest(
            f"request-{next(self._request_numbers)}",
            input_ids,
            payload.to_generation_config(),
        )

        async with self._iteration_lock:
            self.engine.scheduler.submit(request)

        return request

    async def _cancel_if_active(
        self,
        request: ScheduledRequest,
    ) -> None:
        """Cancel a failed request through scheduler ownership."""

        async with self._iteration_lock:
            if request.state in (
                RequestState.WAITING,
                RequestState.RUNNING,
            ):
                self.engine.scheduler.cancel(request.request_id)
