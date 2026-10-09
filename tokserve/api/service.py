"""Asynchronous adapter over TokServe's continuous-batching engine."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from itertools import count

import torch

from tokserve.api.schemas import GenerateRequest, GenerateResponse
from tokserve.api.streaming import (
    CancelledEvent,
    DoneEvent,
    FailureEvent,
    GenerationEvent,
    RequestEventBroker,
    RequestEventStream,
    StreamBackpressureError,
)
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


@dataclass(frozen=True)
class GenerationStreamSession:
    """One submitted request and its isolated event stream."""

    request: ScheduledRequest
    event_stream: RequestEventStream

    @property
    def request_id(self) -> str:
        """Return the server-generated request identifier."""

        return self.request.request_id


class GenerationService:
    """Submit API requests to one shared continuous-batching engine."""

    def __init__(
        self,
        engine: ContinuousBatchEngine,
        tokenizer: Tokenizer,
        *,
        stream_buffer_size: int = 16,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self._iteration_lock = asyncio.Lock()
        self._request_numbers = count(1)
        self._event_broker = RequestEventBroker(
            buffer_size=stream_buffer_size,
        )

    @property
    def active_stream_request_ids(self) -> tuple[str, ...]:
        """Return request IDs with registered event streams."""

        return self._event_broker.active_request_ids

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
                await self._advance_engine()
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

    async def start_stream(
        self,
        payload: GenerateRequest,
    ) -> GenerationStreamSession:
        """Submit one request and register its event stream."""

        request = await self._submit(payload)
        event_stream = self._event_broker.open_stream(
            request.request_id,
        )
        return GenerationStreamSession(
            request=request,
            event_stream=event_stream,
        )

    async def stream_events(
        self,
        session: GenerationStreamSession,
    ) -> AsyncIterator[GenerationEvent]:
        """Advance generation and yield observable events as they occur."""

        request = session.request
        event_stream = session.event_stream

        try:
            while True:
                if not event_stream.has_buffered_events:
                    try:
                        await self._advance_engine()
                    except Exception:
                        await self._cancel_if_active(request)
                        self._event_broker.replace_with_failure(
                            request.request_id,
                            message="generation failed",
                        )

                event = await event_stream.receive()
                yield event

                if isinstance(
                    event,
                    (DoneEvent, CancelledEvent, FailureEvent),
                ):
                    break
        finally:
            await self._cancel_if_active(request)

            if request.request_id in self._event_broker.active_request_ids:
                self._event_broker.discard_stream(
                    request.request_id,
                )

    async def _advance_engine(self) -> None:
        """Run one shared engine iteration and publish streaming progress."""

        async with self._iteration_lock:
            await asyncio.to_thread(self.engine.step)

            for request_id in tuple(self._event_broker.active_request_ids):
                request = self.engine.scheduler.get_request(request_id)

                try:
                    self._event_broker.publish_progress(request)
                except StreamBackpressureError:
                    if request.state in (
                        RequestState.WAITING,
                        RequestState.RUNNING,
                    ):
                        self.engine.scheduler.cancel(request_id)

                    self._event_broker.replace_with_failure(
                        request_id,
                        message="stream consumer is too slow",
                    )

    async def shutdown(self) -> None:
        """Cancel active work and discard registered streams."""

        async with self._iteration_lock:
            active_requests = (
                self.engine.scheduler.waiting_requests
                + self.engine.scheduler.running_requests
            )

            for request in active_requests:
                if request.state in (
                    RequestState.WAITING,
                    RequestState.RUNNING,
                ):
                    self.engine.scheduler.cancel(request.request_id)

            for request_id in tuple(self._event_broker.active_request_ids):
                self._event_broker.discard_stream(request_id)

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
        """Cancel a failed or abandoned active request."""

        async with self._iteration_lock:
            if request.state in (
                RequestState.WAITING,
                RequestState.RUNNING,
            ):
                self.engine.scheduler.cancel(request.request_id)
