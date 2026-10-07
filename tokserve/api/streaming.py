"""Bounded per-request generation event streams."""

import asyncio
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from tokserve.generation.types import StopReason
from tokserve.scheduler.request import RequestState, ScheduledRequest


@dataclass(frozen=True)
class TokenEvent:
    """One newly generated token and its complete generated prefix."""

    token_id: int
    token_ids: tuple[int, ...]
    kind: Literal["token"] = field(default="token", init=False)


@dataclass(frozen=True)
class DoneEvent:
    """Successful terminal generation event."""

    finish_reason: StopReason
    kind: Literal["done"] = field(default="done", init=False)


@dataclass(frozen=True)
class CancelledEvent:
    """Request cancellation event."""

    kind: Literal["cancelled"] = field(
        default="cancelled",
        init=False,
    )


@dataclass(frozen=True)
class FailureEvent:
    """Sanitized request failure event."""

    message: str
    kind: Literal["error"] = field(default="error", init=False)


GenerationEvent: TypeAlias = TokenEvent | DoneEvent | CancelledEvent | FailureEvent


class StreamBackpressureError(RuntimeError):
    """Raised when a request's bounded event buffer is full."""


class RequestEventStream:
    """Asynchronous consumer side of one request event queue."""

    def __init__(
        self,
        request_id: str,
        *,
        buffer_size: int,
    ) -> None:
        if not request_id:
            raise ValueError("request_id must be nonempty")

        if buffer_size < 1:
            raise ValueError("buffer_size must be positive")

        self.request_id = request_id
        self._queue: asyncio.Queue[GenerationEvent] = asyncio.Queue(maxsize=buffer_size)
        self._terminal_published = False

    @property
    def buffer_size(self) -> int:
        """Return the maximum number of buffered events."""

        return self._queue.maxsize

    @property
    def terminal_published(self) -> bool:
        """Return whether a terminal event has been published."""

        return self._terminal_published

    async def receive(self) -> GenerationEvent:
        """Wait without polling for the next event."""

        return await self._queue.get()

    def _publish(self, event: GenerationEvent) -> None:
        """Publish one event without blocking the inference loop."""

        if self._terminal_published:
            raise RuntimeError("stream already received a terminal event")

        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull as error:
            raise StreamBackpressureError("request event buffer is full") from error

        if isinstance(
            event,
            (DoneEvent, CancelledEvent, FailureEvent),
        ):
            self._terminal_published = True


@dataclass
class _Registration:
    """Internal request-stream publication state."""

    stream: RequestEventStream
    published_tokens: int = 0


class RequestEventBroker:
    """Manage isolated event streams for active requests."""

    def __init__(self, *, buffer_size: int = 16) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be positive")

        self.buffer_size = buffer_size
        self._registrations: dict[str, _Registration] = {}

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        """Return registered stream IDs in insertion order."""

        return tuple(self._registrations)

    def open_stream(self, request_id: str) -> RequestEventStream:
        """Register and return one request's event stream."""

        if request_id in self._registrations:
            raise ValueError("request stream is already registered")

        stream = RequestEventStream(
            request_id,
            buffer_size=self.buffer_size,
        )
        self._registrations[request_id] = _Registration(stream)
        return stream

    def publish_progress(self, request: ScheduledRequest) -> None:
        """Publish new tokens and any terminal request state."""

        registration = self._get_registration(request.request_id)
        generated_ids = tuple(
            int(token_id) for token_id in request.generated_token_ids[0].tolist()
        )

        while registration.published_tokens < len(generated_ids):
            token_index = registration.published_tokens
            registration.stream._publish(
                TokenEvent(
                    token_id=generated_ids[token_index],
                    token_ids=generated_ids[: token_index + 1],
                )
            )
            registration.published_tokens += 1

        if request.state is RequestState.FINISHED:
            result = request.result

            if result is None:
                raise RuntimeError("finished request has no generation result")

            if not registration.stream.terminal_published:
                registration.stream._publish(
                    DoneEvent(finish_reason=result.stop_reason)
                )
        elif request.state is RequestState.CANCELLED:
            if not registration.stream.terminal_published:
                registration.stream._publish(CancelledEvent())

    def publish_failure(
        self,
        request_id: str,
        *,
        message: str = "generation failed",
    ) -> None:
        """Publish one sanitized failure event."""

        registration = self._get_registration(request_id)

        if not registration.stream.terminal_published:
            registration.stream._publish(FailureEvent(message=message))

    def remove_stream(self, request_id: str) -> None:
        """Remove one registered terminal stream."""

        registration = self._get_registration(request_id)

        if not registration.stream.terminal_published:
            raise RuntimeError("cannot remove a nonterminal request stream")

        del self._registrations[request_id]

    def _get_registration(self, request_id: str) -> _Registration:
        """Return one stream registration."""

        try:
            return self._registrations[request_id]
        except KeyError as error:
            raise KeyError("unknown request stream") from error
