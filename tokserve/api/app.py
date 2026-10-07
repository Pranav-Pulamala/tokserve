"""FastAPI application construction."""

import json
from collections.abc import AsyncIterator
from typing import cast

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse

from tokserve.api.schemas import GenerateRequest, GenerateResponse
from tokserve.api.service import (
    GenerationService,
    GenerationServiceError,
    GenerationStreamSession,
    RequestRejectedError,
)
from tokserve.api.streaming import (
    CancelledEvent,
    DoneEvent,
    FailureEvent,
    TokenEvent,
)


def create_app(
    generation_service: GenerationService | None = None,
) -> FastAPI:
    """Create a TokServe application with injectable inference state."""

    application = FastAPI(
        title="TokServe",
        version="0.1.0",
    )
    application.state.generation_service = generation_service

    @application.get("/health")
    async def health() -> dict[str, str]:
        """Report that the HTTP application is available."""

        return {"status": "ok"}

    @application.post(
        "/v1/generate",
        response_model=None,
    )
    async def generate(
        payload: GenerateRequest,
    ) -> GenerateResponse | StreamingResponse:
        """Generate a complete response or stream generation events."""

        service = cast(
            GenerationService | None,
            application.state.generation_service,
        )

        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="generation service is unavailable",
            )

        try:
            if payload.stream:
                session = await service.start_stream(payload)
                return StreamingResponse(
                    stream_sse_events(service, session),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Content-Type-Options": "nosniff",
                    },
                )

            return await service.generate(payload)
        except RequestRejectedError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        except GenerationServiceError as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="generation failed",
            ) from error

    return application


async def stream_sse_events(
    service: GenerationService,
    session: GenerationStreamSession,
) -> AsyncIterator[str]:
    """Encode one request's generation events as SSE records."""

    async for event in service.stream_events(session):
        if isinstance(event, TokenEvent):
            payload: dict[str, object] = {
                "request_id": session.request_id,
                "token_id": event.token_id,
                "generated_text": service.tokenizer.decode(
                    list(event.token_ids),
                ),
            }
        elif isinstance(event, DoneEvent):
            payload = {
                "request_id": session.request_id,
                "finish_reason": event.finish_reason,
            }
        elif isinstance(event, CancelledEvent):
            payload = {
                "request_id": session.request_id,
                "message": "generation cancelled",
            }
        elif isinstance(event, FailureEvent):
            payload = {
                "request_id": session.request_id,
                "message": event.message,
            }
        else:
            raise TypeError("unknown generation event")

        encoded = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        yield f"event: {event.kind}\ndata: {encoded}\n\n"


app = create_app()
