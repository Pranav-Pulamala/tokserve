"""FastAPI application construction."""

from typing import cast

from fastapi import FastAPI, HTTPException, status

from tokserve.api.schemas import GenerateRequest, GenerateResponse
from tokserve.api.service import (
    GenerationService,
    GenerationServiceError,
    RequestRejectedError,
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
        response_model=GenerateResponse,
    )
    async def generate(payload: GenerateRequest) -> GenerateResponse:
        """Generate a complete non-streaming response."""

        if payload.stream:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="streaming is not available yet",
            )

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


app = create_app()
