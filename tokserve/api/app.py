"""FastAPI application construction."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create the TokServe HTTP application without loading a model."""

    application = FastAPI(
        title="TokServe",
        version="0.1.0",
    )

    @application.get("/health")
    async def health() -> dict[str, str]:
        """Report that the HTTP application is available."""

        return {"status": "ok"}

    return application


app = create_app()
