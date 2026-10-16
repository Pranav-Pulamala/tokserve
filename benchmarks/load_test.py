"""Concurrent external HTTP load testing for TokServe."""

import argparse
import asyncio
import codecs
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TypeAlias

import httpx

from tokserve.metrics.aggregation import percentile

Clock: TypeAlias = Callable[[], float]


@dataclass(frozen=True)
class LoadTestConfig:
    """Reproducible HTTP workload configuration."""

    base_url: str
    prompts: tuple[str, ...]
    requests: int
    concurrency: int
    max_new_tokens: int
    stream: bool
    warmup_requests: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        """Validate bounded workload settings."""

        if not self.base_url:
            raise ValueError("base_url must be nonempty")

        if not self.prompts:
            raise ValueError("at least one prompt is required")

        if any(not prompt.strip() for prompt in self.prompts):
            raise ValueError("prompts must contain non-whitespace text")

        if self.requests < 1:
            raise ValueError("requests must be positive")

        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")

        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be nonnegative")

        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class RequestObservation:
    """One client-observed HTTP request result."""

    request_index: int
    success: bool
    status_code: int | None
    latency_seconds: float
    first_token_seconds: float | None
    generated_tokens: int | None
    error: str | None


@dataclass(frozen=True)
class LoadTestSummary:
    """Aggregate client-side network-inclusive results."""

    requested: int
    succeeded: int
    failed: int
    generated_output_tokens: int
    elapsed_seconds: float
    requests_per_second: float | None
    output_tokens_per_second: float | None
    latency_p50_seconds: float | None
    latency_p95_seconds: float | None
    latency_p99_seconds: float | None
    first_token_p50_seconds: float | None
    first_token_p95_seconds: float | None
    first_token_p99_seconds: float | None


@dataclass(frozen=True)
class LoadTestResult:
    """Complete configuration, summary, and observations."""

    config: LoadTestConfig
    summary: LoadTestSummary
    observations: tuple[RequestObservation, ...]


@dataclass(frozen=True)
class SSEEvent:
    """One parsed Server-Sent Event."""

    event: str
    data: str


class SSEDecoder:
    """Incrementally decode UTF-8 and SSE frame boundaries."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""

    def feed(
        self,
        chunk: bytes,
    ) -> tuple[SSEEvent, ...]:
        """Decode every complete event in one byte chunk."""

        text = self._decoder.decode(
            chunk,
            final=False,
        )
        self._buffer += text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )
        return self._extract_events()

    def finish(self) -> tuple[SSEEvent, ...]:
        """Flush UTF-8 state and reject a partial SSE frame."""

        self._buffer += self._decoder.decode(
            b"",
            final=True,
        )
        events = self._extract_events()

        if self._buffer.strip():
            raise ValueError("stream ended with an incomplete SSE event")

        return events

    def _extract_events(
        self,
    ) -> tuple[SSEEvent, ...]:
        """Extract complete frames from the text buffer."""

        events: list[SSEEvent] = []

        while "\n\n" in self._buffer:
            frame, self._buffer = self._buffer.split(
                "\n\n",
                1,
            )
            event_name = "message"
            data_lines: list[str] = []

            for line in frame.splitlines():
                if not line or line.startswith(":"):
                    continue

                field, separator, value = line.partition(":")

                if separator and value.startswith(" "):
                    value = value[1:]

                if field == "event":
                    event_name = value
                elif field == "data":
                    data_lines.append(value)

            events.append(
                SSEEvent(
                    event=event_name,
                    data="\n".join(data_lines),
                )
            )

        return tuple(events)


async def run_load_test(
    config: LoadTestConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Clock = perf_counter,
) -> LoadTestResult:
    """Issue bounded concurrent requests to a live HTTP API."""

    timeout = httpx.Timeout(config.timeout_seconds)

    async with httpx.AsyncClient(
        base_url=config.base_url,
        timeout=timeout,
        transport=transport,
    ) as client:
        for warmup_index in range(config.warmup_requests):
            await _run_one_request(
                client,
                config,
                request_index=warmup_index,
                clock=clock,
            )

        semaphore = asyncio.Semaphore(config.concurrency)
        started = clock()

        async def run_bounded(
            request_index: int,
        ) -> RequestObservation:
            async with semaphore:
                return await _run_one_request(
                    client,
                    config,
                    request_index=request_index,
                    clock=clock,
                )

        observations = await asyncio.gather(
            *(run_bounded(request_index) for request_index in range(config.requests))
        )
        elapsed = clock() - started

    if elapsed < 0.0:
        raise ValueError("load-test clock moved backwards")

    result = LoadTestResult(
        config=config,
        summary=_summarize(
            tuple(observations),
            elapsed_seconds=elapsed,
        ),
        observations=tuple(observations),
    )
    return result


async def _run_one_request(
    client: httpx.AsyncClient,
    config: LoadTestConfig,
    *,
    request_index: int,
    clock: Clock,
) -> RequestObservation:
    """Execute one measured HTTP request without retrying."""

    prompt = config.prompts[request_index % len(config.prompts)]
    payload = {
        "prompt": prompt,
        "max_new_tokens": config.max_new_tokens,
        "stream": config.stream,
    }
    started = clock()

    try:
        if config.stream:
            return await _run_streaming_request(
                client,
                payload,
                request_index=request_index,
                started=started,
                clock=clock,
            )

        response = await client.post(
            "/v1/generate",
            json=payload,
        )
        finished = clock()
        latency = finished - started

        if latency < 0.0:
            raise ValueError("load-test clock moved backwards")

        if not response.is_success:
            return RequestObservation(
                request_index=request_index,
                success=False,
                status_code=response.status_code,
                latency_seconds=latency,
                first_token_seconds=None,
                generated_tokens=None,
                error=(f"HTTP {response.status_code}"),
            )

        body = response.json()
        generated_tokens = body.get("generated_tokens")

        if (
            not isinstance(generated_tokens, int)
            or isinstance(generated_tokens, bool)
            or generated_tokens < 0
        ):
            raise ValueError("response has no valid generated_tokens")

        return RequestObservation(
            request_index=request_index,
            success=True,
            status_code=response.status_code,
            latency_seconds=latency,
            first_token_seconds=None,
            generated_tokens=generated_tokens,
            error=None,
        )
    except (
        httpx.HTTPError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        finished = clock()
        latency = max(0.0, finished - started)

        return RequestObservation(
            request_index=request_index,
            success=False,
            status_code=None,
            latency_seconds=latency,
            first_token_seconds=None,
            generated_tokens=None,
            error=type(error).__name__,
        )


async def _run_streaming_request(
    client: httpx.AsyncClient,
    payload: dict[str, object],
    *,
    request_index: int,
    started: float,
    clock: Clock,
) -> RequestObservation:
    """Consume one SSE response and count actual token events."""

    first_token: float | None = None
    generated_tokens = 0
    received_done = False
    status_code: int | None = None
    decoder = SSEDecoder()

    async with client.stream(
        "POST",
        "/v1/generate",
        json=payload,
    ) as response:
        status_code = response.status_code

        if not response.is_success:
            await response.aread()
            finished = clock()
            return RequestObservation(
                request_index=request_index,
                success=False,
                status_code=status_code,
                latency_seconds=finished - started,
                first_token_seconds=None,
                generated_tokens=None,
                error=f"HTTP {status_code}",
            )

        async for chunk in response.aiter_bytes():
            for event in decoder.feed(chunk):
                if event.event == "token":
                    generated_tokens += 1

                    if first_token is None:
                        first_token = clock() - started
                elif event.event == "done":
                    received_done = True
                elif event.event in (
                    "error",
                    "cancelled",
                ):
                    finished = clock()
                    return RequestObservation(
                        request_index=request_index,
                        success=False,
                        status_code=status_code,
                        latency_seconds=(finished - started),
                        first_token_seconds=first_token,
                        generated_tokens=(generated_tokens),
                        error=event.event,
                    )

        for event in decoder.finish():
            if event.event == "token":
                generated_tokens += 1

                if first_token is None:
                    first_token = clock() - started
            elif event.event == "done":
                received_done = True

    finished = clock()

    return RequestObservation(
        request_index=request_index,
        success=received_done,
        status_code=status_code,
        latency_seconds=finished - started,
        first_token_seconds=first_token,
        generated_tokens=generated_tokens,
        error=(None if received_done else "missing_done_event"),
    )


def _summarize(
    observations: tuple[RequestObservation, ...],
    *,
    elapsed_seconds: float,
) -> LoadTestSummary:
    """Summarize measured requests without fabricating values."""

    succeeded = tuple(
        observation for observation in observations if observation.success
    )
    latency_samples = tuple(observation.latency_seconds for observation in observations)
    first_token_samples = tuple(
        observation.first_token_seconds
        for observation in observations
        if observation.first_token_seconds is not None
    )
    generated_tokens = sum(
        observation.generated_tokens
        for observation in succeeded
        if observation.generated_tokens is not None
    )

    requests_per_second: float | None = None
    output_tokens_per_second: float | None = None

    if elapsed_seconds > 0.0:
        requests_per_second = len(observations) / elapsed_seconds
        output_tokens_per_second = generated_tokens / elapsed_seconds

    return LoadTestSummary(
        requested=len(observations),
        succeeded=len(succeeded),
        failed=len(observations) - len(succeeded),
        generated_output_tokens=generated_tokens,
        elapsed_seconds=elapsed_seconds,
        requests_per_second=requests_per_second,
        output_tokens_per_second=(output_tokens_per_second),
        latency_p50_seconds=percentile(
            latency_samples,
            0.50,
        ),
        latency_p95_seconds=percentile(
            latency_samples,
            0.95,
        ),
        latency_p99_seconds=percentile(
            latency_samples,
            0.99,
        ),
        first_token_p50_seconds=percentile(
            first_token_samples,
            0.50,
        ),
        first_token_p95_seconds=percentile(
            first_token_samples,
            0.95,
        ),
        first_token_p99_seconds=percentile(
            first_token_samples,
            0.99,
        ),
    )


def print_summary(
    summary: LoadTestSummary,
) -> None:
    """Print a concise client-observed summary."""

    print(
        f"requests={summary.requested} "
        f"succeeded={summary.succeeded} "
        f"failed={summary.failed}"
    )
    print(f"client_elapsed_seconds={summary.elapsed_seconds:.6f}")
    print(f"generated_output_tokens={summary.generated_output_tokens}")
    print(f"requests_per_second={summary.requests_per_second}")
    print(f"output_tokens_per_second={summary.output_tokens_per_second}")
    print(
        "latency_seconds "
        f"p50={summary.latency_p50_seconds} "
        f"p95={summary.latency_p95_seconds} "
        f"p99={summary.latency_p99_seconds}"
    )
    print(
        "first_streamed_token_seconds "
        f"p50={summary.first_token_p50_seconds} "
        f"p95={summary.first_token_p95_seconds} "
        f"p99={summary.first_token_p99_seconds}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the load-test command-line parser."""

    parser = argparse.ArgumentParser(
        description=("Issue real concurrent HTTP requests to TokServe.")
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--stream",
        action="store_true",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    return parser


def main(
    arguments: Sequence[str] | None = None,
) -> None:
    """Run the load-test CLI."""

    parsed = build_parser().parse_args(arguments)
    prompts = tuple(
        parsed.prompts
        if parsed.prompts
        else (
            "1 2 3",
            "4 5",
            "6",
        )
    )
    config = LoadTestConfig(
        base_url=parsed.base_url,
        prompts=prompts,
        requests=parsed.requests,
        concurrency=parsed.concurrency,
        max_new_tokens=parsed.max_new_tokens,
        stream=parsed.stream,
        warmup_requests=parsed.warmup_requests,
        timeout_seconds=parsed.timeout_seconds,
    )
    result = asyncio.run(run_load_test(config))
    print_summary(result.summary)

    if parsed.output is not None:
        parsed.output.write_text(
            json.dumps(
                asdict(result),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
