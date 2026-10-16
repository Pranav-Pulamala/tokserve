import asyncio
import json

import httpx

from benchmarks.load_test import (
    LoadTestConfig,
    SSEDecoder,
    run_load_test,
)


def create_config(
    *,
    requests: int = 3,
    concurrency: int = 2,
    stream: bool = False,
    warmup_requests: int = 0,
) -> LoadTestConfig:
    return LoadTestConfig(
        base_url="http://test",
        prompts=("1 2", "3"),
        requests=requests,
        concurrency=concurrency,
        max_new_tokens=2,
        stream=stream,
        warmup_requests=warmup_requests,
        timeout_seconds=1.0,
    )


def test_sse_decoder_handles_split_utf8_and_frames() -> None:
    decoder = SSEDecoder()
    payload = (
        "event: token\n"
        'data: {"generated_text":"café"}\n\n'
        "event: done\n"
        'data: {"finish_reason":"max_new_tokens"}\n\n'
    ).encode()

    split = payload.index("é".encode()) + 1
    first = decoder.feed(payload[:split])
    second = decoder.feed(payload[split:])
    final = decoder.finish()
    events = (*first, *second, *final)

    assert len(events) == 2
    assert events[0].event == "token"
    assert "café" in events[0].data
    assert events[1].event == "done"


def test_non_streaming_load_counts_real_tokens() -> None:
    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.url.path == "/v1/generate"
        return httpx.Response(
            200,
            json={
                "request_id": "request",
                "generated_text": "7 8",
                "generated_tokens": 2,
                "finish_reason": "max_new_tokens",
            },
        )

    result = asyncio.run(
        run_load_test(
            create_config(),
            transport=httpx.MockTransport(handler),
        )
    )

    assert result.summary.requested == 3
    assert result.summary.succeeded == 3
    assert result.summary.failed == 0
    assert result.summary.generated_output_tokens == 6


def test_streaming_load_counts_token_events() -> None:
    body = (
        "event: token\n"
        'data: {"token_id":7}\n\n'
        "event: token\n"
        'data: {"token_id":8}\n\n'
        "event: done\n"
        'data: {"finish_reason":"max_new_tokens"}\n\n'
    )

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    result = asyncio.run(
        run_load_test(
            create_config(
                requests=2,
                stream=True,
            ),
            transport=httpx.MockTransport(handler),
        )
    )

    assert result.summary.succeeded == 2
    assert result.summary.failed == 0
    assert result.summary.generated_output_tokens == 4
    assert result.summary.first_token_p50_seconds is not None


def test_http_failures_are_not_dropped() -> None:
    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            500,
            json={"detail": "failed"},
        )

    result = asyncio.run(
        run_load_test(
            create_config(),
            transport=httpx.MockTransport(handler),
        )
    )

    assert result.summary.requested == 3
    assert result.summary.succeeded == 0
    assert result.summary.failed == 3
    assert len(result.observations) == 3


def test_timeouts_are_reported() -> None:
    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise httpx.ReadTimeout(
            "timed out",
            request=request,
        )

    result = asyncio.run(
        run_load_test(
            create_config(requests=1),
            transport=httpx.MockTransport(handler),
        )
    )

    assert result.summary.failed == 1
    assert result.observations[0].error == "ReadTimeout"


def test_concurrency_never_exceeds_cap() -> None:
    active = 0
    maximum_active = 0
    lock = asyncio.Lock()

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal active
        nonlocal maximum_active

        async with lock:
            active += 1
            maximum_active = max(
                maximum_active,
                active,
            )

        await asyncio.sleep(0.01)

        async with lock:
            active -= 1

        return httpx.Response(
            200,
            json={
                "request_id": "request",
                "generated_text": "7",
                "generated_tokens": 1,
                "finish_reason": "max_new_tokens",
            },
        )

    result = asyncio.run(
        run_load_test(
            create_config(
                requests=6,
                concurrency=2,
            ),
            transport=httpx.MockTransport(handler),
        )
    )

    assert result.summary.succeeded == 6
    assert maximum_active == 2


def test_warmup_is_excluded_from_results() -> None:
    calls = 0

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1

        return httpx.Response(
            200,
            json={
                "request_id": "request",
                "generated_text": "7",
                "generated_tokens": 1,
                "finish_reason": "max_new_tokens",
            },
        )

    result = asyncio.run(
        run_load_test(
            create_config(
                requests=3,
                warmup_requests=2,
            ),
            transport=httpx.MockTransport(handler),
        )
    )

    assert calls == 5
    assert result.summary.requested == 3
    assert len(result.observations) == 3


def test_invalid_generated_token_count_fails() -> None:
    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps(
                {
                    "generated_text": "not counted",
                }
            ),
        )

    result = asyncio.run(
        run_load_test(
            create_config(requests=1),
            transport=httpx.MockTransport(handler),
        )
    )

    assert result.summary.succeeded == 0
    assert result.summary.failed == 1
    assert result.summary.generated_output_tokens == 0
