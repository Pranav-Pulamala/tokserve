import asyncio
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from tokserve.api.benchmark_app import (
    BenchmarkServerSettings,
    create_benchmark_application,
)
from tokserve.benchmarking.serving import (
    ServingBenchmarkConfig,
    ServingMode,
    run_serving_benchmark,
)


def serving_config(
    output_path: Path,
    *,
    mode: ServingMode,
) -> ServingBenchmarkConfig:
    concurrency = 1 if mode is ServingMode.SEQUENTIAL else 2

    return ServingBenchmarkConfig(
        source_revision="test-revision",
        base_url="http://tokserve.test",
        output_path=output_path,
        mode=mode,
        prompts=(
            "1 2",
            "3 4",
        ),
        requests=2,
        concurrency=concurrency,
        max_new_tokens=1,
        warmup_requests=0,
        timeout_seconds=30.0,
        metrics_poll_seconds=0.001,
        stream=True,
        require_clean_server=True,
    )


def test_sequential_mode_requires_one_client(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="concurrency=1",
    ):
        replace(
            serving_config(
                tmp_path / "result.json",
                mode=ServingMode.SEQUENTIAL,
            ),
            concurrency=2,
        )


def test_concurrent_mode_requires_multiple_clients(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="concurrency >= 2",
    ):
        replace(
            serving_config(
                tmp_path / "result.json",
                mode=ServingMode.CONCURRENT,
            ),
            concurrency=1,
        )


def test_small_samples_suppress_p99(
    tmp_path: Path,
) -> None:
    settings = BenchmarkServerSettings(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        max_running_requests=2,
        num_kv_blocks=16,
        kv_block_size=2,
    )
    application = create_benchmark_application(settings)
    transport = httpx.ASGITransport(app=application)
    result = asyncio.run(
        run_serving_benchmark(
            serving_config(
                tmp_path / "sequential.json",
                mode=ServingMode.SEQUENTIAL,
            ),
            transport=transport,
        )
    )
    body = result.to_dict()
    load_test = body["load_test"]

    assert isinstance(load_test, dict)
    summary = load_test["summary"]

    assert isinstance(summary, dict)
    assert summary["latency_p99_seconds"] is None
    assert summary["first_token_p99_seconds"] is None
    assert result.load_test.summary.succeeded == 2
    assert result.load_test.summary.failed == 0


def test_concurrent_requests_overlap_in_engine_plan(
    tmp_path: Path,
) -> None:
    settings = BenchmarkServerSettings(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        max_running_requests=2,
        num_kv_blocks=16,
        kv_block_size=2,
    )
    application = create_benchmark_application(settings)
    transport = httpx.ASGITransport(app=application)
    result = asyncio.run(
        run_serving_benchmark(
            serving_config(
                tmp_path / "concurrent.json",
                mode=ServingMode.CONCURRENT,
            ),
            transport=transport,
        )
    )
    batching = result.server_metrics_after["batching"]

    assert isinstance(batching, dict)
    assert batching["maximum_batch_size"] >= 2
    assert result.load_test.summary.succeeded == 2
    assert result.load_test.summary.failed == 0


def test_server_must_be_restarted_between_scenarios(
    tmp_path: Path,
) -> None:
    settings = BenchmarkServerSettings(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        max_running_requests=2,
        num_kv_blocks=16,
        kv_block_size=2,
    )
    application = create_benchmark_application(settings)
    transport = httpx.ASGITransport(app=application)
    config = serving_config(
        tmp_path / "result.json",
        mode=ServingMode.SEQUENTIAL,
    )

    asyncio.run(
        run_serving_benchmark(
            config,
            transport=transport,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="server is not clean",
    ):
        asyncio.run(
            run_serving_benchmark(
                config,
                transport=transport,
            )
        )
