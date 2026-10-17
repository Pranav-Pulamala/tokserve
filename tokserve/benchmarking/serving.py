"""Deployment-level sequential and concurrent HTTP benchmarks."""

import asyncio
import json
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import httpx

from benchmarks.load_test import (
    LoadTestConfig,
    LoadTestResult,
    run_load_test,
)

SERVING_SCHEMA_VERSION = "tokserve.serving-benchmark.v1"


class ServingMode(StrEnum):
    """Client arrival pattern for one serving benchmark."""

    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"


@dataclass(frozen=True)
class ServingBenchmarkConfig:
    """One reproducible external HTTP workload."""

    source_revision: str
    base_url: str
    output_path: Path
    mode: ServingMode
    prompts: tuple[str, ...]
    requests: int
    concurrency: int
    max_new_tokens: int
    warmup_requests: int
    timeout_seconds: float
    metrics_poll_seconds: float = 0.005
    stream: bool = True
    require_clean_server: bool = True

    def __post_init__(self) -> None:
        """Validate fair sequential or concurrent arrivals."""

        if not self.source_revision.strip():
            raise ValueError("source_revision must be nonempty")

        if not self.base_url.strip():
            raise ValueError("base_url must be nonempty")

        if self.output_path.suffix.lower() != ".json":
            raise ValueError("output_path must end in .json")

        if not self.prompts:
            raise ValueError("at least one prompt is required")

        if any(not prompt.strip() for prompt in self.prompts):
            raise ValueError("prompts must be nonempty")

        if self.requests < 1:
            raise ValueError("requests must be positive")

        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")

        if self.concurrency > self.requests:
            raise ValueError("concurrency cannot exceed requests")

        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be nonnegative")

        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")

        if self.metrics_poll_seconds <= 0.0:
            raise ValueError("metrics_poll_seconds must be positive")

        if not self.stream:
            raise ValueError("serving benchmark requires streaming for client TTFT")

        if self.mode is ServingMode.SEQUENTIAL and self.concurrency != 1:
            raise ValueError("sequential mode requires concurrency=1")

        if self.mode is ServingMode.CONCURRENT and self.concurrency < 2:
            raise ValueError("concurrent mode requires concurrency >= 2")

    def to_load_test_config(self) -> LoadTestConfig:
        """Create the existing M14 load-test configuration."""

        return LoadTestConfig(
            base_url=self.base_url,
            prompts=self.prompts,
            requests=self.requests,
            concurrency=self.concurrency,
            max_new_tokens=self.max_new_tokens,
            stream=True,
            warmup_requests=self.warmup_requests,
            timeout_seconds=self.timeout_seconds,
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe client configuration."""

        return {
            "source_revision": self.source_revision,
            "base_url": self.base_url,
            "mode": self.mode.value,
            "prompts": list(self.prompts),
            "requests": self.requests,
            "concurrency": self.concurrency,
            "max_new_tokens": self.max_new_tokens,
            "warmup_requests": self.warmup_requests,
            "timeout_seconds": self.timeout_seconds,
            "metrics_poll_seconds": self.metrics_poll_seconds,
            "stream": self.stream,
            "require_clean_server": self.require_clean_server,
            "output_path": str(self.output_path),
            "arrival_policy": (
                "bounded client concurrency; all measured tasks created together"
            ),
        }


@dataclass(frozen=True)
class PressureObservation:
    """Maximum live server pressure observed while requests ran."""

    samples: int
    maximum_waiting_requests: int
    maximum_running_requests: int
    maximum_allocated_kv_blocks: int
    maximum_kv_utilization_ratio: float

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe pressure evidence."""

        return {
            "samples": self.samples,
            "maximum_waiting_requests": (self.maximum_waiting_requests),
            "maximum_running_requests": (self.maximum_running_requests),
            "maximum_allocated_kv_blocks": (self.maximum_allocated_kv_blocks),
            "maximum_kv_utilization_ratio": (self.maximum_kv_utilization_ratio),
        }


@dataclass(frozen=True)
class ServingBenchmarkResult:
    """Client observations plus matching server evidence."""

    config: ServingBenchmarkConfig
    server_config: dict[str, object]
    server_metrics_before: dict[str, object]
    server_metrics_after: dict[str, object]
    pressure: PressureObservation
    load_test: LoadTestResult
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a versioned machine-readable artifact."""

        load_test = asdict(self.load_test)
        summary = cast(
            dict[str, object],
            load_test["summary"],
        )

        if self.config.requests < 100:
            summary["latency_p99_seconds"] = None
            summary["first_token_p99_seconds"] = None

        return {
            "schema_version": SERVING_SCHEMA_VERSION,
            "benchmark_kind": ("deployment-level HTTP concurrency comparison"),
            "config": self.config.to_dict(),
            "client_environment": {
                "python_version": platform.python_version(),
                "operating_system": platform.platform(),
            },
            "server_config": self.server_config,
            "server_metrics_before": (self.server_metrics_before),
            "server_metrics_after": self.server_metrics_after,
            "pressure": self.pressure.to_dict(),
            "load_test": load_test,
            "warnings": list(self.warnings),
        }


async def run_serving_benchmark(
    config: ServingBenchmarkConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ServingBenchmarkResult:
    """Run one bounded HTTP workload against a clean server."""

    before = await _fetch_json(
        config.base_url,
        "/metrics",
        transport=transport,
        timeout=config.timeout_seconds,
    )
    server_config = await _fetch_json(
        config.base_url,
        "/benchmark/config",
        transport=transport,
        timeout=config.timeout_seconds,
    )

    if config.require_clean_server:
        _require_clean_metrics(before)

    stop_polling = asyncio.Event()
    pressure_samples: list[dict[str, object]] = []
    polling_task = asyncio.create_task(
        _poll_metrics(
            config,
            pressure_samples,
            stop_polling,
            transport=transport,
        )
    )

    try:
        load_result = await run_load_test(
            config.to_load_test_config(),
            transport=transport,
        )
    finally:
        stop_polling.set()
        await polling_task

    after = await _fetch_json(
        config.base_url,
        "/metrics",
        transport=transport,
        timeout=config.timeout_seconds,
    )
    pressure_samples.append(after)
    pressure = _summarize_pressure(pressure_samples)

    _validate_outcome_reconciliation(
        config,
        load_result,
        before,
        after,
    )

    warnings = [
        (
            "This is a deployment-level arrival-pattern comparison, "
            "not an isolated tensor-batching speedup."
        ),
        (
            "TokServe currently executes requests within each planned "
            "iteration individually."
        ),
        (
            "Client latency includes HTTP, serialization, scheduling, "
            "queueing, and inference."
        ),
        (
            "Server TTFT is acceptance-to-first-token; client TTFT is "
            "request-start-to-first-SSE-token."
        ),
    ]

    if config.requests < 100:
        warnings.append(
            "p99 is suppressed because fewer than 100 measured requests were collected."
        )

    if config.mode is ServingMode.CONCURRENT:
        batching = _nested(after, "batching")
        maximum_batch_size = _integer(
            batching,
            "maximum_batch_size",
        )

        if load_result.summary.succeeded >= 2 and maximum_batch_size < 2:
            raise RuntimeError(
                "concurrent requests did not overlap in an engine iteration"
            )

    return ServingBenchmarkResult(
        config=config,
        server_config=server_config,
        server_metrics_before=before,
        server_metrics_after=after,
        pressure=pressure,
        load_test=load_result,
        warnings=tuple(warnings),
    )


def write_serving_result(
    result: ServingBenchmarkResult,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one serving benchmark artifact."""

    output_path = result.config.output_path

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"benchmark output already exists: {output_path}")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            result.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


async def _poll_metrics(
    config: ServingBenchmarkConfig,
    samples: list[dict[str, object]],
    stop: asyncio.Event,
    *,
    transport: httpx.AsyncBaseTransport | None,
) -> None:
    """Capture live scheduler and KV pressure during the workload."""

    while not stop.is_set():
        sample = await _fetch_json(
            config.base_url,
            "/metrics",
            transport=transport,
            timeout=config.timeout_seconds,
        )
        samples.append(sample)

        try:
            await asyncio.wait_for(
                stop.wait(),
                timeout=config.metrics_poll_seconds,
            )
        except TimeoutError:
            pass


async def _fetch_json(
    base_url: str,
    path: str,
    *,
    transport: httpx.AsyncBaseTransport | None,
    timeout: float,
) -> dict[str, object]:
    """Fetch and validate one JSON object."""

    async with httpx.AsyncClient(
        base_url=base_url,
        transport=transport,
        timeout=timeout,
    ) as client:
        response = await client.get(path)
        response.raise_for_status()

    value: object = json.loads(response.text)

    if not isinstance(value, dict):
        raise ValueError(f"{path} did not return a JSON object")

    return cast(dict[str, object], value)


def _require_clean_metrics(
    metrics: dict[str, object],
) -> None:
    """Require a fresh server for fair scenario isolation."""

    requests = _nested(metrics, "requests")

    for name in (
        "received_total",
        "admitted_total",
        "completed_total",
        "failed_total",
        "cancelled_total",
        "active_metrics_records",
        "waiting_current",
        "running_current",
    ):
        if _integer(requests, name) != 0:
            raise RuntimeError(
                "benchmark server is not clean; restart it before "
                "running another scenario"
            )


def _validate_outcome_reconciliation(
    config: ServingBenchmarkConfig,
    result: LoadTestResult,
    before: dict[str, object],
    after: dict[str, object],
) -> None:
    """Reconcile client outcomes with cumulative server counters."""

    before_requests = _nested(before, "requests")
    after_requests = _nested(after, "requests")
    expected_received = config.requests + config.warmup_requests
    received_delta = _integer(after_requests, "received_total") - _integer(
        before_requests, "received_total"
    )

    if received_delta != expected_received:
        raise RuntimeError(
            "server received count does not match measured plus warmup requests"
        )

    completed_delta = _integer(after_requests, "completed_total") - _integer(
        before_requests, "completed_total"
    )
    failed_delta = _integer(after_requests, "failed_total") - _integer(
        before_requests, "failed_total"
    )
    cancelled_delta = _integer(after_requests, "cancelled_total") - _integer(
        before_requests, "cancelled_total"
    )

    if completed_delta < result.summary.succeeded:
        raise RuntimeError("server completed fewer requests than the client observed")

    if failed_delta + cancelled_delta < result.summary.failed:
        raise RuntimeError("server failure outcomes do not reconcile with the client")


def _summarize_pressure(
    samples: Sequence[dict[str, object]],
) -> PressureObservation:
    """Reduce live metrics polling to maximum observed pressure."""

    waiting: list[int] = []
    running: list[int] = []
    allocated: list[int] = []
    utilization: list[float] = []

    for sample in samples:
        requests = _nested(sample, "requests")
        cache = _nested(sample, "kv_cache")
        waiting.append(_integer(requests, "waiting_current"))
        running.append(_integer(requests, "running_current"))
        allocated.append(_integer(cache, "allocated_blocks"))
        utilization.append(_floating(cache, "utilization_ratio"))

    return PressureObservation(
        samples=len(samples),
        maximum_waiting_requests=max(waiting, default=0),
        maximum_running_requests=max(running, default=0),
        maximum_allocated_kv_blocks=max(allocated, default=0),
        maximum_kv_utilization_ratio=max(
            utilization,
            default=0.0,
        ),
    )


def _nested(
    mapping: dict[str, object],
    key: str,
) -> dict[str, object]:
    """Return one required nested JSON object."""

    value = mapping.get(key)

    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")

    return cast(dict[str, object], value)


def _integer(
    mapping: dict[str, object],
    key: str,
) -> int:
    """Return one required integer JSON field."""

    value = mapping.get(key)

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")

    return value


def _floating(
    mapping: dict[str, object],
    key: str,
) -> float:
    """Return one required numeric JSON field."""

    value = mapping.get(key)

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")

    return float(value)
