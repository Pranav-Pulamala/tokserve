"""Run one controlled TokServe HTTP serving scenario."""

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from tokserve.benchmarking.serving import (
    ServingBenchmarkConfig,
    ServingMode,
    run_serving_benchmark,
    write_serving_result,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the serving benchmark CLI."""

    parser = argparse.ArgumentParser(
        description=("Benchmark sequential or concurrent TokServe HTTP traffic.")
    )
    parser.add_argument(
        "--source-revision",
        required=True,
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mode",
        choices=("sequential", "concurrent"),
        required=True,
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        required=True,
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
        "--warmup-requests",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--metrics-poll-seconds",
        type=float,
        default=0.005,
    )
    return parser


def build_config(
    arguments: Sequence[str] | None = None,
) -> ServingBenchmarkConfig:
    """Parse one serving scenario."""

    parsed = build_parser().parse_args(arguments)
    prompts = tuple(
        parsed.prompts
        if parsed.prompts is not None
        else (
            "1 2 3 4",
            "5 6",
            "7 8 9",
            "10",
        )
    )

    return ServingBenchmarkConfig(
        source_revision=str(parsed.source_revision),
        base_url=str(parsed.base_url),
        output_path=parsed.output,
        mode=ServingMode(str(parsed.mode)),
        prompts=prompts,
        requests=int(parsed.requests),
        concurrency=int(parsed.concurrency),
        max_new_tokens=int(parsed.max_new_tokens),
        warmup_requests=int(parsed.warmup_requests),
        timeout_seconds=float(parsed.timeout_seconds),
        metrics_poll_seconds=float(parsed.metrics_poll_seconds),
        stream=True,
        require_clean_server=True,
    )


def main(
    arguments: Sequence[str] | None = None,
) -> None:
    """Run and persist one external serving benchmark."""

    config = build_config(arguments)
    result = asyncio.run(run_serving_benchmark(config))
    output_path = write_serving_result(result)

    print(f"wrote serving benchmark: {output_path}")
    print(f"mode: {config.mode.value}")
    print(
        f"requests={result.load_test.summary.requested} "
        f"succeeded={result.load_test.summary.succeeded} "
        f"failed={result.load_test.summary.failed}"
    )
    print(
        "classification: deployment-level HTTP concurrency "
        "comparison, not isolated tensor-batching speedup"
    )


if __name__ == "__main__":
    main()
