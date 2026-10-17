import json
from pathlib import Path

import pytest

from tokserve.benchmarking.report import (
    REPORT_SCHEMA_VERSION,
    build_report,
    load_artifact,
    percentile,
    render_markdown,
    summarize_distribution,
    write_report,
)


def inference_artifact(
    *,
    backend: str,
    implementation: str,
    source_revision: str = "fixture-revision",
    durations: list[float] | None = None,
) -> dict[str, object]:
    """Return labeled synthetic fixture data, never real evidence."""

    samples = durations if durations is not None else [2.0, 4.0, 6.0]

    return {
        "schema_version": "tokserve.benchmark.v1",
        "created_at_utc": "2026-01-01T00:00:00Z",
        "config": {
            "name": f"synthetic-fixture-{backend}",
            "source_revision": source_revision,
            "model": {
                "identifier": "synthetic-test-fixture",
                "revision": "fixture-only",
            },
            "prompt": {
                "kind": "token_ids",
                "token_ids": [1, 2],
                "prompt_length": 2,
            },
            "workloads": ["prefill"],
            "backend": backend,
            "device": "cuda",
            "dtype": "float32",
            "seed": 1,
            "max_output_tokens": 2,
            "batch_size": 1,
            "concurrency": 1,
            "cache_mode": "contiguous",
            "kv_block_size": 2,
            "num_kv_blocks": 8,
            "warmup_iterations": 1,
            "measured_repetitions": len(samples),
            "sampling": "greedy",
            "output_path": f"/tmp/{backend}.json",
        },
        "environment": {
            "operating_system": "synthetic-os",
            "machine": "synthetic-machine",
            "processor": "synthetic-processor",
            "torch_version": "fixture",
            "torch_cuda_version": "fixture",
            "requested_device": "cuda",
            "resolved_device": "cuda",
            "cuda_device_name": "synthetic-gpu",
            "cuda_capability": "fixture",
        },
        "interpreter": {
            "implementation": "fixture",
            "executable": "fixture",
        },
        "measurements": [
            {
                "workload": "prefill",
                "implementation": implementation,
                "timing_scope": "synthetic_fixture",
                "throughput_numerator": "input_tokens",
                "backend": backend,
                "cache_mode": "contiguous",
                "timing": {
                    "durations_seconds": samples,
                    "processed_tokens": [2 for _ in samples],
                    "sample_count": len(samples),
                    "median_seconds": sorted(samples)[len(samples) // 2],
                    "total_processed_tokens": 2 * len(samples),
                    "total_measured_seconds": sum(samples),
                    "aggregate_tokens_per_second": (2 * len(samples) / sum(samples)),
                    "peak_allocated_bytes": None,
                    "peak_reserved_bytes": None,
                },
                "generated_token_counts": [],
                "notes": ["synthetic fixture only"],
            }
        ],
    }


def write_fixture(
    path: Path,
    payload: dict[str, object],
) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_percentile_uses_linear_interpolation() -> None:
    samples = (1.0, 2.0, 3.0, 4.0)

    assert percentile(samples, 0.0) == 1.0
    assert percentile(samples, 50.0) == 2.5
    assert percentile(samples, 100.0) == 4.0


def test_distribution_suppresses_small_sample_p99() -> None:
    summary = summarize_distribution((1.0, 2.0, 3.0))

    assert summary.sample_count == 3
    assert summary.median == 2.0
    assert summary.p99 is None
    assert summary.coefficient_of_variation is not None


def test_distribution_reports_p99_for_100_samples() -> None:
    samples = tuple(float(index) for index in range(1, 101))

    summary = summarize_distribution(samples)

    assert summary.p99 == pytest.approx(99.01)


def test_load_artifact_rejects_unknown_schema(tmp_path: Path) -> None:
    artifact = write_fixture(
        tmp_path / "unknown.json",
        {"schema_version": "unknown"},
    )

    with pytest.raises(ValueError, match="unsupported benchmark schema"):
        load_artifact(artifact)


def test_comparable_artifacts_produce_ratios(tmp_path: Path) -> None:
    baseline = write_fixture(
        tmp_path / "baseline.json",
        inference_artifact(
            backend="torch",
            implementation="tokserve-pytorch",
            durations=[4.0, 4.0, 4.0],
        ),
    )
    optimized = write_fixture(
        tmp_path / "optimized.json",
        inference_artifact(
            backend="triton",
            implementation="tokserve-triton",
            durations=[2.0, 2.0, 2.0],
        ),
    )

    report = build_report((baseline, optimized))
    comparison = report.comparisons[0]

    assert comparison.comparable
    assert comparison.latency_speedup == pytest.approx(2.0)
    assert comparison.throughput_speedup == pytest.approx(2.0)


def test_revision_mismatch_suppresses_ratios(tmp_path: Path) -> None:
    baseline = write_fixture(
        tmp_path / "baseline.json",
        inference_artifact(
            backend="torch",
            implementation="tokserve-pytorch",
            source_revision="revision-a",
        ),
    )
    optimized = write_fixture(
        tmp_path / "optimized.json",
        inference_artifact(
            backend="triton",
            implementation="tokserve-triton",
            source_revision="revision-b",
        ),
    )

    comparison = build_report((baseline, optimized)).comparisons[0]

    assert not comparison.comparable
    assert comparison.latency_speedup is None
    assert comparison.throughput_speedup is None
    assert "Source revisions differ." in comparison.warnings


def test_configuration_mismatch_suppresses_ratios(tmp_path: Path) -> None:
    baseline_payload = inference_artifact(
        backend="torch",
        implementation="tokserve-pytorch",
    )
    optimized_payload = inference_artifact(
        backend="triton",
        implementation="tokserve-triton",
    )
    optimized_config = optimized_payload["config"]
    assert isinstance(optimized_config, dict)
    optimized_config["dtype"] = "float16"

    baseline = write_fixture(tmp_path / "baseline.json", baseline_payload)
    optimized = write_fixture(tmp_path / "optimized.json", optimized_payload)

    comparison = build_report((baseline, optimized)).comparisons[0]

    assert not comparison.comparable
    assert comparison.latency_speedup is None


def test_report_writes_markdown_and_json(tmp_path: Path) -> None:
    baseline = write_fixture(
        tmp_path / "baseline.json",
        inference_artifact(
            backend="torch",
            implementation="tokserve-pytorch",
        ),
    )
    optimized = write_fixture(
        tmp_path / "optimized.json",
        inference_artifact(
            backend="triton",
            implementation="tokserve-triton",
        ),
    )
    report = build_report((baseline, optimized))
    markdown_path = tmp_path / "benchmark-results" / "report.md"
    json_path = tmp_path / "benchmark-results" / "report.json"

    write_report(report, markdown_path)
    write_report(report, json_path)

    assert "# TokServe Benchmark Report" in markdown_path.read_text()
    payload = json.loads(json_path.read_text())
    assert payload["schema_version"] == REPORT_SCHEMA_VERSION


def test_report_does_not_overwrite_by_default(tmp_path: Path) -> None:
    artifact = write_fixture(
        tmp_path / "baseline.json",
        inference_artifact(
            backend="torch",
            implementation="tokserve-pytorch",
        ),
    )
    report = build_report((artifact,))
    output = tmp_path / "report.md"

    write_report(report, output)

    with pytest.raises(FileExistsError):
        write_report(report, output)


def test_markdown_contains_fixture_warning(tmp_path: Path) -> None:
    artifact = write_fixture(
        tmp_path / "baseline.json",
        inference_artifact(
            backend="torch",
            implementation="tokserve-pytorch",
        ),
    )

    markdown = render_markdown(build_report((artifact,)))

    assert "not real measurements" in markdown
    assert "No baseline/optimized comparisons" in markdown
