import json
from pathlib import Path

from tokserve.benchmarking.config import (
    BenchmarkConfig,
    CacheMode,
    ModelSpec,
    PromptSpec,
    WorkloadKind,
)
from tokserve.benchmarking.inference import benchmark_baseline
from tokserve.benchmarking.manifest import write_manifest
from tokserve.benchmarking.report import build_report, write_report


def test_cpu_benchmark_artifact_to_report_pipeline(
    tmp_path: Path,
) -> None:
    """Exercise the real CPU benchmark-to-report pipeline."""

    artifact_path = tmp_path / "baseline.json"
    report_path = tmp_path / "report.json"

    config = BenchmarkConfig(
        name="tokserve-pipeline-test",
        source_revision="synthetic-test-fixture",
        model=ModelSpec(
            identifier="tokserve-synthetic-test-llama",
            revision="random-weights-seed-1510",
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
        ),
        prompt=PromptSpec.from_token_ids((1, 2, 3)),
        workloads=(
            WorkloadKind.PREFILL,
            WorkloadKind.DECODE,
            WorkloadKind.GENERATION,
        ),
        backend="torch",
        device="cpu",
        dtype="float32",
        seed=1_510,
        max_output_tokens=2,
        batch_size=1,
        concurrency=1,
        cache_mode=CacheMode.CONTIGUOUS,
        kv_block_size=2,
        num_kv_blocks=8,
        warmup_iterations=0,
        measured_repetitions=2,
        output_path=artifact_path,
    )

    manifest = benchmark_baseline(config)
    written_artifact = write_manifest(manifest)

    assert written_artifact == artifact_path
    assert artifact_path.exists()

    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact_payload["schema_version"] == "tokserve.benchmark.v1"
    assert len(artifact_payload["measurements"]) == 3

    report = build_report((artifact_path,))
    written_report = write_report(report, report_path)

    assert written_report == report_path
    assert len(report.artifacts) == 1
    assert report.comparisons == ()

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_payload["schema_version"] == ("tokserve.benchmark-report.v1")
    assert len(report_payload["artifacts"]) == 1
    assert report_payload["comparisons"] == []
    assert report_payload["resume_metrics"] == []
