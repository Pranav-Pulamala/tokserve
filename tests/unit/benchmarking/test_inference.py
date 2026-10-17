import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tokserve.benchmarking.config import (
    BenchmarkConfig,
    CacheMode,
    InputKind,
    ModelSpec,
    PromptSpec,
    WorkloadKind,
)
from tokserve.benchmarking.inference import (
    benchmark_baseline,
    create_baseline_model,
    create_prompt_ids,
)
from tokserve.benchmarking.manifest import write_manifest
from tokserve.benchmarking.timing import measure_operations


def baseline_config(
    output_path: Path,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        name="baseline-cpu-test",
        source_revision="test-revision",
        model=ModelSpec(
            identifier="tokserve-synthetic-llama",
            revision="random-weights-seed-1502",
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
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
        seed=1_502,
        max_output_tokens=2,
        batch_size=1,
        concurrency=1,
        cache_mode=CacheMode.CONTIGUOUS,
        kv_block_size=2,
        num_kv_blocks=8,
        warmup_iterations=1,
        measured_repetitions=2,
        output_path=output_path,
    )


def test_baseline_model_is_deterministic(
    tmp_path: Path,
) -> None:
    config = baseline_config(tmp_path / "result.json")

    first = create_baseline_model(config)
    second = create_baseline_model(config)

    first_parameters = tuple(first.parameters())
    second_parameters = tuple(second.parameters())

    assert len(first_parameters) == len(second_parameters)

    for first_parameter, second_parameter in zip(
        first_parameters,
        second_parameters,
        strict=True,
    ):
        torch.testing.assert_close(
            first_parameter,
            second_parameter,
            rtol=0,
            atol=0,
        )


def test_fixed_prompt_expands_to_configured_batch(
    tmp_path: Path,
) -> None:
    config = replace(
        baseline_config(tmp_path / "result.json"),
        workloads=(WorkloadKind.PREFILL,),
        batch_size=2,
    )
    prompt = create_prompt_ids(
        config,
        device=torch.device("cpu"),
    )

    assert prompt.shape == (2, 3)
    assert torch.equal(prompt[0], prompt[1])


def test_text_prompt_requires_tokenizer_adapter(
    tmp_path: Path,
) -> None:
    config = replace(
        baseline_config(tmp_path / "result.json"),
        prompt=PromptSpec(
            kind=InputKind.TEXT,
            text="hello",
            tokenizer_identifier="test-tokenizer",
            expected_token_count=1,
        ),
    )

    with pytest.raises(
        ValueError,
        match="require fixed token IDs",
    ):
        create_prompt_ids(
            config,
            device=torch.device("cpu"),
        )


def test_timing_excludes_prepare_calls() -> None:
    prepared = 0
    executed = 0

    def prepare():
        nonlocal prepared
        prepared += 1

        def operation() -> int:
            nonlocal executed
            executed += 1
            return 1

        return operation

    result = measure_operations(
        prepare,
        device=torch.device("cpu"),
        warmup_iterations=2,
        measured_repetitions=3,
    )

    assert prepared == 5
    assert executed == 5
    assert len(result.durations_seconds) == 3
    assert result.processed_tokens == (1, 1, 1)
    assert result.peak_allocated_bytes is None
    assert result.peak_reserved_bytes is None


def test_baseline_records_all_inference_workloads(
    tmp_path: Path,
) -> None:
    manifest = benchmark_baseline(baseline_config(tmp_path / "result.json"))

    assert len(manifest.measurements) == 3

    workloads = tuple(measurement["workload"] for measurement in manifest.measurements)

    assert workloads == (
        "prefill",
        "decode",
        "generation",
    )

    for measurement in manifest.measurements:
        timing = measurement["timing"]

        assert isinstance(timing, dict)
        assert timing["sample_count"] == 2
        assert len(timing["durations_seconds"]) == 2
        assert all(duration >= 0.0 for duration in timing["durations_seconds"])


def test_generation_records_actual_output_counts(
    tmp_path: Path,
) -> None:
    config = replace(
        baseline_config(tmp_path / "result.json"),
        workloads=(WorkloadKind.GENERATION,),
        warmup_iterations=0,
        measured_repetitions=3,
        max_output_tokens=2,
    )
    manifest = benchmark_baseline(config)
    measurement = manifest.measurements[0]

    assert measurement["generated_token_counts"] == [2, 2, 2]

    timing = measurement["timing"]

    assert isinstance(timing, dict)
    assert timing["processed_tokens"] == [2, 2, 2]
    assert timing["total_processed_tokens"] == 6


def test_baseline_artifact_is_machine_readable(
    tmp_path: Path,
) -> None:
    config = baseline_config(tmp_path / "result.json")
    manifest = benchmark_baseline(config)
    output_path = write_manifest(manifest)
    body = json.loads(output_path.read_text(encoding="utf-8"))

    assert body["config"]["backend"] == "torch"
    assert body["config"]["cache_mode"] == "contiguous"
    assert len(body["measurements"]) == 3
    assert body["measurements"][0]["implementation"] == ("tokserve-pytorch")


def test_baseline_rejects_triton_configuration(
    tmp_path: Path,
) -> None:
    config = replace(
        baseline_config(tmp_path / "result.json"),
        backend="triton",
        device="cuda",
    )

    with pytest.raises(
        ValueError,
        match="backend='torch'",
    ):
        benchmark_baseline(config)


def test_baseline_rejects_paged_cache(
    tmp_path: Path,
) -> None:
    config = replace(
        baseline_config(tmp_path / "result.json"),
        cache_mode=CacheMode.PAGED,
    )

    with pytest.raises(
        ValueError,
        match="requires contiguous cache",
    ):
        benchmark_baseline(config)


def test_baseline_rejects_serving_workload(
    tmp_path: Path,
) -> None:
    config = replace(
        baseline_config(tmp_path / "result.json"),
        workloads=(WorkloadKind.SERVING,),
    )

    with pytest.raises(
        ValueError,
        match="HTTP benchmark suite",
    ):
        benchmark_baseline(config)
