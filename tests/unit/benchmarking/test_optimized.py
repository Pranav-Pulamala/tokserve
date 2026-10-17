from dataclasses import replace
from pathlib import Path

import pytest

from tokserve.benchmarking.config import (
    BenchmarkConfig,
    CacheMode,
    ModelSpec,
    PromptSpec,
    WorkloadKind,
)
from tokserve.benchmarking.optimized import (
    expected_triton_path,
    validate_optimized_config,
)


def optimized_config(
    output_path: Path,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        name="optimized-test",
        source_revision="test-revision",
        model=ModelSpec(
            identifier="tokserve-synthetic-llama",
            revision="random-weights-seed-1503",
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
        backend="triton",
        device="cuda",
        dtype="float32",
        seed=1_503,
        max_output_tokens=2,
        batch_size=1,
        concurrency=1,
        cache_mode=CacheMode.CONTIGUOUS,
        kv_block_size=2,
        num_kv_blocks=8,
        warmup_iterations=0,
        measured_repetitions=1,
        output_path=output_path,
    )


def test_optimized_configuration_is_accepted(
    tmp_path: Path,
) -> None:
    validate_optimized_config(optimized_config(tmp_path / "result.json"))


def test_expected_path_names_real_custom_kernels(
    tmp_path: Path,
) -> None:
    path = expected_triton_path(optimized_config(tmp_path / "result.json"))

    assert path == (
        "triton_rmsnorm",
        "triton_rope",
        "triton_tiled_attention",
        "contiguous_kv_cache",
    )


def test_optimized_benchmark_rejects_torch_backend(
    tmp_path: Path,
) -> None:
    config = replace(
        optimized_config(tmp_path / "result.json"),
        backend="torch",
    )

    with pytest.raises(
        ValueError,
        match="backend='triton'",
    ):
        validate_optimized_config(config)


def test_optimized_benchmark_rejects_paged_cache(
    tmp_path: Path,
) -> None:
    config = replace(
        optimized_config(tmp_path / "result.json"),
        cache_mode=CacheMode.PAGED,
    )

    with pytest.raises(
        ValueError,
        match="contiguous cache",
    ):
        validate_optimized_config(config)


def test_optimized_benchmark_rejects_serving_workload(
    tmp_path: Path,
) -> None:
    config = replace(
        optimized_config(tmp_path / "result.json"),
        workloads=(WorkloadKind.SERVING,),
    )

    with pytest.raises(
        ValueError,
        match="HTTP benchmark suite",
    ):
        validate_optimized_config(config)


def test_generation_rejects_batch_size_above_one(
    tmp_path: Path,
) -> None:
    config = replace(
        optimized_config(tmp_path / "result.json"),
        batch_size=2,
        cache_mode=CacheMode.CONTIGUOUS,
    )

    with pytest.raises(
        ValueError,
        match="batch_size=1",
    ):
        validate_optimized_config(config)
