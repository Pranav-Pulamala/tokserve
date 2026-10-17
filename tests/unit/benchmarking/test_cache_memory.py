from dataclasses import replace
from pathlib import Path

import pytest

from tokserve.benchmarking.cache_memory import (
    benchmark_cache_mode,
    theoretical_cache_bytes,
)
from tokserve.benchmarking.config import (
    BenchmarkConfig,
    CacheMode,
    ModelSpec,
    PromptSpec,
    WorkloadKind,
)


def cache_config(
    output_path: Path,
    *,
    mode: CacheMode,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        name=f"cache-{mode.value}-test",
        source_revision="test-revision",
        model=ModelSpec(
            identifier="tokserve-synthetic-llama",
            revision="random-weights-seed-1504",
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
        ),
        prompt=PromptSpec.from_token_ids((1, 2, 3)),
        workloads=(WorkloadKind.GENERATION,),
        backend="torch",
        device="cpu",
        dtype="float32",
        seed=1_504,
        max_output_tokens=2,
        batch_size=1,
        concurrency=1,
        cache_mode=mode,
        kv_block_size=2,
        num_kv_blocks=8,
        warmup_iterations=0,
        measured_repetitions=1,
        output_path=output_path,
    )


@pytest.mark.parametrize(
    "mode",
    (
        CacheMode.NONE,
        CacheMode.CONTIGUOUS,
        CacheMode.PAGED,
    ),
)
def test_all_cache_modes_preserve_generation(
    tmp_path: Path,
    mode: CacheMode,
) -> None:
    manifest = benchmark_cache_mode(
        cache_config(
            tmp_path / f"{mode.value}.json",
            mode=mode,
        )
    )

    assert len(manifest.measurements) == 2

    for measurement in manifest.measurements:
        parity = measurement["parity"]

        assert parity["contiguous_generation_tokens_match"] is True
        assert parity["paged_generation_tokens_match"] is True
        assert parity["generated_tokens"] == 2


def test_uncached_capacity_is_zero(
    tmp_path: Path,
) -> None:
    config = cache_config(
        tmp_path / "none.json",
        mode=CacheMode.NONE,
    )

    assert (
        theoretical_cache_bytes(
            config,
            CacheMode.NONE,
        )
        == 0
    )


def test_contiguous_capacity_matches_storage_formula(
    tmp_path: Path,
) -> None:
    config = cache_config(
        tmp_path / "contiguous.json",
        mode=CacheMode.CONTIGUOUS,
    )

    expected = (
        2
        * config.model.num_hidden_layers
        * config.batch_size
        * config.model.num_key_value_heads
        * config.model.max_position_embeddings
        * config.model.head_dim
        * 4
    )

    assert (
        theoretical_cache_bytes(
            config,
            CacheMode.CONTIGUOUS,
        )
        == expected
    )


def test_paged_capacity_matches_pool_formula(
    tmp_path: Path,
) -> None:
    config = cache_config(
        tmp_path / "paged.json",
        mode=CacheMode.PAGED,
    )

    expected = (
        2
        * config.model.num_hidden_layers
        * config.num_kv_blocks
        * config.model.num_key_value_heads
        * config.kv_block_size
        * config.model.head_dim
        * 4
    )

    assert (
        theoretical_cache_bytes(
            config,
            CacheMode.PAGED,
        )
        == expected
    )


def test_paged_lifecycle_releases_every_block(
    tmp_path: Path,
) -> None:
    manifest = benchmark_cache_mode(
        cache_config(
            tmp_path / "paged.json",
            mode=CacheMode.PAGED,
        )
    )
    lifecycle = manifest.measurements[0]["allocator_lifecycle"]

    assert lifecycle is not None
    assert lifecycle["allocated_after_prefill"] == 2
    assert lifecycle["allocated_after_decode"] == 2
    assert lifecycle["allocated_after_release"] == 0
    assert lifecycle["free_after_release"] == 8


def test_cache_benchmark_requires_batch_size_one(
    tmp_path: Path,
) -> None:
    config = replace(
        cache_config(
            tmp_path / "contiguous.json",
            mode=CacheMode.CONTIGUOUS,
        ),
        batch_size=2,
    )

    with pytest.raises(
        ValueError,
        match="batch_size=1",
    ):
        benchmark_cache_mode(config)


def test_cache_benchmark_rejects_insufficient_pool(
    tmp_path: Path,
) -> None:
    config = replace(
        cache_config(
            tmp_path / "paged.json",
            mode=CacheMode.PAGED,
        ),
        num_kv_blocks=1,
    )

    with pytest.raises(
        ValueError,
        match="does not cover",
    ):
        benchmark_cache_mode(config)
