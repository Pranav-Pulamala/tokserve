import importlib.util
from pathlib import Path

import pytest
import torch

from tokserve.benchmarking.config import (
    BenchmarkConfig,
    CacheMode,
    ModelSpec,
    PromptSpec,
    WorkloadKind,
)
from tokserve.benchmarking.optimized import benchmark_optimized

CUDA_TRITON_AVAILABLE = (
    torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not CUDA_TRITON_AVAILABLE,
        reason="CUDA and Triton are required",
    ),
]


def gpu_config(
    output_path: Path,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        name="optimized-gpu-test",
        source_revision="test-revision",
        model=ModelSpec(
            identifier="tokserve-synthetic-llama",
            revision="random-weights-seed-1504",
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
        seed=1_504,
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


def test_optimized_benchmark_establishes_parity_and_runs(
    tmp_path: Path,
) -> None:
    manifest = benchmark_optimized(gpu_config(tmp_path / "result.json"))

    assert len(manifest.measurements) == 3

    for measurement in manifest.measurements:
        assert measurement["implementation"] == "tokserve-triton"
        assert measurement["backend"] == "triton"
        assert "triton_rmsnorm" in measurement["executed_path"]
        assert "triton_rope" in measurement["executed_path"]
        assert "triton_tiled_attention" in (measurement["executed_path"])

        parity = measurement["parity"]

        assert parity["reference_backend"] == "torch"
        assert parity["optimized_backend"] == "triton"

    generation = manifest.measurements[2]

    assert generation["generated_token_counts"] == [2]
    assert generation["parity"]["generation_tokens_match"] is True
