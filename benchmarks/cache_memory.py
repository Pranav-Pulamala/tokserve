"""Run reproducible KV-cache and memory benchmarks."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from tokserve.benchmarking.cache_memory import benchmark_cache_mode
from tokserve.benchmarking.config import (
    BenchmarkConfig,
    CacheMode,
    DTypeName,
    ModelSpec,
    PromptSpec,
    WorkloadKind,
)
from tokserve.benchmarking.manifest import write_manifest
from tokserve.engine.kernel_backend import KernelBackend


def build_parser() -> argparse.ArgumentParser:
    """Build the cache benchmark CLI."""

    parser = argparse.ArgumentParser(
        description=("Benchmark uncached, contiguous, or paged TokServe execution.")
    )
    parser.add_argument(
        "--source-revision",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cache-mode",
        choices=("none", "contiguous", "paged"),
        required=True,
    )
    parser.add_argument(
        "--backend",
        choices=("torch", "triton"),
        default="torch",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--prompt-token-id",
        type=int,
        action="append",
        dest="prompt_token_ids",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1_504,
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num-key-value-heads",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--kv-block-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num-kv-blocks",
        type=int,
        default=256,
    )
    return parser


def build_config(
    arguments: Sequence[str] | None = None,
) -> BenchmarkConfig:
    """Create a validated cache benchmark configuration."""

    parsed = build_parser().parse_args(arguments)
    prompt_token_ids = tuple(
        parsed.prompt_token_ids if parsed.prompt_token_ids is not None else (1, 2, 3, 4)
    )
    dtype = cast(DTypeName, str(parsed.dtype))
    backend = cast(KernelBackend, str(parsed.backend))
    cache_mode = CacheMode(str(parsed.cache_mode))
    seed = int(parsed.seed)

    return BenchmarkConfig(
        name=f"tokserve-cache-{cache_mode.value}",
        source_revision=str(parsed.source_revision),
        model=ModelSpec(
            identifier="tokserve-synthetic-llama",
            revision=f"random-weights-seed-{seed}",
            vocab_size=int(parsed.vocab_size),
            hidden_size=int(parsed.hidden_size),
            intermediate_size=int(parsed.intermediate_size),
            num_hidden_layers=int(parsed.num_hidden_layers),
            num_attention_heads=int(parsed.num_attention_heads),
            num_key_value_heads=int(parsed.num_key_value_heads),
            max_position_embeddings=int(parsed.max_position_embeddings),
        ),
        prompt=PromptSpec.from_token_ids(prompt_token_ids),
        workloads=(WorkloadKind.GENERATION,),
        backend=backend,
        device=str(parsed.device),
        dtype=dtype,
        seed=seed,
        max_output_tokens=int(parsed.max_output_tokens),
        batch_size=1,
        concurrency=1,
        cache_mode=cache_mode,
        kv_block_size=int(parsed.kv_block_size),
        num_kv_blocks=int(parsed.num_kv_blocks),
        warmup_iterations=int(parsed.warmup),
        measured_repetitions=int(parsed.repetitions),
        output_path=parsed.output,
    )


def main(
    arguments: Sequence[str] | None = None,
) -> None:
    """Run and persist one cache-mode benchmark."""

    config = build_config(arguments)
    manifest = benchmark_cache_mode(config)
    output_path = write_manifest(manifest)

    print(f"wrote benchmark artifact: {output_path}")
    print(f"measured cache mode: {config.cache_mode.value}")
    print(
        "memory policy: theoretical capacity and measured CUDA peaks "
        "are reported separately"
    )


if __name__ == "__main__":
    main()
