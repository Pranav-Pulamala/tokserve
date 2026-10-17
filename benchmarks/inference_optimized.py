"""Run parity-gated TokServe Triton inference benchmarks."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from tokserve.benchmarking.config import (
    BenchmarkConfig,
    CacheMode,
    ModelSpec,
    PromptSpec,
    WorkloadKind,
)
from tokserve.benchmarking.manifest import write_manifest
from tokserve.benchmarking.optimized import benchmark_optimized


def build_parser() -> argparse.ArgumentParser:
    """Build the optimized benchmark CLI."""

    parser = argparse.ArgumentParser(
        description=("Benchmark TokServe's parity-checked Triton backend.")
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
        "--prompt-token-id",
        type=int,
        action="append",
        dest="prompt_token_ids",
    )
    parser.add_argument(
        "--workload",
        choices=("prefill", "decode", "generation"),
        action="append",
        dest="workloads",
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
        default=1_501,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
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
    """Parse CLI arguments into an optimized configuration."""

    parsed = build_parser().parse_args(arguments)
    prompt_token_ids = tuple(
        parsed.prompt_token_ids if parsed.prompt_token_ids is not None else (1, 2, 3, 4)
    )
    workload_names = tuple(
        parsed.workloads
        if parsed.workloads is not None
        else (
            "prefill",
            "decode",
            "generation",
        )
    )
    workloads = tuple(WorkloadKind(name) for name in workload_names)
    seed = int(parsed.seed)

    return BenchmarkConfig(
        name="tokserve-triton-optimized",
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
        workloads=workloads,
        backend="triton",
        device="cuda",
        dtype="float32",
        seed=seed,
        max_output_tokens=int(parsed.max_output_tokens),
        batch_size=int(parsed.batch_size),
        concurrency=1,
        cache_mode=CacheMode.CONTIGUOUS,
        kv_block_size=int(parsed.kv_block_size),
        num_kv_blocks=int(parsed.num_kv_blocks),
        warmup_iterations=int(parsed.warmup),
        measured_repetitions=int(parsed.repetitions),
        output_path=parsed.output,
    )


def main(
    arguments: Sequence[str] | None = None,
) -> None:
    """Run and persist an optimized inference benchmark."""

    config = build_config(arguments)
    manifest = benchmark_optimized(config)
    output_path = write_manifest(manifest)

    print(f"wrote benchmark artifact: {output_path}")
    print(
        "result classification: synthetic TokServe model on CUDA; "
        "not a pretrained-model measurement"
    )
    print("parity gate: PyTorch and Triton logits/tokens matched before timing")


if __name__ == "__main__":
    main()
