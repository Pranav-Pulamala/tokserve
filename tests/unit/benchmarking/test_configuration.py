import json
from dataclasses import replace
from pathlib import Path

import pytest

from tokserve.benchmarking.config import (
    BenchmarkConfig,
    CacheMode,
    InputKind,
    ModelSpec,
    PromptSpec,
    WorkloadKind,
)
from tokserve.benchmarking.environment import (
    EnvironmentMetadata,
    capture_environment,
)
from tokserve.benchmarking.manifest import (
    BENCHMARK_SCHEMA_VERSION,
    create_manifest,
    write_manifest,
)


def model_spec() -> ModelSpec:
    return ModelSpec(
        identifier="tokserve-synthetic-llama",
        revision="test-weights-seed-1501",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


def benchmark_config(
    output_path: Path,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        name="cpu-smoke",
        source_revision="test-source-revision",
        model=model_spec(),
        prompt=PromptSpec.from_token_ids((1, 2, 3)),
        workloads=(
            WorkloadKind.PREFILL,
            WorkloadKind.DECODE,
            WorkloadKind.GENERATION,
        ),
        backend="torch",
        device="cpu",
        dtype="float32",
        seed=1_501,
        max_output_tokens=4,
        batch_size=1,
        concurrency=1,
        cache_mode=CacheMode.CONTIGUOUS,
        kv_block_size=4,
        num_kv_blocks=16,
        warmup_iterations=1,
        measured_repetitions=3,
        output_path=output_path,
    )


def environment_metadata() -> EnvironmentMetadata:
    return EnvironmentMetadata(
        python_version="3.13.0",
        operating_system="test-os",
        machine="test-machine",
        processor="test-processor",
        tokserve_version="0.1.0",
        torch_version="test-torch",
        torch_cuda_version=None,
        triton_version=None,
        requested_device="cpu",
        resolved_device="cpu",
        cuda_available=False,
        cuda_device_name=None,
        cuda_device_count=0,
        cuda_capability=None,
    )


def test_fixed_token_prompt_has_no_tokenizer_assumption() -> None:
    prompt = PromptSpec.from_token_ids((3, 5, 8))

    assert prompt.kind is InputKind.TOKEN_IDS
    assert prompt.prompt_length == 3
    assert prompt.tokenizer_identifier is None


def test_text_prompt_requires_explicit_tokenizer_and_count() -> None:
    prompt = PromptSpec.from_text(
        "hello world",
        tokenizer_identifier="example/tokenizer@revision",
        expected_token_count=2,
    )

    assert prompt.kind is InputKind.TEXT
    assert prompt.prompt_length == 2
    assert prompt.token_ids is None


def test_fixed_prompt_rejects_text_field() -> None:
    with pytest.raises(ValueError, match="cannot also contain text"):
        PromptSpec(
            kind=InputKind.TOKEN_IDS,
            token_ids=(1,),
            text="ambiguous",
        )


def test_text_prompt_rejects_token_ids() -> None:
    with pytest.raises(ValueError, match="cannot also contain token IDs"):
        PromptSpec(
            kind=InputKind.TEXT,
            text="hello",
            tokenizer_identifier="tokenizer",
            expected_token_count=1,
            token_ids=(1,),
        )


def test_empty_fixed_token_prompt_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one token"):
        PromptSpec.from_token_ids(())


def test_text_prompt_requires_tokenizer_identity() -> None:
    with pytest.raises(ValueError, match="tokenizer identifier"):
        PromptSpec(
            kind=InputKind.TEXT,
            text="hello",
            expected_token_count=1,
        )


def test_model_dimension_constraints_are_validated() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        replace(
            model_spec(),
            hidden_size=15,
        )


def test_context_overflow_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="exceed max_position_embeddings",
    ):
        replace(
            benchmark_config(tmp_path / "result.json"),
            max_output_tokens=30,
        )


def test_out_of_vocabulary_token_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        replace(
            benchmark_config(tmp_path / "result.json"),
            prompt=PromptSpec.from_token_ids((64,)),
        )


def test_decode_requires_a_cache(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires a KV cache"):
        replace(
            benchmark_config(tmp_path / "result.json"),
            cache_mode=CacheMode.NONE,
        )


def test_paged_execution_requires_batch_size_one(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="batch_size=1"):
        replace(
            benchmark_config(tmp_path / "result.json"),
            cache_mode=CacheMode.PAGED,
            batch_size=2,
        )


def test_triton_requires_cuda(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="device='cuda'"):
        replace(
            benchmark_config(tmp_path / "result.json"),
            backend="triton",
        )


def test_full_model_triton_requires_float32(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires float32"):
        replace(
            benchmark_config(tmp_path / "result.json"),
            backend="triton",
            device="cuda",
            dtype="float16",
        )


def test_triton_rejects_unsupported_head_dimension(
    tmp_path: Path,
) -> None:
    large_head_model = ModelSpec(
        identifier="large-head-test",
        revision="test",
        vocab_size=64,
        hidden_size=516,
        intermediate_size=1_024,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )

    with pytest.raises(ValueError, match="head_dim <= 128"):
        replace(
            benchmark_config(tmp_path / "result.json"),
            model=large_head_model,
            backend="triton",
            device="cuda",
        )


def test_output_destination_must_be_json(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"\.json"):
        replace(
            benchmark_config(tmp_path / "result.json"),
            output_path=tmp_path / "result.txt",
        )


def test_cpu_environment_capture_is_json_safe(
    tmp_path: Path,
) -> None:
    metadata = capture_environment(benchmark_config(tmp_path / "result.json"))
    encoded = json.dumps(metadata.to_dict())

    assert metadata.requested_device == "cpu"
    assert metadata.resolved_device == "cpu"
    assert "python_version" in encoded
    assert "torch_version" in encoded


def test_manifest_writes_versioned_empty_measurements(
    tmp_path: Path,
) -> None:
    config = benchmark_config(tmp_path / "result.json")
    manifest = create_manifest(
        config,
        environment=environment_metadata(),
        created_at_utc="2026-10-16T12:00:00Z",
    )

    written_path = write_manifest(manifest)
    body = json.loads(written_path.read_text(encoding="utf-8"))

    assert body["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert body["created_at_utc"] == "2026-10-16T12:00:00Z"
    assert body["config"]["backend"] == "torch"
    assert body["config"]["prompt"]["token_ids"] == [1, 2, 3]
    assert body["measurements"] == []


def test_manifest_refuses_to_overwrite_results(
    tmp_path: Path,
) -> None:
    config = benchmark_config(tmp_path / "result.json")
    manifest = create_manifest(
        config,
        environment=environment_metadata(),
        created_at_utc="2026-10-16T12:00:00Z",
    )
    write_manifest(manifest)

    with pytest.raises(FileExistsError, match="already exists"):
        write_manifest(manifest)
