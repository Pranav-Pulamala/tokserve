"""CPU-safe tests for reproducible profiling workloads."""

import pytest
import torch
from torch.testing import assert_close

from tokserve.profiling.workloads import (
    InferenceProfilingWorkloads,
    ProfilingConfig,
    environment_metadata,
)


def cpu_config() -> ProfilingConfig:
    """Return a very small CPU-safe workload."""

    return ProfilingConfig(
        batch_size=1,
        prompt_length=3,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=16,
        max_sequence_length=8,
        backend="torch",
        dtype=torch.float32,
        seed=1_001,
    )


def test_config_exposes_attention_dimensions() -> None:
    config = cpu_config()

    assert config.head_dim == 2
    assert config.model_config().hidden_size == 8
    assert config.model_config().num_attention_heads == 4
    assert config.model_config().num_key_value_heads == 2


def test_prompt_is_deterministic() -> None:
    first = InferenceProfilingWorkloads(
        cpu_config(),
        device=torch.device("cpu"),
    )
    second = InferenceProfilingWorkloads(
        cpu_config(),
        device=torch.device("cpu"),
    )

    assert torch.equal(first.prompt_ids, second.prompt_ids)
    assert_close(
        first.model.embed_tokens.weight,
        second.model.embed_tokens.weight,
    )


def test_prefill_and_decode_are_distinct_workloads() -> None:
    workloads = InferenceProfilingWorkloads(
        cpu_config(),
        device=torch.device("cpu"),
    )

    prefill_logits = workloads.prepare_prefill()()
    decode_logits = workloads.prepare_decode()()

    assert prefill_logits.shape == (1, 3, 16)
    assert decode_logits.shape == (1, 1, 16)


def test_triton_workload_rejects_cpu_device() -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        InferenceProfilingWorkloads(
            ProfilingConfig(
                prompt_length=3,
                max_sequence_length=8,
                backend="triton",
            ),
            device=torch.device("cpu"),
        )


def test_metadata_is_available_without_cuda() -> None:
    metadata = environment_metadata()

    assert "pytorch" in metadata
    assert "pytorch_cuda" in metadata
    assert "cuda_available" in metadata
    assert "triton" in metadata
