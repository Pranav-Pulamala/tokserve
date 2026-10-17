"""Equivalent baseline inference benchmark workloads."""

from dataclasses import dataclass, replace

import torch

from tokserve.benchmarking.config import (
    BenchmarkConfig,
    CacheMode,
    InputKind,
    ModelSpec,
    WorkloadKind,
)
from tokserve.benchmarking.manifest import (
    BenchmarkManifest,
    create_manifest,
)
from tokserve.benchmarking.timing import (
    MeasuredOperation,
    TimingResult,
    measure_operations,
)
from tokserve.engine.device import (
    resolve_device,
    resolve_dtype,
)
from tokserve.engine.inference import decode, prefill
from tokserve.engine.model import LlamaModel
from tokserve.generation.cached_generate import generate_with_cache
from tokserve.generation.types import GenerationConfig
from tokserve.reference.llama.config import LlamaConfig


@dataclass(frozen=True)
class InferenceMeasurement:
    """One benchmark workload and its raw measurements."""

    workload: WorkloadKind
    implementation: str
    timing_scope: str
    throughput_numerator: str
    backend: str
    cache_mode: str
    timing: TimingResult
    generated_token_counts: tuple[int, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate measurement sample populations."""

        if not self.implementation:
            raise ValueError("implementation must be nonempty")

        if not self.timing_scope:
            raise ValueError("timing_scope must be nonempty")

        if self.throughput_numerator not in (
            "input_tokens",
            "output_tokens",
        ):
            raise ValueError("unsupported throughput numerator")

        if self.workload is WorkloadKind.GENERATION:
            if len(self.generated_token_counts) != len(self.timing.durations_seconds):
                raise ValueError("generation counts must match timing samples")
        elif self.generated_token_counts:
            raise ValueError("only generation measurements record generated counts")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe measurement record."""

        return {
            "workload": self.workload.value,
            "implementation": self.implementation,
            "timing_scope": self.timing_scope,
            "throughput_numerator": self.throughput_numerator,
            "backend": self.backend,
            "cache_mode": self.cache_mode,
            "timing": self.timing.to_dict(),
            "generated_token_counts": list(self.generated_token_counts),
            "notes": list(self.notes),
        }


def llama_config(model: ModelSpec) -> LlamaConfig:
    """Create TokServe's model configuration from benchmark metadata."""

    return LlamaConfig(
        vocab_size=model.vocab_size,
        hidden_size=model.hidden_size,
        intermediate_size=model.intermediate_size,
        num_hidden_layers=model.num_hidden_layers,
        num_attention_heads=model.num_attention_heads,
        num_key_value_heads=model.num_key_value_heads,
        max_position_embeddings=model.max_position_embeddings,
        rms_norm_eps=model.rms_norm_eps,
        rope_theta=model.rope_theta,
    )


def create_baseline_model(
    config: BenchmarkConfig,
) -> LlamaModel:
    """Create deterministic PyTorch baseline weights."""

    if config.backend != "torch":
        raise ValueError("baseline benchmark requires backend='torch'")

    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)

    torch.manual_seed(config.seed)
    model = LlamaModel(
        llama_config(config.model),
        backend="torch",
    )
    model.to(
        device=device,
        dtype=dtype,
    )
    model.eval()
    return model


def create_prompt_ids(
    config: BenchmarkConfig,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Create the exact configured fixed-token input batch."""

    if config.prompt.kind is not InputKind.TOKEN_IDS:
        raise ValueError(
            "direct inference benchmarks require fixed token IDs; "
            "text prompts require an external tokenizer adapter"
        )

    if config.prompt.token_ids is None:
        raise RuntimeError("fixed-token prompt lost its token IDs")

    row = torch.tensor(
        config.prompt.token_ids,
        device=device,
        dtype=torch.int64,
    ).unsqueeze(0)

    return row.repeat(
        config.batch_size,
        1,
    )


def benchmark_baseline(
    config: BenchmarkConfig,
) -> BenchmarkManifest:
    """Benchmark TokServe's PyTorch contiguous-cache execution path."""

    _validate_baseline_config(config)

    device = resolve_device(config.device)
    model = create_baseline_model(config)
    prompt_ids = create_prompt_ids(
        config,
        device=device,
    )
    measurements: list[dict[str, object]] = []

    for workload in config.workloads:
        if workload is WorkloadKind.PREFILL:
            measurement = _measure_prefill(
                model,
                prompt_ids,
                config,
                device=device,
            )
        elif workload is WorkloadKind.DECODE:
            measurement = _measure_decode(
                model,
                prompt_ids,
                config,
                device=device,
            )
        elif workload is WorkloadKind.GENERATION:
            measurement = _measure_generation(
                model,
                prompt_ids,
                config,
                device=device,
            )
        else:
            raise ValueError("serving workloads use the HTTP load-test benchmark")

        measurements.append(measurement.to_dict())

    manifest = create_manifest(config)
    return replace(
        manifest,
        measurements=tuple(measurements),
    )


def _measure_prefill(
    model: LlamaModel,
    prompt_ids: torch.Tensor,
    config: BenchmarkConfig,
    *,
    device: torch.device,
) -> InferenceMeasurement:
    """Measure complete prompt prefill including cache allocation."""

    capacity = prompt_ids.shape[1] + config.max_output_tokens

    def prepare_operation() -> MeasuredOperation:
        def operation() -> int:
            result = prefill(
                model,
                prompt_ids,
                max_sequence_length=capacity,
            )

            if result.cache.current_length != prompt_ids.shape[1]:
                raise RuntimeError("prefill produced an unexpected cache length")

            return prompt_ids.numel()

        return operation

    timing = measure_operations(
        prepare_operation,
        device=device,
        warmup_iterations=config.warmup_iterations,
        measured_repetitions=config.measured_repetitions,
    )

    return InferenceMeasurement(
        workload=WorkloadKind.PREFILL,
        implementation="tokserve-pytorch",
        timing_scope=("model prefill plus contiguous KV-cache allocation"),
        throughput_numerator="input_tokens",
        backend="torch",
        cache_mode=CacheMode.CONTIGUOUS.value,
        timing=timing,
        generated_token_counts=(),
        notes=(
            "Warmup samples are excluded.",
            "CUDA synchronization occurs only at benchmark boundaries.",
            "This is engine timing, not HTTP latency.",
        ),
    )


def _measure_decode(
    model: LlamaModel,
    prompt_ids: torch.Tensor,
    config: BenchmarkConfig,
    *,
    device: torch.device,
) -> InferenceMeasurement:
    """Measure one cached decode token after untimed prefill."""

    capacity = prompt_ids.shape[1] + config.max_output_tokens
    decode_ids = torch.ones(
        (config.batch_size, 1),
        device=device,
        dtype=torch.int64,
    )

    def prepare_operation() -> MeasuredOperation:
        prepared = prefill(
            model,
            prompt_ids,
            max_sequence_length=capacity,
        )
        cache = prepared.cache

        def operation() -> int:
            logits = decode(
                model,
                decode_ids,
                cache,
            )

            if logits.shape[:2] != decode_ids.shape:
                raise RuntimeError("decode logits do not match token dimensions")

            return decode_ids.numel()

        return operation

    timing = measure_operations(
        prepare_operation,
        device=device,
        warmup_iterations=config.warmup_iterations,
        measured_repetitions=config.measured_repetitions,
    )

    return InferenceMeasurement(
        workload=WorkloadKind.DECODE,
        implementation="tokserve-pytorch",
        timing_scope=("one cached decode model call; prefill is excluded"),
        throughput_numerator="output_tokens",
        backend="torch",
        cache_mode=CacheMode.CONTIGUOUS.value,
        timing=timing,
        generated_token_counts=(),
        notes=(
            "Each sample uses a freshly populated equivalent cache.",
            "The fixed decode token ID is 1.",
            "This is single-step engine timing, not request latency.",
        ),
    )


def _measure_generation(
    model: LlamaModel,
    prompt_ids: torch.Tensor,
    config: BenchmarkConfig,
    *,
    device: torch.device,
) -> InferenceMeasurement:
    """Measure complete deterministic cached generation."""

    if config.batch_size != 1:
        raise ValueError("TokServe generation currently requires batch_size=1")

    generation_config = GenerationConfig(
        max_new_tokens=config.max_output_tokens,
        do_sample=False,
    )
    generated_counts: list[int] = []

    def prepare_operation() -> MeasuredOperation:
        def operation() -> int:
            result = generate_with_cache(
                model,
                prompt_ids,
                generation_config,
            )
            count = result.num_generated_tokens
            generated_counts.append(count)
            return count

        return operation

    timing = measure_operations(
        prepare_operation,
        device=device,
        warmup_iterations=config.warmup_iterations,
        measured_repetitions=config.measured_repetitions,
    )

    measured_counts = tuple(generated_counts[config.warmup_iterations :])

    if measured_counts != timing.processed_tokens:
        raise RuntimeError("generation token counts do not match timing samples")

    return InferenceMeasurement(
        workload=WorkloadKind.GENERATION,
        implementation="tokserve-pytorch",
        timing_scope=("complete greedy generation with contiguous KV cache"),
        throughput_numerator="output_tokens",
        backend="torch",
        cache_mode=CacheMode.CONTIGUOUS.value,
        timing=timing,
        generated_token_counts=measured_counts,
        notes=(
            "Prompt prefill is included in end-to-end generation.",
            "Greedy decoding controls output-length differences.",
            "Generated token counts are measured, not assumed.",
        ),
    )


def _validate_baseline_config(
    config: BenchmarkConfig,
) -> None:
    """Require the equivalent baseline path supported in this checkpoint."""

    if config.backend != "torch":
        raise ValueError("baseline benchmark requires backend='torch'")

    if config.cache_mode is not CacheMode.CONTIGUOUS:
        raise ValueError("baseline inference benchmark requires contiguous cache")

    if WorkloadKind.SERVING in config.workloads:
        raise ValueError("serving workloads belong to the HTTP benchmark suite")

    if WorkloadKind.GENERATION in config.workloads and config.batch_size != 1:
        raise ValueError("TokServe generation currently requires batch_size=1")
