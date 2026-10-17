"""Parity-gated benchmarks for TokServe's Triton model backend."""

from dataclasses import dataclass, replace

import torch

from tokserve.benchmarking.config import (
    BenchmarkConfig,
    CacheMode,
    WorkloadKind,
)
from tokserve.benchmarking.inference import (
    create_prompt_ids,
    llama_config,
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
from tokserve.engine.device import resolve_device
from tokserve.engine.inference import decode, prefill
from tokserve.engine.model import (
    LlamaDecoderBlock,
    LlamaModel,
)
from tokserve.generation.cached_generate import generate_with_cache
from tokserve.generation.types import GenerationConfig

PARITY_RTOL = 1e-4
PARITY_ATOL = 1e-5


@dataclass(frozen=True)
class ParityEvidence:
    """Numerical and token parity established before measurement."""

    rtol: float
    atol: float
    prefill_max_absolute_error: float | None
    decode_max_absolute_error: float | None
    generation_tokens_match: bool | None
    reference_backend: str
    optimized_backend: str

    def __post_init__(self) -> None:
        """Reject incomplete or impossible parity evidence."""

        if self.rtol < 0.0 or self.atol < 0.0:
            raise ValueError("parity tolerances must be nonnegative")

        for error in (
            self.prefill_max_absolute_error,
            self.decode_max_absolute_error,
        ):
            if error is not None and error < 0.0:
                raise ValueError("parity errors must be nonnegative")

        if self.reference_backend != "torch":
            raise ValueError("reference backend must be torch")

        if self.optimized_backend != "triton":
            raise ValueError("optimized backend must be triton")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe parity evidence."""

        return {
            "rtol": self.rtol,
            "atol": self.atol,
            "prefill_max_absolute_error": (self.prefill_max_absolute_error),
            "decode_max_absolute_error": (self.decode_max_absolute_error),
            "generation_tokens_match": (self.generation_tokens_match),
            "reference_backend": self.reference_backend,
            "optimized_backend": self.optimized_backend,
        }


@dataclass(frozen=True)
class OptimizedMeasurement:
    """One optimized workload measurement."""

    workload: WorkloadKind
    implementation: str
    executed_path: tuple[str, ...]
    timing_scope: str
    throughput_numerator: str
    cache_mode: str
    timing: TimingResult
    generated_token_counts: tuple[int, ...]
    parity: ParityEvidence
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate path and sample metadata."""

        if self.implementation != "tokserve-triton":
            raise ValueError("optimized implementation must be tokserve-triton")

        if not self.executed_path:
            raise ValueError("executed_path must be recorded")

        if self.throughput_numerator not in (
            "input_tokens",
            "output_tokens",
        ):
            raise ValueError("unsupported throughput numerator")

        if self.workload is WorkloadKind.GENERATION:
            if len(self.generated_token_counts) != len(self.timing.durations_seconds):
                raise ValueError("generation counts must match timing samples")
        elif self.generated_token_counts:
            raise ValueError("only generation records generated-token counts")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe measurement."""

        return {
            "workload": self.workload.value,
            "implementation": self.implementation,
            "backend": "triton",
            "executed_path": list(self.executed_path),
            "timing_scope": self.timing_scope,
            "throughput_numerator": self.throughput_numerator,
            "cache_mode": self.cache_mode,
            "timing": self.timing.to_dict(),
            "generated_token_counts": list(self.generated_token_counts),
            "parity": self.parity.to_dict(),
            "notes": list(self.notes),
        }


def validate_optimized_config(
    config: BenchmarkConfig,
) -> None:
    """Require a configuration supported by the Triton model path."""

    if config.backend != "triton":
        raise ValueError("optimized benchmark requires backend='triton'")

    if config.device != "cuda":
        raise ValueError("optimized benchmark requires device='cuda'")

    if config.dtype != "float32":
        raise ValueError("current full-model Triton benchmark requires float32")

    if config.cache_mode is not CacheMode.CONTIGUOUS:
        raise ValueError(
            "Checkpoint 3 requires the contiguous cache so the "
            "baseline and optimized workloads remain equivalent"
        )

    if WorkloadKind.SERVING in config.workloads:
        raise ValueError("serving workloads belong to the HTTP benchmark suite")

    if WorkloadKind.GENERATION in config.workloads and config.batch_size != 1:
        raise ValueError("TokServe generation currently requires batch_size=1")

    if config.model.head_dim > 128:
        raise ValueError("TokServe Triton attention requires head_dim <= 128")


def expected_triton_path(
    config: BenchmarkConfig,
) -> tuple[str, ...]:
    """Describe the custom path that must execute."""

    validate_optimized_config(config)

    return (
        "triton_rmsnorm",
        "triton_rope",
        "triton_tiled_attention",
        "contiguous_kv_cache",
    )


def create_equivalent_models(
    config: BenchmarkConfig,
) -> tuple[LlamaModel, LlamaModel]:
    """Create PyTorch and Triton models with identical weights."""

    validate_optimized_config(config)
    device = resolve_device(config.device)

    torch.manual_seed(config.seed)
    reference = LlamaModel(
        llama_config(config.model),
        backend="torch",
    )
    optimized = LlamaModel(
        llama_config(config.model),
        backend="triton",
    )
    optimized.load_state_dict(reference.state_dict())

    reference.to(
        device=device,
        dtype=torch.float32,
    )
    optimized.to(
        device=device,
        dtype=torch.float32,
    )
    reference.eval()
    optimized.eval()

    _verify_triton_model(optimized)
    return reference, optimized


@torch.inference_mode()
def establish_parity(
    reference: LlamaModel,
    optimized: LlamaModel,
    prompt_ids: torch.Tensor,
    config: BenchmarkConfig,
) -> ParityEvidence:
    """Establish requested workload parity before timing."""

    prefill_error: float | None = None
    decode_error: float | None = None
    generation_match: bool | None = None
    capacity = prompt_ids.shape[1] + config.max_output_tokens

    if (
        WorkloadKind.PREFILL in config.workloads
        or WorkloadKind.DECODE in config.workloads
        or WorkloadKind.GENERATION in config.workloads
    ):
        reference_prefill = prefill(
            reference,
            prompt_ids,
            max_sequence_length=capacity,
        )
        optimized_prefill = prefill(
            optimized,
            prompt_ids,
            max_sequence_length=capacity,
        )
        _assert_logits_close(
            optimized_prefill.logits,
            reference_prefill.logits,
            workload="prefill",
        )
        prefill_error = _maximum_absolute_error(
            optimized_prefill.logits,
            reference_prefill.logits,
        )

    if WorkloadKind.DECODE in config.workloads:
        decode_ids = torch.ones(
            (config.batch_size, 1),
            device=prompt_ids.device,
            dtype=torch.int64,
        )
        reference_prefill = prefill(
            reference,
            prompt_ids,
            max_sequence_length=capacity,
        )
        optimized_prefill = prefill(
            optimized,
            prompt_ids,
            max_sequence_length=capacity,
        )
        reference_decode = decode(
            reference,
            decode_ids,
            reference_prefill.cache,
        )
        optimized_decode = decode(
            optimized,
            decode_ids,
            optimized_prefill.cache,
        )
        _assert_logits_close(
            optimized_decode,
            reference_decode,
            workload="decode",
        )
        decode_error = _maximum_absolute_error(
            optimized_decode,
            reference_decode,
        )

    if WorkloadKind.GENERATION in config.workloads:
        generation_config = GenerationConfig(
            max_new_tokens=config.max_output_tokens,
            do_sample=False,
        )
        reference_result = generate_with_cache(
            reference,
            prompt_ids,
            generation_config,
        )
        optimized_result = generate_with_cache(
            optimized,
            prompt_ids,
            generation_config,
        )
        generation_match = torch.equal(
            optimized_result.generated_token_ids,
            reference_result.generated_token_ids,
        )

        if not generation_match:
            raise AssertionError(
                "optimized generation tokens do not match the reference"
            )

    return ParityEvidence(
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
        prefill_max_absolute_error=prefill_error,
        decode_max_absolute_error=decode_error,
        generation_tokens_match=generation_match,
        reference_backend="torch",
        optimized_backend="triton",
    )


def benchmark_optimized(
    config: BenchmarkConfig,
) -> BenchmarkManifest:
    """Parity-check and benchmark TokServe's Triton backend."""

    validate_optimized_config(config)
    device = resolve_device(config.device)
    reference, optimized = create_equivalent_models(config)
    prompt_ids = create_prompt_ids(
        config,
        device=device,
    )

    parity = establish_parity(
        reference,
        optimized,
        prompt_ids,
        config,
    )
    executed_path = expected_triton_path(config)
    measurements: list[dict[str, object]] = []

    for workload in config.workloads:
        if workload is WorkloadKind.PREFILL:
            measurement = _measure_prefill(
                optimized,
                prompt_ids,
                config,
                parity,
                executed_path=executed_path,
                device=device,
            )
        elif workload is WorkloadKind.DECODE:
            measurement = _measure_decode(
                optimized,
                prompt_ids,
                config,
                parity,
                executed_path=executed_path,
                device=device,
            )
        elif workload is WorkloadKind.GENERATION:
            measurement = _measure_generation(
                optimized,
                prompt_ids,
                config,
                parity,
                executed_path=executed_path,
                device=device,
            )
        else:
            raise ValueError("serving workloads use the HTTP benchmark suite")

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
    parity: ParityEvidence,
    *,
    executed_path: tuple[str, ...],
    device: torch.device,
) -> OptimizedMeasurement:
    """Measure optimized prefill including cache allocation."""

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

    return OptimizedMeasurement(
        workload=WorkloadKind.PREFILL,
        implementation="tokserve-triton",
        executed_path=executed_path,
        timing_scope=("model prefill plus contiguous KV-cache allocation"),
        throughput_numerator="input_tokens",
        cache_mode=CacheMode.CONTIGUOUS.value,
        timing=timing,
        generated_token_counts=(),
        parity=parity,
        notes=(
            "Numerical parity was established before timing.",
            "Warmup samples are excluded.",
            "This is engine timing, not HTTP latency.",
        ),
    )


def _measure_decode(
    model: LlamaModel,
    prompt_ids: torch.Tensor,
    config: BenchmarkConfig,
    parity: ParityEvidence,
    *,
    executed_path: tuple[str, ...],
    device: torch.device,
) -> OptimizedMeasurement:
    """Measure one optimized cached decode call."""

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

        def operation() -> int:
            logits = decode(
                model,
                decode_ids,
                prepared.cache,
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

    return OptimizedMeasurement(
        workload=WorkloadKind.DECODE,
        implementation="tokserve-triton",
        executed_path=executed_path,
        timing_scope=("one cached decode model call; prefill is excluded"),
        throughput_numerator="output_tokens",
        cache_mode=CacheMode.CONTIGUOUS.value,
        timing=timing,
        generated_token_counts=(),
        parity=parity,
        notes=(
            "Each sample uses a freshly populated equivalent cache.",
            "The fixed decode token ID is 1.",
            "This is single-step engine timing.",
        ),
    )


def _measure_generation(
    model: LlamaModel,
    prompt_ids: torch.Tensor,
    config: BenchmarkConfig,
    parity: ParityEvidence,
    *,
    executed_path: tuple[str, ...],
    device: torch.device,
) -> OptimizedMeasurement:
    """Measure complete optimized greedy generation."""

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

    return OptimizedMeasurement(
        workload=WorkloadKind.GENERATION,
        implementation="tokserve-triton",
        executed_path=executed_path,
        timing_scope=("complete greedy generation with contiguous KV cache"),
        throughput_numerator="output_tokens",
        cache_mode=CacheMode.CONTIGUOUS.value,
        timing=timing,
        generated_token_counts=measured_counts,
        parity=parity,
        notes=(
            "Prompt prefill is included.",
            "Greedy output tokens match the PyTorch reference.",
            "Generated token counts are measured, not assumed.",
        ),
    )


def _verify_triton_model(
    model: LlamaModel,
) -> None:
    """Reject any model that is not configured for Triton kernels."""

    if model.backend != "triton":
        raise RuntimeError("optimized model is not using Triton")

    for layer_index, layer in enumerate(model.layers):
        if not isinstance(layer, LlamaDecoderBlock):
            raise TypeError(f"model layer {layer_index} is not a decoder block")

        if layer.backend != "triton":
            raise RuntimeError(f"model layer {layer_index} is not using Triton")

        if layer.self_attention.backend != "triton":
            raise RuntimeError(f"attention layer {layer_index} is not using Triton")


def _assert_logits_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    workload: str,
) -> None:
    """Require numerical parity before measurement."""

    try:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
        )
    except AssertionError as error:
        raise AssertionError(f"{workload} numerical parity failed") from error


def _maximum_absolute_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    """Return the largest absolute elementwise difference."""

    difference = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    return float(difference.max().item())
