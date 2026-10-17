"""KV-cache execution, lifecycle, timing, and memory benchmarks."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TypeAlias

import torch

from tokserve.benchmarking.config import (
    BenchmarkConfig,
    CacheMode,
    InputKind,
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
    synchronize_device,
)
from tokserve.engine.device import (
    resolve_device,
    resolve_dtype,
)
from tokserve.engine.inference import decode, prefill
from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.inference import (
    decode_paged,
    prefill_paged,
)
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.generation.cached_generate import generate_with_cache
from tokserve.generation.generate import generate
from tokserve.generation.paged_generate import (
    generate_with_paged_cache,
)
from tokserve.generation.types import GenerationConfig

PreparedOperation: TypeAlias = tuple[
    MeasuredOperation,
    Callable[[], None],
]
PrepareWithCleanup: TypeAlias = Callable[[], PreparedOperation]

PARITY_RTOL = 1e-4
PARITY_ATOL = 1e-5


@dataclass(frozen=True)
class CacheLifecycleEvidence:
    """Physical paged-cache ownership observed outside timing."""

    capacity_blocks: int
    allocated_after_prefill: int
    allocated_after_decode: int
    allocated_after_release: int
    free_after_release: int

    def __post_init__(self) -> None:
        """Reject impossible allocator observations."""

        values = (
            self.capacity_blocks,
            self.allocated_after_prefill,
            self.allocated_after_decode,
            self.allocated_after_release,
            self.free_after_release,
        )

        if any(value < 0 for value in values):
            raise ValueError("cache lifecycle values must be nonnegative")

        if self.capacity_blocks < 1:
            raise ValueError("cache capacity must be positive")

        if self.allocated_after_prefill > self.capacity_blocks:
            raise ValueError("prefill allocation exceeds capacity")

        if self.allocated_after_decode > self.capacity_blocks:
            raise ValueError("decode allocation exceeds capacity")

        if self.allocated_after_release != 0:
            raise ValueError("release must return every physical block")

        if self.free_after_release != self.capacity_blocks:
            raise ValueError("released allocator must be completely free")

    def to_dict(self) -> dict[str, int]:
        """Return JSON-safe lifecycle evidence."""

        return {
            "capacity_blocks": self.capacity_blocks,
            "allocated_after_prefill": self.allocated_after_prefill,
            "allocated_after_decode": self.allocated_after_decode,
            "allocated_after_release": self.allocated_after_release,
            "free_after_release": self.free_after_release,
        }


@dataclass(frozen=True)
class CacheParityEvidence:
    """Equivalence of uncached, contiguous, and paged execution."""

    rtol: float
    atol: float
    contiguous_decode_max_absolute_error: float
    paged_decode_max_absolute_error: float
    contiguous_generation_tokens_match: bool
    paged_generation_tokens_match: bool
    generated_tokens: int

    def __post_init__(self) -> None:
        """Reject failed or incomplete parity evidence."""

        if self.rtol < 0.0 or self.atol < 0.0:
            raise ValueError("parity tolerances must be nonnegative")

        if self.contiguous_decode_max_absolute_error < 0.0:
            raise ValueError("contiguous parity error must be nonnegative")

        if self.paged_decode_max_absolute_error < 0.0:
            raise ValueError("paged parity error must be nonnegative")

        if not self.contiguous_generation_tokens_match:
            raise ValueError("contiguous generation must match uncached generation")

        if not self.paged_generation_tokens_match:
            raise ValueError("paged generation must match uncached generation")

        if self.generated_tokens < 0:
            raise ValueError("generated token count must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe parity evidence."""

        return {
            "rtol": self.rtol,
            "atol": self.atol,
            "contiguous_decode_max_absolute_error": (
                self.contiguous_decode_max_absolute_error
            ),
            "paged_decode_max_absolute_error": (self.paged_decode_max_absolute_error),
            "contiguous_generation_tokens_match": (
                self.contiguous_generation_tokens_match
            ),
            "paged_generation_tokens_match": (self.paged_generation_tokens_match),
            "generated_tokens": self.generated_tokens,
        }


@dataclass(frozen=True)
class CacheMeasurement:
    """Timing and memory evidence for one cache mode."""

    operation: str
    cache_mode: CacheMode
    timing_scope: str
    timing: TimingResult
    theoretical_capacity_bytes: int
    parity: CacheParityEvidence
    lifecycle: CacheLifecycleEvidence | None
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate cache measurement metadata."""

        if self.operation not in (
            "decode_step",
            "generation",
        ):
            raise ValueError("unsupported cache operation")

        if self.theoretical_capacity_bytes < 0:
            raise ValueError("theoretical cache capacity must be nonnegative")

        if self.cache_mode is CacheMode.NONE and self.theoretical_capacity_bytes != 0:
            raise ValueError("uncached execution cannot report cache capacity")

        if self.cache_mode is CacheMode.PAGED and self.lifecycle is None:
            raise ValueError("paged measurements require lifecycle evidence")

        if self.cache_mode is not CacheMode.PAGED and self.lifecycle is not None:
            raise ValueError(
                "only paged measurements have allocator lifecycle evidence"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe measurement."""

        return {
            "workload": "cache_memory",
            "operation": self.operation,
            "cache_mode": self.cache_mode.value,
            "timing_scope": self.timing_scope,
            "timing": self.timing.to_dict(),
            "theoretical_capacity_bytes": (self.theoretical_capacity_bytes),
            "parity": self.parity.to_dict(),
            "allocator_lifecycle": (
                self.lifecycle.to_dict() if self.lifecycle is not None else None
            ),
            "notes": list(self.notes),
        }


def theoretical_cache_bytes(
    config: BenchmarkConfig,
    mode: CacheMode,
) -> int:
    """Return tensor-storage capacity without claiming measured usage."""

    if mode is CacheMode.NONE:
        return 0

    dtype = resolve_dtype(config.dtype)
    element_size = torch.empty(
        (),
        dtype=dtype,
    ).element_size()

    if mode is CacheMode.CONTIGUOUS:
        capacity_tokens = config.model.max_position_embeddings
        batch_factor = config.batch_size
    else:
        capacity_tokens = config.num_kv_blocks * config.kv_block_size
        batch_factor = 1

    return (
        2
        * config.model.num_hidden_layers
        * batch_factor
        * config.model.num_key_value_heads
        * capacity_tokens
        * config.model.head_dim
        * element_size
    )


def benchmark_cache_mode(
    config: BenchmarkConfig,
) -> BenchmarkManifest:
    """Benchmark one cache mode after three-way parity validation."""

    _validate_cache_benchmark_config(config)
    device = resolve_device(config.device)
    model = _create_model(config)
    prompt_ids = create_prompt_ids(
        config,
        device=device,
    )
    generation_config = GenerationConfig(
        max_new_tokens=config.max_output_tokens,
        do_sample=False,
    )
    parity = _establish_cache_parity(
        model,
        prompt_ids,
        generation_config,
        config,
    )
    lifecycle = (
        _capture_paged_lifecycle(
            model,
            prompt_ids,
            config,
        )
        if config.cache_mode is CacheMode.PAGED
        else None
    )
    capacity_bytes = theoretical_cache_bytes(
        config,
        config.cache_mode,
    )

    decode_timing = _measure_decode_mode(
        model,
        prompt_ids,
        config,
        device=device,
    )
    generation_timing = _measure_generation_mode(
        model,
        prompt_ids,
        generation_config,
        config,
        device=device,
    )

    measurements = (
        CacheMeasurement(
            operation="decode_step",
            cache_mode=config.cache_mode,
            timing_scope=_decode_scope(config.cache_mode),
            timing=decode_timing,
            theoretical_capacity_bytes=capacity_bytes,
            parity=parity,
            lifecycle=lifecycle,
            notes=_measurement_notes(config.cache_mode),
        ).to_dict(),
        CacheMeasurement(
            operation="generation",
            cache_mode=config.cache_mode,
            timing_scope=_generation_scope(config.cache_mode),
            timing=generation_timing,
            theoretical_capacity_bytes=capacity_bytes,
            parity=parity,
            lifecycle=lifecycle,
            notes=_measurement_notes(config.cache_mode),
        ).to_dict(),
    )

    manifest = create_manifest(config)
    return replace(
        manifest,
        measurements=measurements,
    )


def _create_model(
    config: BenchmarkConfig,
) -> LlamaModel:
    """Create deterministic weights for the selected backend."""

    torch.manual_seed(config.seed)
    model = LlamaModel(
        llama_config(config.model),
        backend=config.backend,
    )
    model.to(
        device=resolve_device(config.device),
        dtype=resolve_dtype(config.dtype),
    )
    model.eval()
    return model


@torch.inference_mode()
def _establish_cache_parity(
    model: LlamaModel,
    prompt_ids: torch.Tensor,
    generation_config: GenerationConfig,
    config: BenchmarkConfig,
) -> CacheParityEvidence:
    """Require equivalent logits and generated tokens."""

    decode_ids = torch.ones(
        (1, 1),
        device=prompt_ids.device,
        dtype=torch.int64,
    )
    full_decode_input = torch.cat(
        (
            prompt_ids,
            decode_ids,
        ),
        dim=1,
    )
    uncached_decode_logits = model(full_decode_input)[:, -1:, :]

    contiguous_prefill = prefill(
        model,
        prompt_ids,
        max_sequence_length=(prompt_ids.shape[1] + config.max_output_tokens),
    )
    contiguous_decode_logits = decode(
        model,
        decode_ids,
        contiguous_prefill.cache,
    )

    paged_manager = _create_manager(model, config)
    paged_prefill = prefill_paged(
        model,
        prompt_ids,
        paged_manager,
        sequence_id="parity-decode",
    )

    try:
        paged_decode_logits = decode_paged(
            model,
            decode_ids,
            paged_prefill.cache,
        )
    finally:
        paged_manager.release_sequence("parity-decode")

    _assert_close(
        contiguous_decode_logits,
        uncached_decode_logits,
        name="contiguous decode",
    )
    _assert_close(
        paged_decode_logits,
        uncached_decode_logits,
        name="paged decode",
    )

    uncached_result = generate(
        model,
        prompt_ids,
        generation_config,
    )
    contiguous_result = generate_with_cache(
        model,
        prompt_ids,
        generation_config,
    )
    generation_manager = _create_manager(model, config)
    paged_result = generate_with_paged_cache(
        model,
        prompt_ids,
        generation_config,
        generation_manager,
        sequence_id="parity-generation",
    )

    contiguous_match = torch.equal(
        contiguous_result.generated_token_ids,
        uncached_result.generated_token_ids,
    )
    paged_match = torch.equal(
        paged_result.generated_token_ids,
        uncached_result.generated_token_ids,
    )

    if not contiguous_match:
        raise AssertionError(
            "contiguous generation tokens do not match uncached generation"
        )

    if not paged_match:
        raise AssertionError("paged generation tokens do not match uncached generation")

    return CacheParityEvidence(
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
        contiguous_decode_max_absolute_error=_maximum_error(
            contiguous_decode_logits,
            uncached_decode_logits,
        ),
        paged_decode_max_absolute_error=_maximum_error(
            paged_decode_logits,
            uncached_decode_logits,
        ),
        contiguous_generation_tokens_match=contiguous_match,
        paged_generation_tokens_match=paged_match,
        generated_tokens=uncached_result.num_generated_tokens,
    )


def _measure_decode_mode(
    model: LlamaModel,
    prompt_ids: torch.Tensor,
    config: BenchmarkConfig,
    *,
    device: torch.device,
) -> TimingResult:
    """Measure one equivalent next-token model step."""

    decode_ids = torch.ones(
        (1, 1),
        device=device,
        dtype=torch.int64,
    )

    if config.cache_mode is CacheMode.NONE:
        full_input = torch.cat(
            (
                prompt_ids,
                decode_ids,
            ),
            dim=1,
        )

        def prepare() -> MeasuredOperation:
            def operation() -> int:
                logits = model(full_input)

                if logits.shape[1] != full_input.shape[1]:
                    raise RuntimeError("uncached logits have an unexpected length")

                return 1

            return operation

        return measure_operations(
            prepare,
            device=device,
            warmup_iterations=config.warmup_iterations,
            measured_repetitions=config.measured_repetitions,
        )

    if config.cache_mode is CacheMode.CONTIGUOUS:

        def prepare() -> MeasuredOperation:
            populated = prefill(
                model,
                prompt_ids,
                max_sequence_length=(prompt_ids.shape[1] + config.max_output_tokens),
            )

            def operation() -> int:
                decode(
                    model,
                    decode_ids,
                    populated.cache,
                )
                return 1

            return operation

        return measure_operations(
            prepare,
            device=device,
            warmup_iterations=config.warmup_iterations,
            measured_repetitions=config.measured_repetitions,
        )

    def prepare_paged() -> PreparedOperation:
        manager = _create_manager(model, config)
        populated = prefill_paged(
            model,
            prompt_ids,
            manager,
            sequence_id="measured-decode",
        )

        def operation() -> int:
            decode_paged(
                model,
                decode_ids,
                populated.cache,
            )
            return 1

        def cleanup() -> None:
            manager.release_sequence("measured-decode")

        return operation, cleanup

    return _measure_with_cleanup(
        prepare_paged,
        device=device,
        warmup_iterations=config.warmup_iterations,
        measured_repetitions=config.measured_repetitions,
    )


def _measure_generation_mode(
    model: LlamaModel,
    prompt_ids: torch.Tensor,
    generation_config: GenerationConfig,
    config: BenchmarkConfig,
    *,
    device: torch.device,
) -> TimingResult:
    """Measure complete greedy generation for one cache mode."""

    if config.cache_mode is CacheMode.NONE:

        def prepare() -> MeasuredOperation:
            def operation() -> int:
                result = generate(
                    model,
                    prompt_ids,
                    generation_config,
                )
                return result.num_generated_tokens

            return operation

    elif config.cache_mode is CacheMode.CONTIGUOUS:

        def prepare() -> MeasuredOperation:
            def operation() -> int:
                result = generate_with_cache(
                    model,
                    prompt_ids,
                    generation_config,
                )
                return result.num_generated_tokens

            return operation

    else:

        def prepare() -> MeasuredOperation:
            manager = _create_manager(model, config)

            def operation() -> int:
                result = generate_with_paged_cache(
                    model,
                    prompt_ids,
                    generation_config,
                    manager,
                    sequence_id="measured-generation",
                )
                return result.num_generated_tokens

            return operation

    return measure_operations(
        prepare,
        device=device,
        warmup_iterations=config.warmup_iterations,
        measured_repetitions=config.measured_repetitions,
    )


def _measure_with_cleanup(
    prepare: PrepareWithCleanup,
    *,
    device: torch.device,
    warmup_iterations: int,
    measured_repetitions: int,
) -> TimingResult:
    """Measure operations while excluding resource cleanup."""

    for _ in range(warmup_iterations):
        operation, cleanup = prepare()

        try:
            synchronize_device(device)
            operation()
            synchronize_device(device)
        finally:
            cleanup()

    durations: list[float] = []
    token_counts: list[int] = []
    allocated: list[int] | None = [] if device.type == "cuda" else None
    reserved: list[int] | None = [] if device.type == "cuda" else None

    for _ in range(measured_repetitions):
        operation, cleanup = prepare()

        try:
            synchronize_device(device)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            started = perf_counter()
            token_count = operation()
            synchronize_device(device)
            finished = perf_counter()

            durations.append(finished - started)
            token_counts.append(token_count)

            if allocated is not None:
                allocated.append(torch.cuda.max_memory_allocated(device))

            if reserved is not None:
                reserved.append(torch.cuda.max_memory_reserved(device))
        finally:
            cleanup()

    return TimingResult(
        durations_seconds=tuple(durations),
        processed_tokens=tuple(token_counts),
        peak_allocated_bytes=(tuple(allocated) if allocated is not None else None),
        peak_reserved_bytes=(tuple(reserved) if reserved is not None else None),
    )


def _capture_paged_lifecycle(
    model: LlamaModel,
    prompt_ids: torch.Tensor,
    config: BenchmarkConfig,
) -> CacheLifecycleEvidence:
    """Observe allocation and release without timing the audit."""

    manager = _create_manager(model, config)
    result = prefill_paged(
        model,
        prompt_ids,
        manager,
        sequence_id="lifecycle",
    )
    after_prefill = manager.allocator.allocated_count
    decode_ids = torch.ones(
        (1, 1),
        device=prompt_ids.device,
        dtype=torch.int64,
    )
    decode_paged(
        model,
        decode_ids,
        result.cache,
    )
    after_decode = manager.allocator.allocated_count
    manager.release_sequence("lifecycle")

    return CacheLifecycleEvidence(
        capacity_blocks=manager.allocator.num_blocks,
        allocated_after_prefill=after_prefill,
        allocated_after_decode=after_decode,
        allocated_after_release=(manager.allocator.allocated_count),
        free_after_release=manager.allocator.free_count,
    )


def _create_manager(
    model: LlamaModel,
    config: BenchmarkConfig,
) -> PagedKVCacheManager:
    """Create a manager matching the model and benchmark capacity."""

    parameter = model.embed_tokens.weight

    return PagedKVCacheManager(
        num_layers=model.config.num_hidden_layers,
        num_blocks=config.num_kv_blocks,
        num_key_value_heads=model.config.num_key_value_heads,
        block_size=config.kv_block_size,
        head_dim=model.config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )


def _validate_cache_benchmark_config(
    config: BenchmarkConfig,
) -> None:
    """Require a comparable single-sequence cache experiment."""

    if config.prompt.kind is not InputKind.TOKEN_IDS:
        raise ValueError("cache benchmark requires fixed prompt token IDs")

    if config.batch_size != 1:
        raise ValueError("cache benchmark currently requires batch_size=1")

    if config.workloads != (WorkloadKind.GENERATION,):
        raise ValueError("cache benchmark config must select generation")

    required_blocks = (
        config.prompt.prompt_length
        + config.max_output_tokens
        + config.kv_block_size
        - 1
    ) // config.kv_block_size

    if required_blocks > config.num_kv_blocks:
        raise ValueError("paged cache does not cover prompt and output capacity")


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
) -> None:
    """Require cache-mode numerical parity."""

    try:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
        )
    except AssertionError as error:
        raise AssertionError(f"{name} parity failed") from error


def _maximum_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    """Return maximum float32 absolute error."""

    difference = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    return float(difference.max().item())


def _decode_scope(mode: CacheMode) -> str:
    """Describe the measured decode boundary."""

    if mode is CacheMode.NONE:
        return "full-sequence forward for one next-token step"

    if mode is CacheMode.CONTIGUOUS:
        return "one-token decode after untimed contiguous prefill"

    return "one-token decode after untimed paged prefill; paged release excluded"


def _generation_scope(mode: CacheMode) -> str:
    """Describe the complete generation boundary."""

    if mode is CacheMode.NONE:
        return "complete greedy full-sequence recomputation"

    if mode is CacheMode.CONTIGUOUS:
        return "complete greedy generation with contiguous KV cache"

    return (
        "complete greedy generation with a persistent paged pool; "
        "sequence allocation and release included"
    )


def _measurement_notes(
    mode: CacheMode,
) -> tuple[str, ...]:
    """Return interpretation limits for a cache mode."""

    common = (
        "Warmup samples are excluded.",
        "Generated tokens are parity-checked before timing.",
        "CUDA memory is measured separately from theoretical capacity.",
    )

    if mode is CacheMode.PAGED:
        return (
            *common,
            "Paged storage capacity is preallocated outside generation timing.",
            "Paged caching is not assumed to reduce total memory.",
        )

    if mode is CacheMode.CONTIGUOUS:
        return (
            *common,
            "Contiguous cache allocation is included in generation timing.",
        )

    return (
        *common,
        "Uncached decoding recomputes the complete sequence.",
    )
