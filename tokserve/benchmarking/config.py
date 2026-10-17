"""Validated configurations for reproducible TokServe benchmarks."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypeAlias

from tokserve.engine.kernel_backend import KernelBackend


class WorkloadKind(StrEnum):
    """Supported benchmark workload categories."""

    PREFILL = "prefill"
    DECODE = "decode"
    GENERATION = "generation"
    SERVING = "serving"


class CacheMode(StrEnum):
    """KV-cache strategy used by a benchmark."""

    NONE = "none"
    CONTIGUOUS = "contiguous"
    PAGED = "paged"


class InputKind(StrEnum):
    """How a benchmark prompt is represented."""

    TOKEN_IDS = "token_ids"
    TEXT = "text"


DTypeName: TypeAlias = Literal[
    "float32",
    "float16",
    "bfloat16",
]
SamplingMode: TypeAlias = Literal["greedy"]


@dataclass(frozen=True)
class ModelSpec:
    """Model architecture and identity used by one benchmark."""

    identifier: str
    revision: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0

    def __post_init__(self) -> None:
        """Reject invalid or ambiguous model specifications."""

        if not self.identifier.strip():
            raise ValueError("model identifier must be nonempty")

        if not self.revision.strip():
            raise ValueError("model revision must be nonempty")

        dimensions = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
        }

        for name, value in dimensions.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        if self.rms_norm_eps <= 0.0:
            raise ValueError("rms_norm_eps must be positive")

        if self.rope_theta <= 0.0:
            raise ValueError("rope_theta must be positive")

    @property
    def head_dim(self) -> int:
        """Return the dimension assigned to each attention head."""

        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe model description."""

        return {
            "identifier": self.identifier,
            "revision": self.revision,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "head_dim": self.head_dim,
        }


@dataclass(frozen=True)
class PromptSpec:
    """One fixed-token or tokenizer-dependent benchmark prompt."""

    kind: InputKind
    token_ids: tuple[int, ...] | None = None
    text: str | None = None
    tokenizer_identifier: str | None = None
    expected_token_count: int | None = None

    def __post_init__(self) -> None:
        """Require an unambiguous prompt representation."""

        if self.kind is InputKind.TOKEN_IDS:
            self._validate_token_ids()
            return

        if self.kind is InputKind.TEXT:
            self._validate_text()
            return

        raise ValueError("unsupported prompt kind")

    @classmethod
    def from_token_ids(
        cls,
        token_ids: tuple[int, ...],
    ) -> "PromptSpec":
        """Create an already-tokenized prompt."""

        return cls(
            kind=InputKind.TOKEN_IDS,
            token_ids=token_ids,
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        tokenizer_identifier: str,
        expected_token_count: int,
    ) -> "PromptSpec":
        """Create a text prompt with explicit tokenizer assumptions."""

        return cls(
            kind=InputKind.TEXT,
            text=text,
            tokenizer_identifier=tokenizer_identifier,
            expected_token_count=expected_token_count,
        )

    @property
    def prompt_length(self) -> int:
        """Return the expected number of prompt tokens."""

        if self.token_ids is not None:
            return len(self.token_ids)

        if self.expected_token_count is None:
            raise RuntimeError("text prompt has no expected token count")

        return self.expected_token_count

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe prompt description."""

        return {
            "kind": self.kind.value,
            "token_ids": (list(self.token_ids) if self.token_ids is not None else None),
            "text": self.text,
            "tokenizer_identifier": self.tokenizer_identifier,
            "expected_token_count": self.expected_token_count,
            "prompt_length": self.prompt_length,
        }

    def _validate_token_ids(self) -> None:
        """Validate an already-tokenized prompt."""

        if self.token_ids is None or not self.token_ids:
            raise ValueError("token_ids prompt must contain at least one token")

        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token IDs must be nonnegative")

        if self.text is not None:
            raise ValueError("token_ids prompt cannot also contain text")

        if self.tokenizer_identifier is not None:
            raise ValueError("fixed token IDs must not claim a tokenizer identifier")

        if self.expected_token_count is not None:
            raise ValueError("fixed token IDs derive their count from token_ids")

    def _validate_text(self) -> None:
        """Validate a tokenizer-dependent text prompt."""

        if self.text is None or not self.text.strip():
            raise ValueError("text prompt must contain non-whitespace text")

        if self.token_ids is not None:
            raise ValueError("text prompt cannot also contain token IDs")

        if self.tokenizer_identifier is None or not self.tokenizer_identifier.strip():
            raise ValueError("text prompt requires a tokenizer identifier")

        if self.expected_token_count is None or self.expected_token_count < 1:
            raise ValueError("text prompt requires a positive expected token count")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Complete reproducible configuration for one benchmark run."""

    name: str
    source_revision: str
    model: ModelSpec
    prompt: PromptSpec
    workloads: tuple[WorkloadKind, ...]
    backend: KernelBackend
    device: str
    dtype: DTypeName
    seed: int
    max_output_tokens: int
    batch_size: int
    concurrency: int
    cache_mode: CacheMode
    kv_block_size: int
    num_kv_blocks: int
    warmup_iterations: int
    measured_repetitions: int
    output_path: Path
    sampling: SamplingMode = "greedy"

    def __post_init__(self) -> None:
        """Validate benchmark comparability and runtime constraints."""

        if not self.name.strip():
            raise ValueError("benchmark name must be nonempty")

        if not self.source_revision.strip():
            raise ValueError("source_revision must be nonempty")

        if not self.workloads:
            raise ValueError("at least one workload must be selected")

        if len(set(self.workloads)) != len(self.workloads):
            raise ValueError("workloads must not contain duplicates")

        if self.backend not in ("torch", "triton"):
            raise ValueError("backend must be 'torch' or 'triton'")

        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be 'cpu', 'cuda', or 'mps'")

        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("unsupported benchmark dtype")

        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")

        if self.kv_block_size < 1:
            raise ValueError("kv_block_size must be positive")

        if self.num_kv_blocks < 1:
            raise ValueError("num_kv_blocks must be positive")

        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be nonnegative")

        if self.measured_repetitions < 1:
            raise ValueError("measured_repetitions must be positive")

        if self.output_path.suffix.lower() != ".json":
            raise ValueError("benchmark output path must end in .json")

        if self.sampling != "greedy":
            raise ValueError(
                "Milestone 15 comparable benchmarks require greedy decoding"
            )

        total_length = self.prompt.prompt_length + self.max_output_tokens

        if total_length > self.model.max_position_embeddings:
            raise ValueError("prompt and output tokens exceed max_position_embeddings")

        if self.prompt.token_ids is not None and any(
            token_id >= self.model.vocab_size for token_id in self.prompt.token_ids
        ):
            raise ValueError("prompt token ID exceeds model vocabulary")

        if self.cache_mode is CacheMode.NONE and (
            WorkloadKind.DECODE in self.workloads
        ):
            raise ValueError("decode workload requires a KV cache")

        if self.cache_mode is CacheMode.PAGED and self.batch_size != 1:
            raise ValueError("paged model execution currently requires batch_size=1")

        if self.backend == "triton":
            self._validate_triton_configuration()

        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU-safe benchmark configurations must use float32")

    def comparability_key(self) -> tuple[object, ...]:
        """Return fields that must match for a direct comparison."""

        return (
            self.model.to_dict(),
            self.prompt.to_dict(),
            tuple(workload.value for workload in self.workloads),
            self.device,
            self.dtype,
            self.seed,
            self.max_output_tokens,
            self.batch_size,
            self.concurrency,
            self.cache_mode.value,
            self.kv_block_size,
            self.num_kv_blocks,
            self.warmup_iterations,
            self.measured_repetitions,
            self.sampling,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe benchmark configuration."""

        return {
            "name": self.name,
            "source_revision": self.source_revision,
            "model": self.model.to_dict(),
            "prompt": self.prompt.to_dict(),
            "workloads": [workload.value for workload in self.workloads],
            "backend": self.backend,
            "device": self.device,
            "dtype": self.dtype,
            "seed": self.seed,
            "max_output_tokens": self.max_output_tokens,
            "batch_size": self.batch_size,
            "concurrency": self.concurrency,
            "cache_mode": self.cache_mode.value,
            "kv_block_size": self.kv_block_size,
            "num_kv_blocks": self.num_kv_blocks,
            "warmup_iterations": self.warmup_iterations,
            "measured_repetitions": self.measured_repetitions,
            "sampling": self.sampling,
            "output_path": str(self.output_path),
        }

    def _validate_triton_configuration(self) -> None:
        """Apply constraints of TokServe's current full-model backend."""

        if self.device != "cuda":
            raise ValueError("Triton backend requires device='cuda'")

        if self.dtype != "float32":
            raise ValueError(
                "TokServe's current full-model Triton backend requires float32"
            )

        if self.model.head_dim > 128:
            raise ValueError("TokServe Triton attention requires head_dim <= 128")
