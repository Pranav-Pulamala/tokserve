"""Reproducible prefill and cached-decode profiling workloads."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

from tokserve.engine.inference import create_kv_cache
from tokserve.engine.kernel_backend import KernelBackend
from tokserve.engine.model import LlamaModel
from tokserve.reference.llama.config import LlamaConfig

WorkloadKind: TypeAlias = Literal["prefill", "decode"]


@dataclass(frozen=True)
class ProfilingConfig:
    """Explicit model and workload dimensions used during profiling."""

    batch_size: int = 1
    prompt_length: int = 128
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 2
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    vocab_size: int = 256
    max_sequence_length: int = 256
    backend: KernelBackend = "triton"
    dtype: torch.dtype = torch.float32
    seed: int = 1_000

    def __post_init__(self) -> None:
        dimensions = {
            "batch_size": self.batch_size,
            "prompt_length": self.prompt_length,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
        }

        for name, value in dimensions.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")

        if self.backend not in ("torch", "triton"):
            raise ValueError("backend must be 'torch' or 'triton'")

        if self.prompt_length >= self.max_sequence_length:
            raise ValueError("prompt_length must be smaller than max_sequence_length")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        if self.dtype not in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ):
            raise TypeError("dtype must be float32, float16, or bfloat16")

        if self.backend == "triton" and self.dtype != torch.float32:
            raise ValueError(
                "the current Triton model backend requires float32 weights"
            )

    @property
    def head_dim(self) -> int:
        """Return the dimension of each attention head."""

        return self.hidden_size // self.num_attention_heads

    def model_config(self) -> LlamaConfig:
        """Create the corresponding Llama configuration."""

        return LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_sequence_length,
            rms_norm_eps=1e-6,
            rope_theta=10_000.0,
        )


class InferenceProfilingWorkloads:
    """Own deterministic model inputs and prepare isolated inference calls."""

    def __init__(
        self,
        config: ProfilingConfig,
        *,
        device: torch.device,
    ) -> None:
        if config.backend == "triton" and device.type != "cuda":
            raise ValueError("Triton profiling requires a CUDA device")

        self.config = config
        self.device = device

        torch.manual_seed(config.seed)
        self.model = LlamaModel(
            config.model_config(),
            backend=config.backend,
        ).to(device=device, dtype=config.dtype)
        self.model.eval()

        prompt_values = torch.arange(
            config.batch_size * config.prompt_length,
            device=device,
            dtype=torch.int64,
        )
        self.prompt_ids = (
            prompt_values.reshape(
                config.batch_size,
                config.prompt_length,
            )
            % config.vocab_size
        )
        self.decode_ids = torch.full(
            (config.batch_size, 1),
            1,
            device=device,
            dtype=torch.int64,
        )

    def prepare_prefill(self) -> Callable[[], torch.Tensor]:
        """Prepare a fresh cache and return one prefill call."""

        cache = create_kv_cache(
            self.model,
            batch_size=self.config.batch_size,
            max_sequence_length=self.config.max_sequence_length,
        )

        def run() -> torch.Tensor:
            result: torch.Tensor = self.model(
                self.prompt_ids,
                cache=cache,
            )
            return result

        return run

    @torch.inference_mode()
    def prepare_decode(self) -> Callable[[], torch.Tensor]:
        """Populate a fresh cache and return one single-token decode call."""

        cache = create_kv_cache(
            self.model,
            batch_size=self.config.batch_size,
            max_sequence_length=self.config.max_sequence_length,
        )
        self.model(self.prompt_ids, cache=cache)

        def run() -> torch.Tensor:
            result: torch.Tensor = self.model(
                self.decode_ids,
                cache=cache,
            )
            return result

        return run

    def prepare(
        self,
        kind: WorkloadKind,
    ) -> Callable[[], torch.Tensor]:
        """Prepare one isolated workload invocation."""

        if kind == "prefill":
            return self.prepare_prefill()

        if kind == "decode":
            return self.prepare_decode()

        raise ValueError("kind must be 'prefill' or 'decode'")


def environment_metadata() -> dict[str, str]:
    """Return software and accelerator metadata."""

    metadata = {
        "pytorch": torch.__version__,
        "pytorch_cuda": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
    }

    if torch.cuda.is_available():
        metadata["gpu"] = torch.cuda.get_device_name(0)

    try:
        import triton
    except ImportError:
        metadata["triton"] = "unavailable"
    else:
        metadata["triton"] = triton.__version__

    return metadata
