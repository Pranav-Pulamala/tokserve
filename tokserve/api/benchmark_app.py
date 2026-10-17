"""Controlled TokServe application for serving benchmarks."""

import os
from dataclasses import dataclass
from typing import cast

import torch
from fastapi import FastAPI

from tokserve.api.app import create_app
from tokserve.api.service import GenerationService
from tokserve.batching.engine import ContinuousBatchEngine
from tokserve.engine.device import resolve_device
from tokserve.engine.kernel_backend import KernelBackend
from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.metrics.request_metrics import RequestMetricsCollector
from tokserve.reference.llama.config import LlamaConfig
from tokserve.scheduler.scheduler import RequestScheduler

BENCHMARK_MODEL_IDENTIFIER = "tokserve-synthetic-serving-llama"


@dataclass(frozen=True)
class BenchmarkServerSettings:
    """Explicit model and serving settings for benchmark runs."""

    device: str = "cpu"
    backend: KernelBackend = "torch"
    seed: int = 1_505
    vocab_size: int = 256
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 2
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    max_position_embeddings: int = 256
    max_running_requests: int = 8
    num_kv_blocks: int = 256
    kv_block_size: int = 16
    metrics_history_size: int = 4_096

    def __post_init__(self) -> None:
        """Validate the controlled benchmark server."""

        if self.device not in ("cpu", "cuda"):
            raise ValueError("benchmark server device must be cpu or cuda")

        if self.backend not in ("torch", "triton"):
            raise ValueError("benchmark server backend must be torch or triton")

        if self.backend == "triton" and self.device != "cuda":
            raise ValueError("Triton benchmark server requires CUDA")

        integer_settings = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "max_running_requests": self.max_running_requests,
            "num_kv_blocks": self.num_kv_blocks,
            "kv_block_size": self.kv_block_size,
            "metrics_history_size": self.metrics_history_size,
        }

        for name, value in integer_settings.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")

        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("attention heads must be divisible by KV heads")

        if (
            self.backend == "triton"
            and self.hidden_size // self.num_attention_heads > 128
        ):
            raise ValueError("Triton attention requires head_dim <= 128")

    def to_dict(self) -> dict[str, object]:
        """Return public benchmark configuration."""

        return {
            "model_identifier": BENCHMARK_MODEL_IDENTIFIER,
            "model_revision": f"random-weights-seed-{self.seed}",
            "device": self.device,
            "backend": self.backend,
            "dtype": "float32",
            "seed": self.seed,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": (self.max_position_embeddings),
            "max_running_requests": self.max_running_requests,
            "num_kv_blocks": self.num_kv_blocks,
            "kv_block_size": self.kv_block_size,
            "metrics_history_size": self.metrics_history_size,
            "tokenizer": "space-separated-integer-token-ids",
            "sampling": "request-controlled; use greedy for comparisons",
        }


class IntegerTokenizer:
    """Encode and decode space-separated integer token IDs."""

    def encode(self, text: str) -> list[int]:
        """Encode an integer-token prompt."""

        return [int(piece) for piece in text.split()]

    def decode(
        self,
        token_ids: list[int],
    ) -> str:
        """Decode token IDs without an external tokenizer."""

        return " ".join(str(token_id) for token_id in token_ids)


def settings_from_environment() -> BenchmarkServerSettings:
    """Read the small documented benchmark environment surface."""

    backend_value = os.environ.get(
        "TOKSERVE_BENCHMARK_BACKEND",
        "torch",
    )

    if backend_value not in ("torch", "triton"):
        raise ValueError("TOKSERVE_BENCHMARK_BACKEND must be torch or triton")

    backend = cast(KernelBackend, backend_value)

    return BenchmarkServerSettings(
        device=os.environ.get(
            "TOKSERVE_BENCHMARK_DEVICE",
            "cpu",
        ),
        backend=backend,
        seed=int(
            os.environ.get(
                "TOKSERVE_BENCHMARK_SEED",
                "1505",
            )
        ),
    )


def create_benchmark_service(
    settings: BenchmarkServerSettings,
) -> GenerationService:
    """Create a real deterministic TokServe benchmark service."""

    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)

    model = LlamaModel(
        LlamaConfig(
            vocab_size=settings.vocab_size,
            hidden_size=settings.hidden_size,
            intermediate_size=settings.intermediate_size,
            num_hidden_layers=settings.num_hidden_layers,
            num_attention_heads=settings.num_attention_heads,
            num_key_value_heads=settings.num_key_value_heads,
            max_position_embeddings=(settings.max_position_embeddings),
        ),
        backend=settings.backend,
    )
    model.to(
        device=device,
        dtype=torch.float32,
    )
    model.eval()

    parameter = model.embed_tokens.weight
    manager = PagedKVCacheManager(
        num_layers=model.config.num_hidden_layers,
        num_blocks=settings.num_kv_blocks,
        num_key_value_heads=model.config.num_key_value_heads,
        block_size=settings.kv_block_size,
        head_dim=model.config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    scheduler = RequestScheduler(
        max_running_requests=settings.max_running_requests,
        cache_manager=manager,
    )
    metrics = RequestMetricsCollector(history_size=settings.metrics_history_size)
    engine = ContinuousBatchEngine(
        model,
        scheduler,
        manager,
        metrics=metrics,
    )

    return GenerationService(
        engine,
        IntegerTokenizer(),
        metrics=metrics,
    )


def create_benchmark_application(
    settings: BenchmarkServerSettings,
) -> FastAPI:
    """Create an isolated application with benchmark metadata."""

    application = create_app(create_benchmark_service(settings))

    @application.get("/benchmark/config")
    async def benchmark_config() -> dict[str, object]:
        """Return the exact controlled server configuration."""

        return settings.to_dict()

    return application


benchmark_settings = settings_from_environment()
app = create_benchmark_application(benchmark_settings)
