"""Deterministic CPU development application for local testing."""

import torch

from tokserve.api.app import create_app
from tokserve.api.service import GenerationService
from tokserve.batching.engine import ContinuousBatchEngine
from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.manager import (
    PagedKVCacheManager,
)
from tokserve.metrics.request_metrics import (
    RequestMetricsCollector,
)
from tokserve.reference.llama.config import LlamaConfig
from tokserve.scheduler.scheduler import RequestScheduler

DEVELOPMENT_SEED = 14_001


class IntegerTokenizer:
    """Encode and decode space-separated integer token IDs."""

    def encode(self, text: str) -> list[int]:
        """Encode a prompt containing integer IDs."""

        return [int(piece) for piece in text.split()]

    def decode(
        self,
        token_ids: list[int],
    ) -> str:
        """Decode IDs into a deterministic text representation."""

        return " ".join(str(token_id) for token_id in token_ids)


def create_development_service() -> GenerationService:
    """Create a lightweight real TokServe CPU service."""

    torch.manual_seed(DEVELOPMENT_SEED)

    model = LlamaModel(
        LlamaConfig(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    )
    model.eval()

    parameter = model.embed_tokens.weight
    cache_manager = PagedKVCacheManager(
        num_layers=model.config.num_hidden_layers,
        num_blocks=256,
        num_key_value_heads=(model.config.num_key_value_heads),
        block_size=16,
        head_dim=model.config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    scheduler = RequestScheduler(
        max_running_requests=8,
        cache_manager=cache_manager,
    )
    metrics = RequestMetricsCollector(history_size=1_024)
    engine = ContinuousBatchEngine(
        model,
        scheduler,
        cache_manager,
        metrics=metrics,
    )

    return GenerationService(
        engine,
        IntegerTokenizer(),
        metrics=metrics,
    )


development_service = create_development_service()
app = create_app(development_service)
