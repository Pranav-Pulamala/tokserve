"""Sequential generation execution for scheduled requests."""

from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.generation.paged_generate import generate_with_paged_cache
from tokserve.generation.types import GenerationResult
from tokserve.scheduler.scheduler import RequestScheduler


class ScheduledGenerationExecutor:
    """Run scheduled requests one at a time using existing generation."""

    def __init__(
        self,
        model: LlamaModel,
        scheduler: RequestScheduler,
        cache_manager: PagedKVCacheManager,
    ) -> None:
        if scheduler.cache_manager is not cache_manager:
            raise ValueError("scheduler and executor must share the cache manager")

        self.model = model
        self.scheduler = scheduler
        self.cache_manager = cache_manager

    def run_next(self) -> GenerationResult | None:
        """Execute the earliest running or newly admitted request."""

        if not self.scheduler.running_requests:
            self.scheduler.schedule()

        if not self.scheduler.running_requests:
            return None

        request = self.scheduler.running_requests[0]

        try:
            result = generate_with_paged_cache(
                self.model,
                request.input_ids,
                request.generation_config,
                self.cache_manager,
                sequence_id=request.request_id,
            )
        except Exception:
            self.scheduler.cancel(request.request_id)
            raise

        self.scheduler.complete(request.request_id, result)
        return result

    def run_until_idle(self) -> tuple[GenerationResult, ...]:
        """Run requests sequentially until no work remains."""

        results: list[GenerationResult] = []

        while self.scheduler.waiting_count or self.scheduler.running_count:
            result = self.run_next()

            if result is None:
                break

            results.append(result)

        return tuple(results)
