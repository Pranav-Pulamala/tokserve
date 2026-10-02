"""Iteration-level continuous-batching control loop."""

from dataclasses import dataclass

from tokserve.batching.execution import execute_decode, execute_prefill
from tokserve.batching.plan import build_iteration_plan
from tokserve.engine.model import LlamaModel
from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.generation.types import GenerationResult
from tokserve.scheduler.scheduler import RequestScheduler


@dataclass(frozen=True)
class IterationRecord:
    """Debug history for one continuous-batching iteration."""

    iteration: int
    admitted_ids: tuple[str, ...]
    prefill_ids: tuple[str, ...]
    decode_ids: tuple[str, ...]
    completed_ids: tuple[str, ...]
    active_ids: tuple[str, ...]


class ContinuousBatchEngine:
    """Advance dynamically changing requests one iteration at a time."""

    def __init__(
        self,
        model: LlamaModel,
        scheduler: RequestScheduler,
        cache_manager: PagedKVCacheManager,
    ) -> None:
        if scheduler.cache_manager is not cache_manager:
            raise ValueError("scheduler and engine must share the cache manager")

        self.model = model
        self.scheduler = scheduler
        self.cache_manager = cache_manager
        self._iteration = 0
        self._history: list[IterationRecord] = []
        self._results: list[GenerationResult] = []

    @property
    def history(self) -> tuple[IterationRecord, ...]:
        """Return immutable iteration history."""

        return tuple(self._history)

    @property
    def results(self) -> tuple[GenerationResult, ...]:
        """Return completed results in completion order."""

        return tuple(self._results)

    def step(self) -> IterationRecord:
        """Run one admission and execution iteration."""

        schedule_result = self.scheduler.schedule()
        plan = build_iteration_plan(self.scheduler.running_requests)

        execute_prefill(
            self.model,
            self.cache_manager,
            plan.prefill_requests,
        )
        execute_decode(
            self.model,
            self.cache_manager,
            plan.decode_requests,
        )

        completed_ids: list[str] = []

        for request in tuple(self.scheduler.running_requests):
            if request.completion_reason is None:
                continue

            result = request.build_result()
            self.scheduler.complete(request.request_id, result)
            self._results.append(result)
            completed_ids.append(request.request_id)

        record = IterationRecord(
            iteration=self._iteration,
            admitted_ids=tuple(
                request.request_id for request in schedule_result.admitted_requests
            ),
            prefill_ids=tuple(request.request_id for request in plan.prefill_requests),
            decode_ids=tuple(request.request_id for request in plan.decode_requests),
            completed_ids=tuple(completed_ids),
            active_ids=tuple(
                request.request_id for request in self.scheduler.running_requests
            ),
        )
        self._history.append(record)
        self._iteration += 1
        return record

    def run_until_idle(
        self,
        *,
        max_iterations: int = 1_000,
    ) -> tuple[GenerationResult, ...]:
        """Run iterations until no waiting or running requests remain."""

        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")

        iterations = 0

        while self.scheduler.waiting_count or self.scheduler.running_count:
            if iterations >= max_iterations:
                raise RuntimeError("continuous batching exceeded max_iterations")

            record = self.step()
            iterations += 1

            if (
                not record.admitted_ids
                and not record.prefill_ids
                and not record.decode_ids
                and not record.completed_ids
            ):
                raise RuntimeError("continuous batching cannot make progress")

        return self.results
