"""Deterministic FIFO request scheduler."""

from collections import deque
from dataclasses import dataclass

from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.generation.types import GenerationResult
from tokserve.scheduler.request import RequestState, ScheduledRequest


@dataclass(frozen=True)
class ScheduleResult:
    """Summary of one scheduler admission decision."""

    admitted_requests: tuple[ScheduledRequest, ...]
    waiting_count: int
    running_count: int


class RequestScheduler:
    """Track waiting and running requests in FIFO order."""

    def __init__(
        self,
        *,
        max_running_requests: int,
        cache_manager: PagedKVCacheManager | None = None,
    ) -> None:
        if max_running_requests < 1:
            raise ValueError("max_running_requests must be positive")

        self.max_running_requests = max_running_requests
        self.cache_manager = cache_manager
        self._requests: dict[str, ScheduledRequest] = {}
        self._waiting: deque[ScheduledRequest] = deque()
        self._running: dict[str, ScheduledRequest] = {}
        self._reserved_blocks: dict[str, int] = {}

    @property
    def waiting_requests(self) -> tuple[ScheduledRequest, ...]:
        """Return an immutable FIFO snapshot of waiting requests."""

        return tuple(self._waiting)

    @property
    def running_requests(self) -> tuple[ScheduledRequest, ...]:
        """Return an immutable snapshot of running requests."""

        return tuple(self._running.values())

    @property
    def waiting_count(self) -> int:
        """Return the number of waiting requests."""

        return len(self._waiting)

    @property
    def running_count(self) -> int:
        """Return the number of running requests."""

        return len(self._running)

    def submit(self, request: ScheduledRequest) -> None:
        """Add one waiting request to the FIFO queue."""

        if request.request_id in self._requests:
            raise ValueError("request_id is already registered")

        if request.state is not RequestState.WAITING:
            raise ValueError("submitted request must be waiting")

        self._requests[request.request_id] = request
        self._waiting.append(request)

    def schedule(self) -> ScheduleResult:
        """Admit eligible waiting requests in FIFO order."""

        available_slots = self.max_running_requests - self.running_count
        available_blocks = self._available_block_capacity()
        admitted: list[ScheduledRequest] = []

        while available_slots > 0 and self._waiting:
            request = self._waiting[0]
            required_blocks = self._required_blocks(request)

            if required_blocks > available_blocks:
                break

            self._waiting.popleft()
            request.mark_running()
            self._running[request.request_id] = request

            if required_blocks:
                self._reserved_blocks[request.request_id] = required_blocks
                available_blocks -= required_blocks

            admitted.append(request)
            available_slots -= 1

        return ScheduleResult(
            admitted_requests=tuple(admitted),
            waiting_count=self.waiting_count,
            running_count=self.running_count,
        )

    def complete(
        self,
        request_id: str,
        result: GenerationResult | None = None,
    ) -> None:
        """Finish a running request and release its resources."""

        request = self._get_running(request_id)
        self._release_resources(request_id)
        del self._running[request_id]
        request.mark_finished(result)

    def cancel(self, request_id: str) -> None:
        """Cancel a waiting or running request."""

        request = self.get_request(request_id)

        if request.state is RequestState.WAITING:
            self._waiting = deque(
                queued for queued in self._waiting if queued.request_id != request_id
            )
            request.cancel()
            return

        if request.state is RequestState.RUNNING:
            self._release_resources(request_id)
            del self._running[request_id]
            request.cancel()
            return

        raise RuntimeError(f"cannot cancel request in state {request.state.value}")

    def _get_running(self, request_id: str) -> ScheduledRequest:
        """Return a running request."""

        try:
            return self._running[request_id]
        except KeyError as error:
            raise RuntimeError("request is not running") from error

    def _required_blocks(self, request: ScheduledRequest) -> int:
        """Estimate maximum paged-cache blocks required by a request."""

        if self.cache_manager is None:
            return 0

        if request.generation_config.max_new_tokens == 0:
            return 0

        total_length = (
            request.input_ids.shape[1] + request.generation_config.max_new_tokens
        )
        block_size = self.cache_manager.storage.block_size
        return (total_length + block_size - 1) // block_size

    def _available_block_capacity(self) -> int:
        """Return capacity not claimed by scheduler reservations."""

        if self.cache_manager is None:
            return 0

        reserved = sum(self._reserved_blocks.values())
        return max(0, self.cache_manager.allocator.num_blocks - reserved)

    def _release_resources(self, request_id: str) -> None:
        """Release scheduler reservations and active paged-cache ownership."""

        self._reserved_blocks.pop(request_id, None)

        if (
            self.cache_manager is not None
            and request_id in self.cache_manager.active_sequence_ids
        ):
            self.cache_manager.release_sequence(request_id)

    def get_request(self, request_id: str) -> ScheduledRequest:
        """Return a registered request."""

        try:
            return self._requests[request_id]
        except KeyError as error:
            raise KeyError("unknown request_id") from error
