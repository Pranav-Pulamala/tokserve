"""Deterministic FIFO request scheduler."""

from collections import deque

from tokserve.scheduler.request import RequestState, ScheduledRequest


class RequestScheduler:
    """Track waiting and running requests in FIFO order."""

    def __init__(self, *, max_running_requests: int) -> None:
        if max_running_requests < 1:
            raise ValueError("max_running_requests must be positive")

        self.max_running_requests = max_running_requests
        self._requests: dict[str, ScheduledRequest] = {}
        self._waiting: deque[ScheduledRequest] = deque()
        self._running: dict[str, ScheduledRequest] = {}

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

    def get_request(self, request_id: str) -> ScheduledRequest:
        """Return a registered request."""

        try:
            return self._requests[request_id]
        except KeyError as error:
            raise KeyError("unknown request_id") from error
