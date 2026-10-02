"""Plan prefill and decode work for one engine iteration."""

from collections.abc import Iterable
from dataclasses import dataclass

from tokserve.scheduler.request import RequestState, ScheduledRequest


@dataclass(frozen=True)
class IterationPlan:
    """Immutable request groups for one continuous-batch iteration."""

    prefill_requests: tuple[ScheduledRequest, ...]
    decode_requests: tuple[ScheduledRequest, ...]

    @property
    def is_empty(self) -> bool:
        """Return whether this iteration contains no executable work."""

        return not self.prefill_requests and not self.decode_requests

    @property
    def request_ids(self) -> tuple[str, ...]:
        """Return every planned request ID in deterministic order."""

        return tuple(
            request.request_id
            for request in (
                *self.prefill_requests,
                *self.decode_requests,
            )
        )


def build_iteration_plan(
    requests: Iterable[ScheduledRequest],
) -> IterationPlan:
    """Classify running requests as prefill or decode work."""

    seen_ids: set[str] = set()
    prefill: list[ScheduledRequest] = []
    decode: list[ScheduledRequest] = []

    for request in requests:
        if request.request_id in seen_ids:
            raise ValueError("iteration plan contains a duplicate request_id")

        seen_ids.add(request.request_id)

        if request.state is not RequestState.RUNNING:
            continue

        if request.completion_reason is not None:
            continue

        if request.prefill_complete:
            if request.decode_ready:
                decode.append(request)
        else:
            prefill.append(request)

    return IterationPlan(
        prefill_requests=tuple(prefill),
        decode_requests=tuple(decode),
    )
