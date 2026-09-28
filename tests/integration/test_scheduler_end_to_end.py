import torch

from tokserve.generation.types import GenerationConfig
from tokserve.scheduler.request import RequestState, ScheduledRequest
from tokserve.scheduler.scheduler import RequestScheduler


def create_request(request_id: str) -> ScheduledRequest:
    return ScheduledRequest(
        request_id,
        torch.tensor([[1, 2]], dtype=torch.int64),
        GenerationConfig(max_new_tokens=2),
    )


def test_three_request_lifecycle_with_capacity_two() -> None:
    scheduler = RequestScheduler(max_running_requests=2)

    for request_id in ("A", "B", "C"):
        scheduler.submit(create_request(request_id))

    first = scheduler.schedule()

    assert tuple(request.request_id for request in first.admitted_requests) == (
        "A",
        "B",
    )
    assert scheduler.get_request("A").state is RequestState.RUNNING
    assert scheduler.get_request("B").state is RequestState.RUNNING
    assert scheduler.get_request("C").state is RequestState.WAITING

    scheduler.complete("A")
    second = scheduler.schedule()

    assert tuple(request.request_id for request in second.admitted_requests) == ("C",)
    assert scheduler.get_request("A").state is RequestState.FINISHED
    assert scheduler.get_request("C").state is RequestState.RUNNING

    scheduler.complete("B")
    scheduler.complete("C")

    assert scheduler.get_request("A").state is RequestState.FINISHED
    assert scheduler.get_request("B").state is RequestState.FINISHED
    assert scheduler.get_request("C").state is RequestState.FINISHED
    assert scheduler.waiting_count == 0
    assert scheduler.running_count == 0
