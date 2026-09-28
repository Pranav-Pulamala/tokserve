import torch

from tokserve.generation.types import GenerationConfig
from tokserve.scheduler.request import RequestState, ScheduledRequest
from tokserve.scheduler.scheduler import RequestScheduler


def create_request(request_id: str) -> ScheduledRequest:
    return ScheduledRequest(
        request_id,
        torch.tensor([[1]], dtype=torch.int64),
        GenerationConfig(max_new_tokens=1),
    )


def test_capacity_one_admits_only_first_request() -> None:
    scheduler = RequestScheduler(max_running_requests=1)
    scheduler.submit(create_request("A"))
    scheduler.submit(create_request("B"))

    result = scheduler.schedule()

    assert tuple(request.request_id for request in result.admitted_requests) == ("A",)
    assert result.running_count == 1
    assert result.waiting_count == 1
    assert scheduler.get_request("A").state is RequestState.RUNNING
    assert scheduler.get_request("B").state is RequestState.WAITING


def test_larger_capacity_preserves_fifo_admission() -> None:
    scheduler = RequestScheduler(max_running_requests=2)

    for request_id in ("A", "B", "C"):
        scheduler.submit(create_request(request_id))

    result = scheduler.schedule()

    assert tuple(request.request_id for request in result.admitted_requests) == (
        "A",
        "B",
    )
    assert tuple(request.request_id for request in scheduler.waiting_requests) == ("C",)


def test_fewer_requests_than_capacity_are_all_admitted() -> None:
    scheduler = RequestScheduler(max_running_requests=3)
    scheduler.submit(create_request("A"))

    result = scheduler.schedule()

    assert len(result.admitted_requests) == 1
    assert result.waiting_count == 0
    assert result.running_count == 1


def test_empty_and_repeated_scheduling_are_safe() -> None:
    scheduler = RequestScheduler(max_running_requests=1)

    empty = scheduler.schedule()
    assert empty.admitted_requests == ()

    scheduler.submit(create_request("A"))
    scheduler.submit(create_request("B"))
    scheduler.schedule()
    repeated = scheduler.schedule()

    assert repeated.admitted_requests == ()
    assert repeated.running_count == 1
    assert repeated.waiting_count == 1
