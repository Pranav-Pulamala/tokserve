import pytest
import torch

from tokserve.generation.types import GenerationConfig
from tokserve.scheduler.request import ScheduledRequest
from tokserve.scheduler.scheduler import RequestScheduler


def create_request(request_id: str) -> ScheduledRequest:
    return ScheduledRequest(
        request_id,
        torch.tensor([[1]], dtype=torch.int64),
        GenerationConfig(max_new_tokens=1),
    )


def test_submissions_preserve_fifo_order() -> None:
    scheduler = RequestScheduler(max_running_requests=2)

    for request_id in ("A", "B", "C"):
        scheduler.submit(create_request(request_id))

    assert tuple(request.request_id for request in scheduler.waiting_requests) == (
        "A",
        "B",
        "C",
    )
    assert scheduler.waiting_count == 3
    assert scheduler.running_count == 0


def test_duplicate_request_id_is_rejected() -> None:
    scheduler = RequestScheduler(max_running_requests=1)
    scheduler.submit(create_request("A"))

    with pytest.raises(ValueError, match="already registered"):
        scheduler.submit(create_request("A"))

    assert scheduler.waiting_count == 1


def test_queue_snapshot_cannot_mutate_scheduler_state() -> None:
    scheduler = RequestScheduler(max_running_requests=1)
    scheduler.submit(create_request("A"))

    snapshot = scheduler.waiting_requests

    assert isinstance(snapshot, tuple)
    assert scheduler.waiting_count == 1


def test_empty_scheduler_is_safe() -> None:
    scheduler = RequestScheduler(max_running_requests=1)

    assert scheduler.waiting_requests == ()
    assert scheduler.running_requests == ()
    assert scheduler.waiting_count == 0
    assert scheduler.running_count == 0

    with pytest.raises(KeyError, match="unknown request_id"):
        scheduler.get_request("missing")


def test_invalid_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        RequestScheduler(max_running_requests=0)
