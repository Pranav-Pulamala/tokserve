import pytest
import torch

from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.generation.types import GenerationConfig
from tokserve.scheduler.request import RequestState, ScheduledRequest
from tokserve.scheduler.scheduler import RequestScheduler


def create_manager() -> PagedKVCacheManager:
    return PagedKVCacheManager(
        num_layers=1,
        num_blocks=4,
        num_key_value_heads=1,
        block_size=2,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def create_request(request_id: str) -> ScheduledRequest:
    return ScheduledRequest(
        request_id,
        torch.tensor([[1, 2]], dtype=torch.int64),
        GenerationConfig(max_new_tokens=2),
    )


def test_completion_restores_capacity_and_admits_next_fifo_request() -> None:
    scheduler = RequestScheduler(max_running_requests=1)
    scheduler.submit(create_request("A"))
    scheduler.submit(create_request("B"))

    scheduler.schedule()
    scheduler.complete("A")
    result = scheduler.schedule()

    assert scheduler.get_request("A").state is RequestState.FINISHED
    assert tuple(request.request_id for request in result.admitted_requests) == ("B",)


def test_completion_releases_active_cache_sequence() -> None:
    manager = create_manager()
    scheduler = RequestScheduler(
        max_running_requests=1,
        cache_manager=manager,
    )
    scheduler.submit(create_request("A"))
    scheduler.schedule()

    cache = manager.create_sequence("A", max_sequence_length=4)
    cache.begin_append(2)

    assert manager.allocator.allocated_count == 1

    scheduler.complete("A")

    assert manager.active_sequence_ids == ()
    assert manager.allocator.allocated_count == 0


def test_waiting_and_running_cancellation() -> None:
    scheduler = RequestScheduler(max_running_requests=1)
    scheduler.submit(create_request("A"))
    scheduler.submit(create_request("B"))
    scheduler.schedule()

    scheduler.cancel("B")
    scheduler.cancel("A")

    assert scheduler.get_request("A").state is RequestState.CANCELLED
    assert scheduler.get_request("B").state is RequestState.CANCELLED
    assert scheduler.running_count == 0
    assert scheduler.waiting_count == 0


def test_double_completion_and_unknown_ids_are_rejected() -> None:
    scheduler = RequestScheduler(max_running_requests=1)
    scheduler.submit(create_request("A"))
    scheduler.schedule()
    scheduler.complete("A")

    with pytest.raises(RuntimeError, match="not running"):
        scheduler.complete("A")

    with pytest.raises(KeyError, match="unknown request_id"):
        scheduler.cancel("missing")
