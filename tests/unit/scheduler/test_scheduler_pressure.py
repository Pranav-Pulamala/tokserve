import torch

from tokserve.engine.paged.manager import PagedKVCacheManager
from tokserve.generation.types import GenerationConfig
from tokserve.scheduler.request import RequestState, ScheduledRequest
from tokserve.scheduler.scheduler import RequestScheduler


def create_manager(*, num_blocks: int = 3) -> PagedKVCacheManager:
    return PagedKVCacheManager(
        num_layers=1,
        num_blocks=num_blocks,
        num_key_value_heads=1,
        block_size=2,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def create_request(
    request_id: str,
    *,
    prompt_length: int,
    max_new_tokens: int,
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id,
        torch.arange(prompt_length, dtype=torch.int64).unsqueeze(0),
        GenerationConfig(max_new_tokens=max_new_tokens),
    )


def test_resource_pressure_keeps_request_waiting_atomically() -> None:
    manager = create_manager(num_blocks=2)
    scheduler = RequestScheduler(
        max_running_requests=2,
        cache_manager=manager,
    )
    too_large = create_request(
        "large",
        prompt_length=3,
        max_new_tokens=2,
    )
    scheduler.submit(too_large)

    result = scheduler.schedule()

    assert result.admitted_requests == ()
    assert too_large.state is RequestState.WAITING
    assert scheduler.waiting_count == 1
    assert scheduler.running_count == 0
    assert manager.allocator.allocated_count == 0


def test_fifo_does_not_bypass_blocked_earlier_request() -> None:
    manager = create_manager(num_blocks=2)
    scheduler = RequestScheduler(
        max_running_requests=2,
        cache_manager=manager,
    )
    scheduler.submit(create_request("A", prompt_length=3, max_new_tokens=2))
    scheduler.submit(create_request("B", prompt_length=1, max_new_tokens=1))

    result = scheduler.schedule()

    assert result.admitted_requests == ()
    assert tuple(request.request_id for request in scheduler.waiting_requests) == (
        "A",
        "B",
    )


def test_completion_allows_next_fifo_request_to_run() -> None:
    manager = create_manager(num_blocks=3)
    scheduler = RequestScheduler(
        max_running_requests=1,
        cache_manager=manager,
    )
    scheduler.submit(create_request("A", prompt_length=2, max_new_tokens=2))
    scheduler.submit(create_request("B", prompt_length=1, max_new_tokens=1))
    scheduler.submit(create_request("C", prompt_length=1, max_new_tokens=1))

    first = scheduler.schedule()
    scheduler.complete("A")
    second = scheduler.schedule()
    scheduler.complete("B")
    third = scheduler.schedule()

    assert tuple(request.request_id for request in first.admitted_requests) == ("A",)
    assert tuple(request.request_id for request in second.admitted_requests) == ("B",)
    assert tuple(request.request_id for request in third.admitted_requests) == ("C",)
    assert scheduler.get_request("A").state is RequestState.FINISHED
    assert scheduler.get_request("B").state is RequestState.FINISHED
    assert scheduler.get_request("C").state is RequestState.RUNNING
