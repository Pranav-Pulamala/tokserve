import torch

from tokserve.engine.paged.manager import (
    PagedKVCacheManager,
)
from tokserve.generation.types import GenerationConfig
from tokserve.metrics.request_metrics import (
    RequestMetricsCollector,
)
from tokserve.metrics.serving import (
    capture_serving_metrics,
)
from tokserve.scheduler.request import ScheduledRequest
from tokserve.scheduler.scheduler import RequestScheduler


def create_manager() -> PagedKVCacheManager:
    return PagedKVCacheManager(
        num_layers=1,
        num_blocks=4,
        num_key_value_heads=1,
        block_size=2,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def create_request(
    request_id: str,
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id,
        torch.tensor(
            [[1]],
            dtype=torch.int64,
        ),
        GenerationConfig(max_new_tokens=1),
    )


def test_empty_cache_snapshot() -> None:
    manager = create_manager()
    scheduler = RequestScheduler(
        max_running_requests=2,
        cache_manager=manager,
    )
    collector = RequestMetricsCollector()

    snapshot = capture_serving_metrics(
        collector,
        scheduler,
        manager.allocator,
    )

    assert snapshot.schema_version == 1
    assert snapshot.duration_unit == "seconds"
    assert snapshot.kv_cache.capacity_blocks == 4
    assert snapshot.kv_cache.allocated_blocks == 0
    assert snapshot.kv_cache.free_blocks == 4
    assert snapshot.kv_cache.utilization_ratio == 0.0
    assert snapshot.requests.waiting_current == 0
    assert snapshot.requests.running_current == 0


def test_partial_and_full_cache_snapshots() -> None:
    manager = create_manager()
    scheduler = RequestScheduler(
        max_running_requests=2,
        cache_manager=manager,
    )
    collector = RequestMetricsCollector()

    first = manager.allocator.allocate()
    second = manager.allocator.allocate()

    partial = capture_serving_metrics(
        collector,
        scheduler,
        manager.allocator,
    )

    assert partial.kv_cache.allocated_blocks == 2
    assert partial.kv_cache.free_blocks == 2
    assert partial.kv_cache.utilization_ratio == 0.5

    manager.allocator.allocate_many(2)

    full = capture_serving_metrics(
        collector,
        scheduler,
        manager.allocator,
    )

    assert full.kv_cache.allocated_blocks == 4
    assert full.kv_cache.free_blocks == 0
    assert full.kv_cache.utilization_ratio == 1.0

    manager.allocator.release(first)
    manager.allocator.release(second)


def test_cache_release_and_block_reuse() -> None:
    manager = create_manager()
    scheduler = RequestScheduler(
        max_running_requests=2,
        cache_manager=manager,
    )
    collector = RequestMetricsCollector()

    block_id = manager.allocator.allocate()
    manager.allocator.release(block_id)
    reused = manager.allocator.allocate()

    snapshot = capture_serving_metrics(
        collector,
        scheduler,
        manager.allocator,
    )

    assert reused == block_id
    assert snapshot.kv_cache.allocated_blocks == 1
    assert snapshot.kv_cache.free_blocks == 3


def test_scheduler_live_counts() -> None:
    manager = create_manager()
    scheduler = RequestScheduler(
        max_running_requests=1,
        cache_manager=manager,
    )
    collector = RequestMetricsCollector()
    request = create_request("A")

    collector.request_received("A")
    scheduler.submit(request)

    waiting = capture_serving_metrics(
        collector,
        scheduler,
        manager.allocator,
    )

    assert waiting.requests.waiting_current == 1
    assert waiting.requests.running_current == 0
    assert waiting.requests.active_metrics_records == 1

    scheduler.schedule()

    running = capture_serving_metrics(
        collector,
        scheduler,
        manager.allocator,
    )

    assert running.requests.waiting_current == 0
    assert running.requests.running_current == 1

    scheduler.cancel("A")
    collector.cancel_request("A")

    idle = capture_serving_metrics(
        collector,
        scheduler,
        manager.allocator,
    )

    assert idle.requests.waiting_current == 0
    assert idle.requests.running_current == 0
    assert idle.requests.cancelled_total == 1
