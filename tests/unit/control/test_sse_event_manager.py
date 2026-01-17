"""
Unit tests for SSE Event Manager.

Tests queue creation, event publishing, stream tracking, and cleanup.
"""
import pytest
import asyncio
from federated_pneumonia_detection.src.control.dl_model.utils.data.sse_event_manager import (
    get_sse_event_manager,
    SSEEventManager
)


@pytest.mark.asyncio
async def test_singleton_initialization():
    """Test that get_sse_event_manager returns singleton instance."""
    manager1 = await get_sse_event_manager()
    manager2 = await get_sse_event_manager()

    assert manager1 is manager2
    assert isinstance(manager1, SSEEventManager)


@pytest.mark.asyncio
async def test_queue_creation():
    """Test that queues are created correctly."""
    manager = await get_sse_event_manager()
    queue = manager.create_queue("test_exp_1")

    assert "test_exp_1" in manager.queues
    assert manager.active_streams["test_exp_1"] == 0
    assert queue.maxsize == 1000


@pytest.mark.asyncio
async def test_queue_reuse():
    """Test that calling create_queue twice returns same queue."""
    manager = await get_sse_event_manager()
    queue1 = manager.create_queue("test_exp_2")
    queue2 = manager.create_queue("test_exp_2")

    assert queue1 is queue2
    assert len(manager.queues) == len([k for k in manager.queues.keys() if "test_exp" in k])


@pytest.mark.asyncio
async def test_publish_event():
    """Test that events are published to queue."""
    manager = await get_sse_event_manager()
    manager.create_queue("test_exp_3")

    event = {
        "type": "test_event",
        "timestamp": "2026-01-16T10:00:00",
        "data": {"value": 42}
    }

    await manager.publish_event("test_exp_3", event)

    queue = manager.queues["test_exp_3"]
    assert queue.qsize() == 1

    received_event = await queue.get()
    assert received_event["type"] == "test_event"
    assert received_event["data"]["value"] == 42


@pytest.mark.asyncio
async def test_publish_to_nonexistent_queue_creates_queue():
    """Test that publishing to non-existent queue creates it."""
    manager = await get_sse_event_manager()

    # Ensure queue doesn't exist
    exp_id = "test_exp_auto_create"
    if exp_id in manager.queues:
        manager.queues.pop(exp_id)

    event = {"type": "test", "timestamp": "2026-01-16T10:00:00", "data": {}}
    await manager.publish_event(exp_id, event)

    assert exp_id in manager.queues
    assert manager.queues[exp_id].qsize() == 1


@pytest.mark.asyncio
async def test_stream_tracking():
    """Test that stream connections are tracked."""
    manager = await get_sse_event_manager()
    manager.create_queue("test_exp_4")

    assert manager.active_streams["test_exp_4"] == 0

    manager.increment_stream("test_exp_4")
    assert manager.active_streams["test_exp_4"] == 1

    manager.increment_stream("test_exp_4")
    assert manager.active_streams["test_exp_4"] == 2

    manager.decrement_stream("test_exp_4")
    assert manager.active_streams["test_exp_4"] == 1

    manager.decrement_stream("test_exp_4")
    assert manager.active_streams["test_exp_4"] == 0


@pytest.mark.asyncio
async def test_cleanup_scheduled_on_zero_streams():
    """Test that cleanup is scheduled when streams reach zero."""
    manager = await get_sse_event_manager()
    manager.create_queue("test_exp_5")
    manager.increment_stream("test_exp_5")

    # Queue should exist
    assert "test_exp_5" in manager.queues

    # Decrement to zero - cleanup should be scheduled
    manager.decrement_stream("test_exp_5")

    # Queue should still exist immediately (grace period)
    assert "test_exp_5" in manager.queues


@pytest.mark.asyncio
async def test_queue_maxsize_timeout():
    """Test that queue respects maxsize and drops events on timeout."""
    manager = await get_sse_event_manager()
    queue = manager.create_queue("test_exp_6")

    # Fill queue to capacity
    for i in range(1000):
        await queue.put({"type": "test", "data": {"i": i}})

    # Queue should be full
    assert queue.full()

    # Attempt to publish should timeout (1s timeout in code)
    start = asyncio.get_event_loop().time()
    await manager.publish_event("test_exp_6", {"type": "overflow", "data": {}})
    elapsed = asyncio.get_event_loop().time() - start

    # Should have timed out at approximately 1 second
    assert 0.9 < elapsed < 1.5


@pytest.mark.asyncio
async def test_get_stats():
    """Test that get_stats returns correct statistics."""
    manager = await get_sse_event_manager()

    # Create test queues
    manager.create_queue("test_stats_1")
    manager.create_queue("test_stats_2")

    # Add some connections
    manager.increment_stream("test_stats_1")
    manager.increment_stream("test_stats_2")
    manager.increment_stream("test_stats_2")

    # Add some events
    await manager.publish_event("test_stats_1", {"type": "test", "data": {}})
    await manager.publish_event("test_stats_1", {"type": "test", "data": {}})

    stats = manager.get_stats()

    assert "total_queues" in stats
    assert "active_experiments" in stats
    assert "total_clients" in stats
    assert "experiments" in stats

    # Verify counts include our test experiments
    assert stats["total_clients"] >= 3  # At least our 3 connections
    assert stats["active_experiments"] >= 2  # At least our 2 active experiments

    # Verify experiment details
    if "test_stats_1" in stats["experiments"]:
        exp1_stats = stats["experiments"]["test_stats_1"]
        assert exp1_stats["clients"] == 1
        assert exp1_stats["queue_size"] >= 2


@pytest.mark.asyncio
async def test_multiple_events_in_sequence():
    """Test publishing multiple events in sequence."""
    manager = await get_sse_event_manager()
    manager.create_queue("test_exp_7")

    events = [
        {"type": "epoch_start", "data": {"epoch": 1}},
        {"type": "batch_metrics", "data": {"loss": 0.5}},
        {"type": "epoch_end", "data": {"epoch": 1, "metrics": {}}}
    ]

    for event in events:
        await manager.publish_event("test_exp_7", event)

    queue = manager.queues["test_exp_7"]
    assert queue.qsize() == 3

    # Verify events in order
    for expected_event in events:
        received = await queue.get()
        assert received["type"] == expected_event["type"]
