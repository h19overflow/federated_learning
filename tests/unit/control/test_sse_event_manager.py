"""
Unit tests for SSE Event Manager.

Tests queue creation, event publishing, stream tracking, and cleanup.
Uses synchronous API (no asyncio) - compatible with Ray actors.
"""

import queue
import threading
import time

import pytest

from federated_pneumonia_detection.src.control.dl_model.internals.data.sse_event_manager import (
    SSEEventManager,
    get_sse_event_manager,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instance before each test."""
    SSEEventManager._instance = None
    import federated_pneumonia_detection.src.control.dl_model.internals.data.sse_event_manager as mod

    mod._manager_instance = None
    yield


def test_singleton_initialization():
    """Test that get_sse_event_manager returns singleton instance."""
    manager1 = get_sse_event_manager()
    manager2 = get_sse_event_manager()

    assert manager1 is manager2
    assert isinstance(manager1, SSEEventManager)


def test_queue_creation():
    """Test that queues are created correctly."""
    manager = get_sse_event_manager()
    q = manager.create_queue("test_exp_1")

    assert "test_exp_1" in manager.queues
    assert manager.active_streams["test_exp_1"] == 0
    assert q.maxsize == 1000
    assert isinstance(q, queue.Queue)


def test_queue_reuse():
    """Test that calling create_queue twice returns same queue."""
    manager = get_sse_event_manager()
    queue1 = manager.create_queue("test_exp_2")
    queue2 = manager.create_queue("test_exp_2")

    assert queue1 is queue2


def test_publish_event():
    """Test that events are published to queue (synchronous)."""
    manager = get_sse_event_manager()
    manager.create_queue("test_exp_3")

    event = {
        "type": "test_event",
        "timestamp": "2026-01-17T10:00:00",
        "data": {"value": 42},
    }

    success = manager.publish_event("test_exp_3", event)

    assert success is True
    q = manager.queues["test_exp_3"]
    assert q.qsize() == 1

    received_event = q.get_nowait()
    assert received_event["type"] == "test_event"
    assert received_event["data"]["value"] == 42


def test_publish_to_nonexistent_queue_creates_queue():
    """Test that publishing to non-existent queue creates it."""
    manager = get_sse_event_manager()

    exp_id = "test_exp_auto_create"
    event = {"type": "test", "timestamp": "2026-01-17T10:00:00", "data": {}}
    success = manager.publish_event(exp_id, event)

    assert success is True
    assert exp_id in manager.queues
    assert manager.queues[exp_id].qsize() == 1


def test_get_event():
    """Test get_event retrieves events from queue."""
    manager = get_sse_event_manager()
    manager.create_queue("test_get_event")

    event = {"type": "test", "timestamp": "2026-01-17T10:00:00", "data": {"x": 1}}
    manager.publish_event("test_get_event", event)

    retrieved = manager.get_event("test_get_event", timeout=0.1)
    assert retrieved is not None
    assert retrieved["data"]["x"] == 1


def test_get_event_timeout():
    """Test get_event returns None on timeout."""
    manager = get_sse_event_manager()
    manager.create_queue("test_timeout")

    start = time.time()
    result = manager.get_event("test_timeout", timeout=0.1)
    elapsed = time.time() - start

    assert result is None
    assert 0.09 < elapsed < 0.2


def test_get_event_nonexistent_queue():
    """Test get_event returns None for non-existent queue."""
    manager = get_sse_event_manager()
    result = manager.get_event("nonexistent_queue", timeout=0.01)
    assert result is None


def test_stream_tracking():
    """Test that stream connections are tracked."""
    manager = get_sse_event_manager()
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


def test_cleanup_scheduled_on_zero_streams():
    """Test that cleanup is scheduled when streams reach zero."""
    manager = get_sse_event_manager()
    manager.create_queue("test_exp_5")
    manager.increment_stream("test_exp_5")

    # Queue should exist
    assert "test_exp_5" in manager.queues

    # Decrement to zero - cleanup should be scheduled (spawns thread)
    manager.decrement_stream("test_exp_5")

    # Queue should still exist immediately (5 minute grace period)
    assert "test_exp_5" in manager.queues


def test_queue_full_returns_false():
    """Test that queue respects maxsize and returns False when full."""
    manager = get_sse_event_manager()
    manager.create_queue("test_exp_full")

    # Fill queue to capacity
    for i in range(1000):
        manager.queues["test_exp_full"].put_nowait({"type": "test", "data": {"i": i}})

    # Queue should be full
    assert manager.queues["test_exp_full"].full()

    # Attempt to publish should timeout and return False
    start = time.time()
    success = manager.publish_event("test_exp_full", {"type": "overflow", "data": {}})
    elapsed = time.time() - start

    assert success is False
    # Should have timed out at approximately 1 second
    assert 0.9 < elapsed < 1.5


def test_get_stats():
    """Test that get_stats returns correct statistics."""
    manager = get_sse_event_manager()

    # Create test queues
    manager.create_queue("test_stats_1")
    manager.create_queue("test_stats_2")

    # Add some connections
    manager.increment_stream("test_stats_1")
    manager.increment_stream("test_stats_2")
    manager.increment_stream("test_stats_2")

    # Add some events
    manager.publish_event("test_stats_1", {"type": "test", "data": {}})
    manager.publish_event("test_stats_1", {"type": "test", "data": {}})

    stats = manager.get_stats()

    assert "total_queues" in stats
    assert "active_experiments" in stats
    assert "total_clients" in stats
    assert "experiments" in stats

    # Verify counts include our test experiments
    assert stats["total_clients"] >= 3  # At least our 3 connections
    assert stats["active_experiments"] >= 2  # At least our 2 active experiments

    # Verify experiment details
    assert "test_stats_1" in stats["experiments"]
    exp1_stats = stats["experiments"]["test_stats_1"]
    assert exp1_stats["clients"] == 1
    assert exp1_stats["queue_size"] >= 2


def test_multiple_events_in_sequence():
    """Test publishing multiple events in sequence."""
    manager = get_sse_event_manager()
    manager.create_queue("test_exp_7")

    events = [
        {"type": "epoch_start", "data": {"epoch": 1}},
        {"type": "batch_metrics", "data": {"loss": 0.5}},
        {"type": "epoch_end", "data": {"epoch": 1, "metrics": {}}},
    ]

    for event in events:
        success = manager.publish_event("test_exp_7", event)
        assert success is True

    q = manager.queues["test_exp_7"]
    assert q.qsize() == 3

    # Verify events in order
    for expected_event in events:
        received = q.get_nowait()
        assert received["type"] == expected_event["type"]


def test_thread_safety():
    """Test that publish_event is thread-safe."""
    manager = get_sse_event_manager()
    manager.create_queue("test_thread_safe")

    num_threads = 10
    events_per_thread = 100
    results = []

    def publish_events(thread_id):
        successes = 0
        for i in range(events_per_thread):
            event = {"type": "test", "data": {"thread": thread_id, "i": i}}
            if manager.publish_event("test_thread_safe", event):
                successes += 1
        results.append(successes)

    threads = [
        threading.Thread(target=publish_events, args=(i,)) for i in range(num_threads)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All events should have been published
    total_successes = sum(results)
    assert total_successes == num_threads * events_per_thread


def test_concurrent_queue_creation():
    """Test that queue creation is thread-safe."""
    manager = get_sse_event_manager()
    queues_created = []

    def create_queue(exp_id):
        q = manager.create_queue(exp_id)
        queues_created.append((exp_id, q))

    threads = [
        threading.Thread(target=create_queue, args=(f"concurrent_exp_{i}",))
        for i in range(20)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All queues should be unique
    assert len(queues_created) == 20
    unique_queues = set(id(q) for _, q in queues_created)
    assert len(unique_queues) == 20
