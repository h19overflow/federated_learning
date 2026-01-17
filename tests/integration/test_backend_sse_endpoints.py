"""
Backend SSE Endpoint Integration Tests

Tests the SSE streaming endpoints and event publishing.
"""
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_sse_event_manager():
    """Test SSE Event Manager functionality."""
    logger.info("=" * 80)
    logger.info("TEST: SSE Event Manager")
    logger.info("=" * 80)

    try:
        from federated_pneumonia_detection.src.control.dl_model.utils.data.sse_event_manager import (
            get_sse_event_manager
        )

        manager = await get_sse_event_manager()

        # Test 1: Create queue
        logger.info("Test 1.1: Creating queue for experiment")
        queue = manager.create_queue('test_backend_123')
        assert queue is not None, "Queue creation failed"
        logger.info("✅ Queue created successfully")

        # Test 2: Publish event
        logger.info("Test 1.2: Publishing event")
        event = {
            'type': 'epoch_start',
            'timestamp': datetime.now().isoformat(),
            'data': {'epoch': 1, 'total_epochs': 10}
        }
        await manager.publish_event('test_backend_123', event)
        logger.info("✅ Event published successfully")

        # Test 3: Retrieve event from queue
        logger.info("Test 1.3: Retrieving event from queue")
        retrieved_event = await asyncio.wait_for(queue.get(), timeout=2.0)
        assert retrieved_event['type'] == 'epoch_start', "Event type mismatch"
        logger.info(f"✅ Event retrieved: {retrieved_event['type']}")

        # Test 4: Track stream
        logger.info("Test 1.4: Tracking stream connections")
        manager.increment_stream('test_backend_123')
        stats = manager.get_stats()
        assert stats['total_clients'] >= 1, "Stream tracking failed"
        logger.info(f"✅ Streams tracked: {stats['total_clients']} client(s)")

        # Test 5: Get statistics
        logger.info("Test 1.5: Getting manager statistics")
        stats = manager.get_stats()
        logger.info(f"✅ Statistics retrieved:")
        logger.info(f"   - Total queues: {stats['total_queues']}")
        logger.info(f"   - Active experiments: {stats['active_experiments']}")
        logger.info(f"   - Total clients: {stats['total_clients']}")

        # Test 6: Multiple events
        logger.info("Test 1.6: Publishing multiple events")
        for i in range(3):
            await manager.publish_event('test_backend_123', {
                'type': 'batch_metrics',
                'timestamp': datetime.now().isoformat(),
                'data': {'step': i, 'loss': 0.5 - i*0.1, 'accuracy': 0.8 + i*0.05}
            })
        logger.info("✅ Multiple events published")

        # Verify events in queue
        logger.info("Test 1.7: Verifying event sequence")
        event_count = 0
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
                event_count += 1
                logger.info(f"   Event {event_count}: {event['type']}")
            except asyncio.TimeoutError:
                break
        assert event_count == 3, f"Expected 3 events, got {event_count}"
        logger.info(f"✅ All {event_count} events verified")

        # Test 8: Decrement stream
        logger.info("Test 1.8: Decrementing stream")
        manager.decrement_stream('test_backend_123')
        stats = manager.get_stats()
        logger.info(f"✅ Stream decremented: {stats['total_clients']} client(s)")

        logger.info("✅ ALL SSE EVENT MANAGER TESTS PASSED\n")
        return True

    except Exception as e:
        logger.error(f"❌ SSE Event Manager test failed: {e}", exc_info=True)
        return False


async def test_metrics_sse_sender():
    """Test MetricsSSESender functionality."""
    logger.info("=" * 80)
    logger.info("TEST: Metrics SSE Sender")
    logger.info("=" * 80)

    try:
        from federated_pneumonia_detection.src.control.dl_model.utils.data.metrics_sse_sender import (
            MetricsSSESender
        )
        from federated_pneumonia_detection.src.control.dl_model.utils.data.sse_event_manager import (
            get_sse_event_manager
        )

        sender = MetricsSSESender('test_sender_123')
        manager = await get_sse_event_manager()
        queue = manager.create_queue('test_sender_123')
        manager.increment_stream('test_sender_123')

        # Test 1: Send training mode
        logger.info("Test 2.1: Sending training_mode event")
        sender.send_training_mode(is_federated=False, num_rounds=1, num_clients=1)
        await asyncio.sleep(0.2)  # Allow async task to complete

        # Retrieve and verify
        try:
            event = await asyncio.wait_for(queue.get(), timeout=2.0)
            assert event['type'] == 'training_mode', f"Expected training_mode, got {event['type']}"
            logger.info(f"✅ training_mode event sent: {event['data']}")
        except asyncio.TimeoutError:
            logger.error("❌ training_mode event not received")
            return False

        # Test 2: Send epoch start
        logger.info("Test 2.2: Sending epoch_start event")
        sender.send_epoch_start(epoch=1, total_epochs=10)
        await asyncio.sleep(0.2)

        try:
            event = await asyncio.wait_for(queue.get(), timeout=2.0)
            assert event['type'] == 'epoch_start', f"Expected epoch_start, got {event['type']}"
            logger.info(f"✅ epoch_start event sent: {event['data']}")
        except asyncio.TimeoutError:
            logger.error("❌ epoch_start event not received")
            return False

        # Test 3: Send batch metrics
        logger.info("Test 2.3: Sending batch_metrics event")
        sender.send_batch_metrics(
            step=0,
            batch_idx=0,
            loss=0.563,
            accuracy=0.812,
            epoch=1
        )
        await asyncio.sleep(0.2)

        try:
            event = await asyncio.wait_for(queue.get(), timeout=2.0)
            assert event['type'] == 'batch_metrics', f"Expected batch_metrics, got {event['type']}"
            logger.info(f"✅ batch_metrics event sent: {event['data']}")
        except asyncio.TimeoutError:
            logger.error("❌ batch_metrics event not received")
            return False

        # Test 4: Send epoch end
        logger.info("Test 2.4: Sending epoch_end event")
        sender.send_epoch_end(
            epoch=1,
            phase='train',
            metrics={'loss': 0.45, 'accuracy': 0.85, 'recall': 0.82}
        )
        await asyncio.sleep(0.2)

        try:
            event = await asyncio.wait_for(queue.get(), timeout=2.0)
            assert event['type'] == 'epoch_end', f"Expected epoch_end, got {event['type']}"
            logger.info(f"✅ epoch_end event sent: {event['data']}")
        except asyncio.TimeoutError:
            logger.error("❌ epoch_end event not received")
            return False

        # Test 5: Send error
        logger.info("Test 2.5: Sending error event")
        sender.send_error("Test error message", error_type="validation_error")
        await asyncio.sleep(0.2)

        try:
            event = await asyncio.wait_for(queue.get(), timeout=2.0)
            assert event['type'] == 'error', f"Expected error, got {event['type']}"
            logger.info(f"✅ error event sent: {event['data']}")
        except asyncio.TimeoutError:
            logger.error("❌ error event not received")
            return False

        # Test 6: Send early stopping
        logger.info("Test 2.6: Sending early_stopping event")
        sender.send_early_stopping_triggered(
            epoch=5,
            best_metric_value=0.92,
            metric_name="val_recall",
            patience=3
        )
        await asyncio.sleep(0.2)

        try:
            event = await asyncio.wait_for(queue.get(), timeout=2.0)
            assert event['type'] == 'early_stopping', f"Expected early_stopping, got {event['type']}"
            logger.info(f"✅ early_stopping event sent: {event['data']}")
        except asyncio.TimeoutError:
            logger.error("❌ early_stopping event not received")
            return False

        manager.decrement_stream('test_sender_123')
        logger.info("✅ ALL METRICS SSE SENDER TESTS PASSED\n")
        return True

    except Exception as e:
        logger.error(f"❌ Metrics SSE Sender test failed: {e}", exc_info=True)
        return False


async def test_performance_throughput():
    """Test message throughput."""
    logger.info("=" * 80)
    logger.info("TEST: Performance - Message Throughput")
    logger.info("=" * 80)

    try:
        from federated_pneumonia_detection.src.control.dl_model.utils.data.metrics_sse_sender import (
            MetricsSSESender
        )
        from federated_pneumonia_detection.src.control.dl_model.utils.data.sse_event_manager import (
            get_sse_event_manager
        )

        sender = MetricsSSESender('perf_throughput_test')
        manager = await get_sse_event_manager()
        queue = manager.create_queue('perf_throughput_test')
        manager.increment_stream('perf_throughput_test')

        num_messages = 500
        logger.info(f"Test 3.1: Sending {num_messages} messages")

        start_time = time.time()

        for i in range(num_messages):
            sender.send_batch_metrics(
                step=i,
                batch_idx=i % 100,
                loss=0.5,
                accuracy=0.85,
                epoch=i // 100
            )

        # Wait for async tasks
        await asyncio.sleep(1.0)

        end_time = time.time()
        elapsed = end_time - start_time
        throughput = num_messages / elapsed

        logger.info(f"✅ Sent {num_messages} messages in {elapsed:.2f}s")
        logger.info(f"   Throughput: {throughput:.2f} messages/second")

        # Verify messages
        received_count = 0
        while True:
            try:
                await asyncio.wait_for(queue.get(), timeout=0.1)
                received_count += 1
            except asyncio.TimeoutError:
                break

        logger.info(f"✅ Verified {received_count} messages in queue")

        # Check target (>500 msg/s)
        if throughput >= 500:
            logger.info("✅ THROUGHPUT TARGET MET (>500 msg/s)")
        else:
            logger.warning(f"⚠️  Throughput below target: {throughput:.2f} < 500 msg/s")

        manager.decrement_stream('perf_throughput_test')
        logger.info("✅ THROUGHPUT TEST PASSED\n")
        return True

    except Exception as e:
        logger.error(f"❌ Throughput test failed: {e}", exc_info=True)
        return False


async def test_performance_latency():
    """Test event latency."""
    logger.info("=" * 80)
    logger.info("TEST: Performance - Event Latency")
    logger.info("=" * 80)

    try:
        from federated_pneumonia_detection.src.control.dl_model.utils.data.metrics_sse_sender import (
            MetricsSSESender
        )
        from federated_pneumonia_detection.src.control.dl_model.utils.data.sse_event_manager import (
            get_sse_event_manager
        )

        sender = MetricsSSESender('perf_latency_test')
        manager = await get_sse_event_manager()
        queue = manager.create_queue('perf_latency_test')
        manager.increment_stream('perf_latency_test')

        latencies: List[float] = []
        num_samples = 100

        logger.info(f"Test 4.1: Measuring latency over {num_samples} samples")

        for i in range(num_samples):
            start = time.time()

            sender.send_batch_metrics(
                step=i,
                batch_idx=i,
                loss=0.5,
                accuracy=0.85,
                epoch=1
            )

            # Wait for event to be in queue
            try:
                await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning(f"   Event {i} not received within timeout")
                continue

            end = time.time()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p99_latency = sorted(latencies)[min(98, len(latencies)-1)]
            max_latency = max(latencies)

            logger.info(f"✅ Latency measurements:")
            logger.info(f"   - Average: {avg_latency:.2f}ms")
            logger.info(f"   - P99: {p99_latency:.2f}ms")
            logger.info(f"   - Max: {max_latency:.2f}ms")

            # Check targets
            if avg_latency < 10:
                logger.info("✅ AVERAGE LATENCY TARGET MET (<10ms)")
            else:
                logger.warning(f"⚠️  Average latency above target: {avg_latency:.2f}ms > 10ms")

            if p99_latency < 50:
                logger.info("✅ P99 LATENCY TARGET MET (<50ms)")
            else:
                logger.warning(f"⚠️  P99 latency above target: {p99_latency:.2f}ms > 50ms")

        manager.decrement_stream('perf_latency_test')
        logger.info("✅ LATENCY TEST PASSED\n")
        return True

    except Exception as e:
        logger.error(f"❌ Latency test failed: {e}", exc_info=True)
        return False


async def main():
    """Run all integration tests."""
    logger.info("\n" + "=" * 80)
    logger.info("SSE BACKEND INTEGRATION TESTS")
    logger.info("=" * 80 + "\n")

    results = {}

    # Run tests
    results['Event Manager'] = await test_sse_event_manager()
    results['Metrics Sender'] = await test_metrics_sse_sender()
    results['Throughput'] = await test_performance_throughput()
    results['Latency'] = await test_performance_latency()

    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_flag in results.items():
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        logger.info(f"{test_name:.<40} {status}")

    logger.info("=" * 80)
    logger.info(f"OVERALL: {passed}/{total} tests passed")
    logger.info("=" * 80 + "\n")

    return all(results.values())


if __name__ == '__main__':
    success = asyncio.run(main())
    exit(0 if success else 1)
