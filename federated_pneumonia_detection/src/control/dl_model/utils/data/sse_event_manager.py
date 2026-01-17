"""
SSE Event Manager for Training Metrics Streaming.

Manages per-experiment event queues and connection tracking.
Provides thread-safe publishing and subscription with automatic cleanup.
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SSEEventManager:
    """
    Manages SSE event queues for training experiments.

    Provides thread-safe publishing and subscription to per-experiment
    event streams. Automatically cleans up inactive queues.
    """

    _instance: Optional['SSEEventManager'] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}
        self.active_streams: Dict[str, int] = {}
        self.queue_created_at: Dict[str, datetime] = {}
        self.logger = logging.getLogger(__name__)

    @classmethod
    async def get_instance(cls) -> 'SSEEventManager':
        """Get singleton instance (async-safe)."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.info("SSE Event Manager initialized")
        return cls._instance

    def create_queue(self, experiment_id: str) -> asyncio.Queue:
        """
        Create or retrieve event queue for experiment.

        Args:
            experiment_id: Unique experiment identifier

        Returns:
            asyncio.Queue for event streaming
        """
        if experiment_id not in self.queues:
            self.queues[experiment_id] = asyncio.Queue(maxsize=1000)
            self.active_streams[experiment_id] = 0
            self.queue_created_at[experiment_id] = datetime.now()
            self.logger.info(f"Created event queue for experiment: {experiment_id}")
        return self.queues[experiment_id]

    async def publish_event(self, experiment_id: str, event: Dict[str, Any]):
        """
        Publish event to experiment's queue.

        Args:
            experiment_id: Target experiment
            event: Event dictionary with 'type', 'timestamp', 'data' keys
        """
        if experiment_id not in self.queues:
            self.logger.warning(
                f"Attempted to publish to non-existent queue: {experiment_id}. "
                "Creating queue on demand."
            )
            self.create_queue(experiment_id)

        try:
            await asyncio.wait_for(
                self.queues[experiment_id].put(event),
                timeout=1.0
            )
            self.logger.debug(
                f"Published {event.get('type')} event to {experiment_id}"
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                f"Event queue full for {experiment_id}, dropping event: {event.get('type')}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to publish event to {experiment_id}: {e}",
                exc_info=True
            )

    def increment_stream(self, experiment_id: str):
        """Track new client connection."""
        self.active_streams[experiment_id] = self.active_streams.get(experiment_id, 0) + 1
        self.logger.info(
            f"Client connected to {experiment_id}. "
            f"Active streams: {self.active_streams[experiment_id]}"
        )

    def decrement_stream(self, experiment_id: str):
        """Track client disconnection and schedule cleanup."""
        if experiment_id in self.active_streams:
            self.active_streams[experiment_id] -= 1
            self.logger.info(
                f"Client disconnected from {experiment_id}. "
                f"Active streams: {self.active_streams[experiment_id]}"
            )

            if self.active_streams[experiment_id] == 0:
                self.logger.info(
                    f"No active streams for {experiment_id}. "
                    "Scheduling cleanup in 5 minutes."
                )
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._cleanup_after_delay(experiment_id, delay=300))
                except RuntimeError:
                    # No running event loop - this is fine, cleanup will happen on next connection
                    self.logger.debug(
                        f"No event loop running, skipping cleanup scheduling for {experiment_id}"
                    )

    async def _cleanup_after_delay(self, experiment_id: str, delay: int = 300):
        """
        Clean up inactive queue after delay.

        Args:
            experiment_id: Experiment to clean up
            delay: Wait time in seconds (default: 5 minutes)
        """
        await asyncio.sleep(delay)

        if self.active_streams.get(experiment_id, 0) == 0:
            self.logger.info(f"Cleaning up inactive queue: {experiment_id}")
            self.queues.pop(experiment_id, None)
            self.active_streams.pop(experiment_id, None)
            self.queue_created_at.pop(experiment_id, None)
        else:
            self.logger.info(
                f"Queue {experiment_id} became active again, skipping cleanup"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics for monitoring."""
        return {
            "total_queues": len(self.queues),
            "active_experiments": len([c for c in self.active_streams.values() if c > 0]),
            "total_clients": sum(self.active_streams.values()),
            "experiments": {
                exp_id: {
                    "clients": self.active_streams.get(exp_id, 0),
                    "queue_size": self.queues[exp_id].qsize() if exp_id in self.queues else 0,
                    "created_at": self.queue_created_at.get(exp_id).isoformat() if exp_id in self.queue_created_at else None,
                }
                for exp_id in self.queues.keys()
            }
        }


_manager_instance: Optional[SSEEventManager] = None


async def get_sse_event_manager() -> SSEEventManager:
    """Get global SSE event manager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = await SSEEventManager.get_instance()
    return _manager_instance
