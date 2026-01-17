"""
SSE Streaming Endpoints for Training Metrics.

Provides real-time training metrics streaming via Server-Sent Events.
Uses polling with asyncio.sleep() to read from thread-safe queue.Queue.
"""
import json
import asyncio
import logging
import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from federated_pneumonia_detection.src.control.dl_model.utils.data.sse_event_manager import (
    get_sse_event_manager
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["streaming"])


@router.get("/stream/{experiment_id}")
async def stream_training_metrics(experiment_id: str):
    """
    Server-Sent Events endpoint for streaming training metrics.

    Opens a persistent HTTP connection and streams events as they occur.
    Automatically handles keepalive and disconnection cleanup.

    Args:
        experiment_id: Unique identifier for the training experiment

    Returns:
        StreamingResponse with text/event-stream content type
    """
    logger.info(f"SSE connection requested for experiment: {experiment_id}")

    event_manager = get_sse_event_manager()
    event_manager.create_queue(experiment_id)
    event_manager.increment_stream(experiment_id)

    async def event_generator():
        """
        Async generator for SSE events.

        Yields formatted SSE messages from the event queue using polling.
        Implements keepalive comments and graceful disconnection.
        """
        try:
            initial_event = {
                "experiment_id": experiment_id,
                "timestamp": time.time()
            }
            yield f"event: connected\ndata: {json.dumps(initial_event)}\n\n"
            logger.info(f"SSE connection established for: {experiment_id}")

            keepalive_interval = 30.0  # seconds
            poll_interval = 0.1  # seconds
            last_keepalive = time.time()

            while True:
                # Poll for events with short timeout
                event = event_manager.get_event(experiment_id, timeout=poll_interval)

                if event is not None:
                    event_type = event.get('type', 'message')
                    event_data = json.dumps(event.get('data', {}))

                    yield f"event: {event_type}\ndata: {event_data}\n\n"

                    logger.debug(
                        f"Sent {event_type} event to {experiment_id}"
                    )

                # Check if keepalive is needed
                current_time = time.time()
                if current_time - last_keepalive >= keepalive_interval:
                    yield ": keepalive\n\n"
                    logger.debug(f"Sent keepalive to {experiment_id}")
                    last_keepalive = current_time

                # Yield control to event loop
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info(f"SSE connection cancelled for: {experiment_id}")
        except Exception as e:
            logger.error(
                f"Error in SSE stream for {experiment_id}: {e}",
                exc_info=True
            )
            error_data = json.dumps({"error": str(e), "experiment_id": experiment_id})
            yield f"event: error\ndata: {error_data}\n\n"
        finally:
            event_manager.decrement_stream(experiment_id)
            logger.info(f"SSE connection closed for: {experiment_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@router.get("/stream/stats")
async def get_streaming_stats():
    """
    Get SSE manager statistics for monitoring.

    Returns:
        Dictionary with queue counts, active connections, etc.
    """
    event_manager = get_sse_event_manager()
    return event_manager.get_stats()
