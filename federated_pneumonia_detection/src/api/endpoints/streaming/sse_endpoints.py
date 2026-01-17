"""
SSE Streaming Endpoints for Training Metrics.

Provides real-time training metrics streaming via Server-Sent Events.
"""
import json
import asyncio
import logging
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

    event_manager = await get_sse_event_manager()
    queue = event_manager.create_queue(experiment_id)
    event_manager.increment_stream(experiment_id)

    async def event_generator():
        """
        Async generator for SSE events.

        Yields formatted SSE messages from the event queue.
        Implements keepalive comments and graceful disconnection.
        """
        try:
            initial_event = {
                "experiment_id": experiment_id,
                "timestamp": asyncio.get_event_loop().time()
            }
            yield f"event: connected\ndata: {json.dumps(initial_event)}\n\n"
            logger.info(f"SSE connection established for: {experiment_id}")

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)

                    event_type = event.get('type', 'message')
                    event_data = json.dumps(event.get('data', {}))

                    yield f"event: {event_type}\ndata: {event_data}\n\n"

                    logger.debug(
                        f"Sent {event_type} event to {experiment_id} "
                        f"(queue size: {queue.qsize()})"
                    )

                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    logger.debug(f"Sent keepalive to {experiment_id}")

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
    event_manager = await get_sse_event_manager()
    return event_manager.get_stats()
