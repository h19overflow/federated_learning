"""
WebSocket endpoint for real-time training progress streaming.

This module provides WebSocket connections for broadcasting real-time training
progress updates to connected frontend clients. It integrates with the ProgressLogger
to stream training events as they occur.

WebSocket Message Format:
{
    "type": "epoch_start" | "epoch_end" | "status" | "error" | "round_start" | "round_end",
    "data": {
        "epoch": int,
        "total_epochs": int,
        "metrics": {...},
        ...
    },
    "timestamp": ISO8601 string
}

Dependencies:
- fastapi.WebSocket: WebSocket communication
- asyncio: Async event handling
- json: Message serialization
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
from datetime import datetime

from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
from federated_pneumonia_detection.src.utils.connection_manager import ConnectionManager

router = APIRouter(
    prefix="/ws",
    tags=["websocket", "logging"],
)

logger = get_logger(__name__)


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/training-progress/{experiment_id}")
async def websocket_training_progress(websocket: WebSocket, experiment_id: str):
    """
    WebSocket endpoint for real-time training progress updates.

    Clients connect to this endpoint to receive live updates about training
    progress for a specific experiment. The connection remains open for the
    duration of the training or until the client disconnects.

    **URL:** `/ws/training-progress/{experiment_id}`

    **Parameters:**
    - `experiment_id`: Unique identifier for the experiment to monitor

    **Message Types Sent:**
    - `connected`: Connection confirmation
    - `epoch_start`: Training epoch started
    - `epoch_end`: Training epoch completed with metrics
    - `round_start`: Federated round started
    - `round_end`: Federated round completed
    - `status`: Training status change (started, running, completed, failed)
    - `error`: Training error occurred

    **Example Client Code (JavaScript):**
    ```javascript
    const ws = new WebSocket("ws://localhost:8000/ws/training-progress/exp123");

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        console.log(`${message.type}:`, message.data);
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
    };

    ws.onclose = () => {
        console.log("WebSocket closed");
    };
    ```

    **Note:**
    This endpoint maintains a persistent connection. The backend can push
    messages at any time during training. Clients should handle reconnection
    logic if the connection drops.
    """
    await manager.connect(websocket, experiment_id)

    try:
        # Keep connection alive and handle incoming messages
        while True:
            # Receive messages from client (e.g., ping/pong, commands)
            data = await websocket.receive_text()

            # Handle client commands if needed
            try:
                command = json.loads(data)
                if command.get("type") == "ping":
                    await manager.send_personal_message(
                        {
                            "type": "pong",
                            "timestamp": datetime.now().isoformat(),
                        },
                        websocket,
                    )
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {data}")

    except WebSocketDisconnect:
        await manager.disconnect(websocket, experiment_id)
        logger.info(f"Client disconnected from experiment {experiment_id}")
    except Exception as e:
        logger.error(f"WebSocket error for experiment {experiment_id}: {e}")
        await manager.disconnect(websocket, experiment_id)


async def broadcast_progress_event(
    experiment_id: str,
    event_type: str,
    data: dict,
):
    """
    Utility function to broadcast training progress events.

    This function should be called by training processes to send updates
    to all connected WebSocket clients.

    Args:
        experiment_id: Experiment identifier
        event_type: Type of event (epoch_start, epoch_end, status, etc.)
        data: Event data payload

    Example:
        await broadcast_progress_event(
            "exp123",
            "epoch_end",
            {
                "epoch": 5,
                "total_epochs": 10,
                "metrics": {"loss": 0.25, "accuracy": 0.92}
            }
        )
    """
    message = {
        "type": event_type,
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }

    await manager.broadcast(message, experiment_id)
    logger.debug(f"Broadcasted {event_type} for experiment {experiment_id}")
