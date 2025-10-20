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
from typing import Dict, List, Set
import asyncio
import json
from datetime import datetime
from pathlib import Path

from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

router = APIRouter(
    prefix="/ws",
    tags=["websocket", "logging"],
)

logger = get_logger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for training progress broadcasting.

    Maintains a registry of active connections per experiment and provides
    methods to broadcast messages to all connected clients for a specific
    experiment.
    """

    def __init__(self):
        """Initialize connection manager with empty registry."""
        # Map experiment_id -> Set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, experiment_id: str):
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: WebSocket connection to register
            experiment_id: Experiment identifier for this connection
        """
        await websocket.accept()

        async with self._lock:
            if experiment_id not in self.active_connections:
                self.active_connections[experiment_id] = set()
            self.active_connections[experiment_id].add(websocket)

        logger.info(
            f"WebSocket connected for experiment {experiment_id}. "
            f"Total connections: {len(self.active_connections[experiment_id])}"
        )

        # Send connection confirmation
        await self.send_personal_message(
            {
                "type": "connected",
                "data": {"experiment_id": experiment_id},
                "timestamp": datetime.now().isoformat(),
            },
            websocket,
        )

    async def disconnect(self, websocket: WebSocket, experiment_id: str):
        """
        Remove a WebSocket connection from the registry.

        Args:
            websocket: WebSocket connection to remove
            experiment_id: Experiment identifier
        """
        async with self._lock:
            if experiment_id in self.active_connections:
                self.active_connections[experiment_id].discard(websocket)

                # Clean up empty experiment entries
                if not self.active_connections[experiment_id]:
                    del self.active_connections[experiment_id]

        logger.info(f"WebSocket disconnected for experiment {experiment_id}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """
        Send message to a specific WebSocket connection.

        Args:
            message: Message dictionary to send
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def broadcast(self, message: dict, experiment_id: str):
        """
        Broadcast message to all connections for an experiment.

        Args:
            message: Message dictionary to broadcast
            experiment_id: Target experiment ID
        """
        if experiment_id not in self.active_connections:
            return

        # Create a copy to avoid modification during iteration
        connections = list(self.active_connections.get(experiment_id, set()))

        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(
                    f"Error broadcasting to connection: {e}. Removing connection."
                )
                await self.disconnect(connection, experiment_id)


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
