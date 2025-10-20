from typing import Dict, Set
import asyncio
from fastapi import WebSocket
from datetime import datetime
from federated_pneumonia_detection.src.utils.loggers.logger import logger

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
