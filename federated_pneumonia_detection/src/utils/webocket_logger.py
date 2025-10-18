import logging

class ConnectionManager:
    """Manager for the WebSocket connections."""
    def __init__(self):
        """Initialize the ConnectionManager."""
        self.active_connections = []
    async def connect(self, websocket):
        """Connect the WebSocket."""
        await websocket.accept()
        self.active_connections.append(websocket)   
    def disconnect(self, websocket):
        """Disconnect the WebSocket."""
        self.active_connections.remove(websocket)
    async def broadcast(self, message):
        """Broadcast the message to all the WebSocket connections."""
        for conn in self.active_connections:
            await conn.send_text(message)


class WebSocketLogHandler(logging.Handler):
    """Handler for the WebSocket logging."""
    def __init__(self, manager):
        """Initialize the WebSocketLogHandler."""
        super().__init__()
        self.manager = manager
    def emit(self, record):
        """Emit the log entry to the WebSocket."""
        log_entry = self.format(record)
        # Schedule sending log to all clients (async)
        import asyncio
        asyncio.create_task(self.manager.broadcast(log_entry))
