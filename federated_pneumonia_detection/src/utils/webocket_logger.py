import logging

class ConnectionManager:
    def __init__(self):
        self.active_connections = []
    async def connect(self, websocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket):
        self.active_connections.remove(websocket)
    async def broadcast(self, message):
        for conn in self.active_connections:
            await conn.send_text(message)


class WebSocketLogHandler(logging.Handler):
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
    def emit(self, record):
        log_entry = self.format(record)
        # Schedule sending log to all clients (async)
        import asyncio
        asyncio.create_task(self.manager.broadcast(log_entry))
