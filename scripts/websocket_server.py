#!/usr/bin/env python3
"""
Simple WebSocket server for broadcasting training metrics.

This server receives metric messages from the backend training loop
and broadcasts them to all connected frontend clients.

Run with:
    python scripts/websocket_server.py

Then metrics from training will be sent to ws://localhost:8765
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Store all connected clients
connected_clients: Set[websockets.WebSocketServerProtocol] = set()


async def handler(websocket: websockets.WebSocketServerProtocol, path: str) -> None:
    """
    Handle individual WebSocket connections.
    
    Receives metrics from backend and broadcasts to all connected clients.
    """
    # Add new client
    connected_clients.add(websocket)
    logger.info(f"Client connected. Total clients: {len(connected_clients)}")
    
    try:
        async for message in websocket:
            try:
                # Parse incoming metric message
                data = json.loads(message)
                message_type = data.get('type', 'unknown')
                _timestamp = data.get('timestamp', datetime.now().isoformat())
                
                logger.info(
                    f"Received {message_type} from backend - "
                    f"Broadcasting to {len(connected_clients)} clients"
                )
                
                # Broadcast to all connected clients
                if connected_clients:
                    # Send to all clients except the sender
                    tasks = []
                    for client in connected_clients:
                        if client != websocket:
                            tasks.append(client.send(message))
                    
                    # Execute all sends concurrently
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Handler error: {e}")
    finally:
        # Remove client
        connected_clients.discard(websocket)
        logger.info(f"Client removed. Total clients: {len(connected_clients)}")


async def main() -> None:
    """
    Start the WebSocket server.
    
    Listens on ws://localhost:8765
    """
    host = "localhost"
    port = 8765
    
    logger.info(f"Starting WebSocket server on ws://{host}:{port}")
    
    async with websockets.serve(handler, host, port):
        logger.info("✓ WebSocket server is running and ready for connections")
        logger.info("Press Ctrl+C to stop")
        
        try:
            # Keep server running
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            logger.info("Shutting down server...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped")
