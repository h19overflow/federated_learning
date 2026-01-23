"""
WebSocket Metrics Relay Server.

Provides real-time metrics broadcasting from training to frontend clients.
Runs in a background thread, independent of the main FastAPI event loop.

Usage:
    from federated_pneumonia_detection.src.api.endpoints.streaming.websocket_server import (
        start_websocket_server_thread,
    )

    # In lifespan handler:
    start_websocket_server_thread()
"""

import logging
import threading
from typing import Set

from federated_pneumonia_detection.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def start_websocket_server_thread() -> threading.Thread:
    """
    Start the WebSocket metrics relay server in a background thread.

    Returns:
        The started daemon thread running the WebSocket server.
    """
    thread = threading.Thread(
        target=_run_websocket_server,
        daemon=True,
        name="WebSocket-Server-Thread",
    )
    thread.start()
    logger.info("WebSocket server startup initiated in background thread")
    return thread


def _run_websocket_server() -> None:
    """
    Run the WebSocket server (blocking).

    This function is designed to run in a separate thread.
    """
    try:
        import asyncio
        import json

        import websockets

        connected_clients: Set[websockets.WebSocketServerProtocol] = set()

        async def handler(websocket: websockets.WebSocketServerProtocol) -> None:
            """
            Handle WebSocket connections and broadcast metrics to all clients.

            Uses per-message error handling to skip malformed messages gracefully.
            """
            connected_clients.add(websocket)
            logger.info(
                f"WebSocket client connected. Total clients: {len(connected_clients)}",
            )

            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        message_type = data.get("type", "unknown")

                        logger.debug(
                            f"Broadcasting {message_type} to {len(connected_clients)} clients",
                        )

                        await _broadcast_to_clients(
                            message,
                            websocket,
                            connected_clients,
                        )

                    except json.JSONDecodeError:
                        logger.warning(
                            "Received malformed JSON from client, skipping message",
                        )
                        continue
                    except KeyError as e:
                        logger.warning(
                            f"WebSocket message missing required field '{e}', skipping",
                        )
                        continue
                    except Exception as e:
                        logger.warning(
                            f"Unexpected error processing WebSocket message: {e}, continuing",
                        )
                        continue

            except websockets.exceptions.ConnectionClosed:
                logger.debug("Client closed WebSocket connection gracefully")
            except Exception as e:
                logger.warning(f"WebSocket handler unexpected error: {e}")
            finally:
                connected_clients.discard(websocket)
                logger.debug(
                    f"WebSocket client removed. Total clients: {len(connected_clients)}",
                )

        async def _broadcast_to_clients(
            message: str,
            sender: websockets.WebSocketServerProtocol,
            clients: Set[websockets.WebSocketServerProtocol],
        ) -> None:
            """Broadcast a message to all connected clients except the sender."""
            tasks = [client.send(message) for client in clients if client != sender]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        async def run_server() -> None:
            """Run the WebSocket server."""
            logger.info(
                f"Starting WebSocket server on {settings.websocket_uri}",
            )

            async with websockets.serve(
                handler,
                settings.WEBSOCKET_HOST,
                settings.WEBSOCKET_PORT,
            ):
                logger.info("[OK] WebSocket metrics server is running")
                await asyncio.Future()  # Run forever

        asyncio.run(run_server())

    except ImportError:
        logger.warning(
            "WebSocket server failed to start: Missing required library. "
            "Install with: pip install websockets (metrics streaming will be unavailable)",
        )
    except OSError as e:
        logger.warning(
            f"WebSocket server could not bind to port: {e}. "
            f"(metrics streaming unavailable, but app continues)",
        )
    except Exception as e:
        logger.warning(
            f"WebSocket server startup had unexpected error: {e} "
            f"(metrics streaming will be unavailable)",
        )
