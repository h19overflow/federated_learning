import logging

from federated_pneumonia_detection.src.api.endpoints.streaming.websocket_server import (
    start_websocket_server_thread,
)

logger = logging.getLogger(__name__)

def initialize_websocket_server() -> None:
    """Initialize WebSocket metrics server in a background thread."""
    try:
        start_websocket_server_thread()
        logger.info("WebSocket metrics server thread started")
    except Exception as e:
        logger.warning(f"Failed to start WebSocket metrics server: {e}")
