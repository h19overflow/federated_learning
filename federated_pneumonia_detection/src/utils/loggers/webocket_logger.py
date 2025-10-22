"""
Backward compatibility module for WebSocket logging.

This module provides aliases for WebSocket-related imports referenced in the
WEBSOCKET_INTEGRATION_GUIDE, where ConnectionManager is imported from here.
"""

# Import ConnectionManager from its actual location
from federated_pneumonia_detection.src.utils.connection_manager import (
    ConnectionManager,
)

__all__ = ["ConnectionManager"]
