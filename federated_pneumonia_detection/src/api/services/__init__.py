"""
API Services module.

Provides startup and shutdown service orchestration.
"""

from federated_pneumonia_detection.src.api.services.startup import (
    initialize_services,
    shutdown_services,
)

__all__ = ["initialize_services", "shutdown_services"]
