"""
API Middleware package.

This package contains FastAPI middleware components for the application.
"""

from federated_pneumonia_detection.src.api.middleware.security import (
    MaliciousPromptMiddleware,
)

__all__ = ["MaliciousPromptMiddleware"]
