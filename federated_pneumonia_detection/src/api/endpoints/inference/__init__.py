"""Inference endpoints for X-ray pneumonia detection.

This module provides API endpoints for running inference on chest X-ray images.
Uses layered architecture:
- Control layer: InferenceEngine (core logic)
- Boundary layer: InferenceService (service abstraction)
- API layer: This module (endpoints + schemas)
"""

from .batch_prediction_endpoints import router as batch_prediction_router
from .gradcam_endpoints import router as gradcam_router
from .health_endpoints import router as health_router
from .single_prediction_endpoint import router as prediction_router

__all__ = [
    "health_router",
    "prediction_router",
    "batch_prediction_router",
    "gradcam_router",
]
