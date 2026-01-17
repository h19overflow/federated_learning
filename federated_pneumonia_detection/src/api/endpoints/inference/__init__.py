"""Inference endpoints for X-ray pneumonia detection.

This module provides API endpoints for running inference on chest X-ray images.
Uses layered architecture:
- Control layer: InferenceEngine (core logic), ClinicalInterpretationAgent (LLM agent)
- API layer: This module (endpoints + schemas), deps.py (singleton management)
"""

from .health_endpoints import router as health_router
from .prediction_endpoints import router as prediction_router
from .batch_prediction_endpoints import router as batch_prediction_router

__all__ = ["health_router", "prediction_router", "batch_prediction_router"]
