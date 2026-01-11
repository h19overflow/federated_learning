"""Inference endpoints for X-ray pneumonia detection.

This module provides the API endpoints for running inference on chest X-ray images.
Uses the layered architecture:
- Control layer: InferenceEngine (core logic)
- Agentic layer: ClinicalInterpretationAgent (LLM agent)
- Boundary layer: InferenceService (service abstraction)
- API layer: This module (endpoints + schemas)
"""

from .inference_endpoints import router

__all__ = ["router"]
