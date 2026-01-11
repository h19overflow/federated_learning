"""Model inference module.

Contains the core inference engine for pneumonia detection.
"""

from .inference_engine import InferenceEngine, DEFAULT_CHECKPOINT_PATH

__all__ = ["InferenceEngine", "DEFAULT_CHECKPOINT_PATH"]
