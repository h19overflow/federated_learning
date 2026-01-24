"""Model inference module.

Contains the core inference engine and unified service for pneumonia detection.

Usage:
    from federated_pneumonia_detection.src.control.model_inferance import (
        InferenceService,
        get_inference_service,
    )

Structure:
    - components/: Modular helper classes
        - inference_engine.py: Core model loading and prediction
        - image_validator.py: File validation
        - image_processor.py: Image I/O
        - batch_statistics.py: Batch metrics calculation
        - observability_logger.py: W&B logging
    - inference_service.py: Main service facade
"""

# Re-export from components
# Main service and singleton getters
from .gradcam_service import GradCAMService
from .inference_service import (
    InferenceService,
    get_inference_engine,
    get_inference_service,
)
from .internals import (
    DEFAULT_CHECKPOINT_PATH,
    BatchStatistics,
    ImageProcessor,
    ImageValidator,
    InferenceEngine,
    ObservabilityLogger,
)

__all__ = [
    # Core engine
    "InferenceEngine",
    "DEFAULT_CHECKPOINT_PATH",
    # Components
    "ImageValidator",
    "ImageProcessor",
    "BatchStatistics",
    "ObservabilityLogger",
    # Main service
    "InferenceService",
    "GradCAMService",
    # Singleton getters
    "get_inference_service",
    "get_inference_engine",
]
