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
        - clinical_interpreter.py: Clinical interpretation generation
        - batch_statistics.py: Batch metrics calculation
        - observability_logger.py: W&B logging
    - inference_service.py: Main service facade
"""
int
# Re-export from components
from .internals import (
    InferenceEngine,
    DEFAULT_CHECKPOINT_PATH,
    ImageValidator,
    ImageProcessor,
    ClinicalInterpreter,
    BatchStatistics,
    ObservabilityLogger,
)

# Main service and singleton getters
from .inference_service import (
    InferenceService,
    get_inference_service,
    get_inference_engine,
    get_clinical_agent,
)

__all__ = [
    # Core engine
    "InferenceEngine",
    "DEFAULT_CHECKPOINT_PATH",
    # Components
    "ImageValidator",
    "ImageProcessor",
    "ClinicalInterpreter",
    "BatchStatistics",
    "ObservabilityLogger",
    # Main service
    "InferenceService",
    # Singleton getters
    "get_inference_service",
    "get_inference_engine",
    "get_clinical_agent",
]
