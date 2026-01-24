"""
X-ray image transformation components.
Provides strategy pattern-based transforms with builder for composable pipelines.
"""

from federated_pneumonia_detection.src.internals.transforms.base import (
    XRayPreprocessor,
)
from federated_pneumonia_detection.src.internals.transforms.builder import (
    TransformBuilder,
)
from federated_pneumonia_detection.src.internals.transforms.strategies import (
    CLAHEStrategy,
    ContrastStretchStrategy,
    EdgeEnhancementStrategy,
    XRayTransformStrategy,
)

__all__ = [
    "XRayPreprocessor",
    "TransformBuilder",
    "XRayTransformStrategy",
    "ContrastStretchStrategy",
    "CLAHEStrategy",
    "EdgeEnhancementStrategy",
]
