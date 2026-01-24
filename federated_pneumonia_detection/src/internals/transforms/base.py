"""
Base preprocessor for X-ray images.
"""

import logging
from typing import Callable, List, Optional

from PIL import Image

from federated_pneumonia_detection.src.internals.loggers.logger import get_logger
from federated_pneumonia_detection.src.internals.transforms.strategies import (
    CLAHEStrategy,
    ContrastStretchStrategy,
    EdgeEnhancementStrategy,
    XRayTransformStrategy,
)


class XRayPreprocessor:
    """
    Custom preprocessing utilities for X-ray images.
    Provides contrast enhancement and other X-ray specific preprocessing.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize preprocessor."""
        self.logger = logger or get_logger(__name__)

    @staticmethod
    def contrast_stretch_percentile(
        image: Image.Image,
        lower_percentile: float = 5.0,
        upper_percentile: float = 95.0,
        eps: float = 1e-8,
    ) -> Image.Image:
        """Apply percentile-based contrast stretching."""
        return ContrastStretchStrategy(lower_percentile, upper_percentile, eps).apply(
            image
        )

    @staticmethod
    def adaptive_histogram_equalization(
        image: Image.Image,
        clip_limit: float = 2.0,
    ) -> Image.Image:
        """Apply adaptive histogram equalization (CLAHE)."""
        return CLAHEStrategy(clip_limit).apply(image)

    @staticmethod
    def edge_enhancement(image: Image.Image, strength: float = 1.0) -> Image.Image:
        """Apply edge enhancement."""
        return EdgeEnhancementStrategy(strength).apply(image)

    def create_custom_preprocessing_pipeline(
        self,
        contrast_stretch: bool = True,
        adaptive_hist: bool = False,
        edge_enhance: bool = False,
        **kwargs,
    ) -> Callable[[Image.Image], Image.Image]:
        """Create custom preprocessing pipeline for X-ray images."""
        strategies: List[XRayTransformStrategy] = []

        if contrast_stretch:
            strategies.append(
                ContrastStretchStrategy(
                    lower_percentile=kwargs.get("lower_percentile", 5.0),
                    upper_percentile=kwargs.get("upper_percentile", 95.0),
                    eps=kwargs.get("eps", 1e-8),
                ),
            )

        if adaptive_hist:
            strategies.append(CLAHEStrategy(clip_limit=kwargs.get("clip_limit", 2.0)))

        if edge_enhance:
            strategies.append(
                EdgeEnhancementStrategy(strength=kwargs.get("edge_strength", 1.0)),
            )

        def preprocess_fn(image: Image.Image) -> Image.Image:
            try:
                result = image
                for strategy in strategies:
                    result = strategy.apply(result)
                return result
            except Exception as e:
                self.logger.warning(
                    f"Preprocessing failed, returning original image: {e}",
                )
                return image

        return preprocess_fn
