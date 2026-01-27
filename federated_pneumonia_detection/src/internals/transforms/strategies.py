"""
Transformation strategies for X-ray images.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class XRayTransformStrategy(ABC):
    """Abstract base class for X-ray transformation strategies."""

    @abstractmethod
    def apply(self, image: Image.Image) -> Image.Image:
        """Apply the transformation to the image."""
        pass


class ContrastStretchStrategy(XRayTransformStrategy):
    """Strategy for percentile-based contrast stretching."""

    def __init__(
        self,
        lower_percentile: float = 5.0,
        upper_percentile: float = 95.0,
        eps: float = 1e-8,
    ):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.eps = eps

    def apply(self, image: Image.Image) -> Image.Image:
        # Convert to numpy array and normalize to [0,1]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Calculate percentiles
        p_lower, p_upper = np.percentile(
            img_array,
            (self.lower_percentile, self.upper_percentile),
        )

        # Apply contrast stretching
        img_array = np.clip(
            (img_array - p_lower) / (p_upper - p_lower + self.eps), 0, 1
        )

        # Convert back to PIL image
        mode = "RGB" if img_array.ndim == 3 else "L"
        return Image.fromarray((img_array * 255).astype(np.uint8), mode=mode)


class CLAHEStrategy(XRayTransformStrategy):
    """Strategy for adaptive histogram equalization (CLAHE)."""

    def __init__(self, clip_limit: float = 2.0):
        self.clip_limit = clip_limit

    def apply(self, image: Image.Image) -> Image.Image:
        # NOTE: This implementation currently returns the original image as
        # cv2 is not available.
        logging.warning("cv2 not installed, skipping adaptive histogram equalization")
        return image


class EdgeEnhancementStrategy(XRayTransformStrategy):
    """Strategy for edge enhancement."""

    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def apply(self, image: Image.Image) -> Image.Image:
        from PIL import ImageFilter

        # Apply unsharp mask filter
        enhanced = image.filter(
            ImageFilter.UnsharpMask(
                radius=2,
                percent=int(self.strength * 150),
                threshold=3,
            ),
        )
        return enhanced
