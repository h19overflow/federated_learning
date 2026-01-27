"""GradCAM heatmap generation service.

Provides methods for generating GradCAM visualizations that highlight
the regions of chest X-rays that contributed most to the model's predictions.
"""

import logging
import time
from typing import List

from PIL import Image

from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    BatchHeatmapItem,
)
from federated_pneumonia_detection.src.control.model_inferance.gradcam import (
    GradCAM,
    heatmap_to_base64,
)
from federated_pneumonia_detection.src.control.model_inferance.inference_service import (  # noqa: E501
    InferenceService,
)

logger = logging.getLogger(__name__)


class GradCAMService:
    """Service for generating GradCAM heatmap visualizations.

    Manages GradCAM lifecycle and provides methods for single and batch
    heatmap generation with proper error handling and timing.
    """

    def __init__(self, inference_service: InferenceService):
        """Initialize GradCAMService with inference service dependency.

        Args:
            inference_service: InferenceService instance for model access
        """
        self.inference_service = inference_service

    async def generate_single_heatmap(
        self,
        image: Image.Image,
        filename: str,
        colormap: str = "jet",
        alpha: float = 0.4,
    ) -> BatchHeatmapItem:
        """Generate heatmap for a single image.

        Args:
            image: PIL Image object (RGB)
            filename: Original filename for tracking
            colormap: Matplotlib colormap name (jet, hot, viridis)
            alpha: Heatmap overlay transparency (0.1-0.9)

        Returns:
            BatchHeatmapItem with heatmap result or error details
        """
        start_time = time.time()

        # Handle None images gracefully
        if image is None:
            return BatchHeatmapItem(
                filename=filename,
                success=False,
                error="Invalid or unreadable image file",
                processing_time_ms=0,
            )

        try:
            engine = self.inference_service.engine
            if not engine or not engine.model:
                return BatchHeatmapItem(
                    filename=filename,
                    success=False,
                    error="Model not loaded",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            gradcam = GradCAM(engine.model)
            input_tensor = engine.preprocess(image)
            heatmap = gradcam(input_tensor)

            heatmap_base64 = heatmap_to_base64(
                heatmap=heatmap,
                original_image=image,
                colormap=colormap,
                alpha=alpha,
            )

            original_base64 = self.inference_service.processor.to_base64(image)
            gradcam.remove_hooks()

            processing_time_ms = (time.time() - start_time) * 1000

            return BatchHeatmapItem(
                filename=filename,
                success=True,
                heatmap_base64=heatmap_base64,
                original_image_base64=original_base64,
                processing_time_ms=processing_time_ms,
            )

        except Exception as e:
            logger.error(
                f"Heatmap generation failed for {filename}: {e}", exc_info=True
            )
            return BatchHeatmapItem(
                filename=filename,
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def generate_batch_heatmaps(
        self,
        images: List[tuple[Image.Image, str]],
        colormap: str = "jet",
        alpha: float = 0.4,
    ) -> List[BatchHeatmapItem]:
        """Generate heatmaps for multiple images.

        Args:
            images: List of (PIL Image, filename) tuples
            colormap: Matplotlib colormap name
            alpha: Heatmap overlay transparency

        Returns:
            List of BatchHeatmapItem results
        """
        results = []

        for image, filename in images:
            result = await self.generate_single_heatmap(
                image=image,
                filename=filename,
                colormap=colormap,
                alpha=alpha,
            )
            results.append(result)

        return results
