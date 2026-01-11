"""Core inference engine for pneumonia detection.

Contains the core logic for loading and running the model.
This belongs in the control layer as it's business logic.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Default checkpoint path (relative to this module)
DEFAULT_CHECKPOINT_PATH = Path(__file__).parent / "pneumonia_model_01_0.988-v2.ckpt"


class InferenceEngine:
    """Core inference engine for pneumonia classification.

    Handles model loading, preprocessing, and prediction logic.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """Initialize the inference engine.

        Args:
            checkpoint_path: Path to the model checkpoint.
            device: Device to run inference on (auto-detected if None).
        """
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_version = self.checkpoint_path.stem
        self._transform = None

        self._load_model()
        self._setup_transforms()

    def _load_model(self) -> None:
        """Load the model from checkpoint."""
        from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet_enhanced import (
            LitResNetEnhanced,
        )

        logger.info(f"Loading model from {self.checkpoint_path}")
        start_time = time.time()

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.model = LitResNetEnhanced.load_from_checkpoint(
            str(self.checkpoint_path),
            map_location=self.device,
            config=None,
            class_weights_tensor=None,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.freeze()

        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s on {self.device}")

    def _setup_transforms(self) -> None:
        """Setup image preprocessing transforms.

        Matches validation transforms from image_transforms.py:246-277.
        """
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a PIL image for inference.

        Args:
            image: PIL Image in any mode.

        Returns:
            Preprocessed tensor ready for model input.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self._transform(image)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Tuple[str, float, float, float]:
        """Run inference on a single image.

        Args:
            image: PIL Image to classify.

        Returns:
            Tuple of (predicted_class, confidence, pneumonia_prob, normal_prob).
        """
        start_time = time.time()
        input_tensor = self.preprocess(image)
        logits = self.model(input_tensor)

        pneumonia_prob = torch.sigmoid(logits).item()
        normal_prob = 1.0 - pneumonia_prob

        if pneumonia_prob >= 0.5:
            predicted_class = "PNEUMONIA"
            confidence = pneumonia_prob
        else:
            predicted_class = "NORMAL"
            confidence = normal_prob

        inference_time = (time.time() - start_time) * 1000
        logger.debug(f"Inference: {predicted_class} ({confidence:.4f}) in {inference_time:.2f}ms")

        return predicted_class, confidence, pneumonia_prob, normal_prob

    @property
    def is_gpu(self) -> bool:
        """Check if model is using GPU."""
        return self.device == "cuda"

    def get_info(self) -> dict:
        """Get engine information."""
        return {
            "model_version": self.model_version,
            "device": self.device,
            "gpu_available": torch.cuda.is_available(),
            "checkpoint_path": str(self.checkpoint_path),
        }
