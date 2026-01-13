"""
PyTorch-based inference wrapper.
Implements the baseline PyTorch inference approach.
"""

import time
import torch
from pathlib import Path
from typing import Tuple
from PIL import Image

from optimization_analysis.inference_wrappers.base_inference import BaseInferenceWrapper


class PyTorchInferenceWrapper(BaseInferenceWrapper):
    """PyTorch inference wrapper using the existing InferenceEngine."""

    def __init__(self, checkpoint_path: str = None):
        """
        Initialize PyTorch inference wrapper.

        Args:
            checkpoint_path: Path to model checkpoint
        """
        if checkpoint_path is None:
            checkpoint_path = Path(__file__).parent.parent.parent / \
                "federated_pneumonia_detection" / \
                "src" / "control" / "model_inferance" / "pneumonia_model_07_0.928.ckpt"

        from federated_pneumonia_detection.src.control.model_inferance.inference_engine import (
            InferenceEngine
        )

        self.engine = InferenceEngine(checkpoint_path=Path(checkpoint_path))
        super().__init__(
            name="PyTorch_FP32",
            checkpoint_path=str(checkpoint_path)
        )

    def _load_model(self):
        """Model is loaded by InferenceEngine.__init__"""
        pass

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image.

        Args:
            image: PIL Image

        Returns:
            Preprocessed tensor
        """
        start = time.perf_counter()
        tensor = self.engine.preprocess(image)
        elapsed = (time.perf_counter() - start) * 1000
        return tensor

    def extract_features(self, preprocessed_data: torch.Tensor) -> torch.Tensor:
        """
        Extract features from preprocessed data.

        For ResNet, this is the forward pass up to the feature layer.
        We need to intercept the model's forward pass.

        Args:
            preprocessed_data: Preprocessed tensor

        Returns:
            Feature tensor
        """
        start = time.perf_counter()

        with torch.no_grad():
            features = self._extract_backbone_features(preprocessed_data)

        elapsed = (time.perf_counter() - start) * 1000
        return features

    def _extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from ResNet backbone."""
        model = self.engine.model
        if hasattr(model, 'model'):
            # Model structure: LitResNetEnhanced -> ResNetWithCustomHead -> features (ResNet50 backbone)
            resnet_backbone = model.model.features
            with torch.no_grad():
                features = resnet_backbone(x)
                return features
        else:
            # Fallback: run full forward and return as features
            with torch.no_grad():
                logits = model(x)
                return logits

    def classify(self, features: torch.Tensor) -> Tuple[str, float]:
        """
        Classify features.

        Args:
            features: Feature tensor

        Returns:
            Tuple of (predicted_class, confidence)
        """
        start = time.perf_counter()

        with torch.no_grad():
            if features.shape[1] == 1:
                logits = features
            else:
                model = self.engine.model
                if hasattr(model, 'model'):
                    if hasattr(model.model, 'classifier'):
                        logits = model.model.classifier(features)
                    else:
                        logits = model(features)
                else:
                    logits = model(features)

            pneumonia_prob = torch.sigmoid(logits).item()

            if pneumonia_prob >= 0.5:
                predicted_class = "PNEUMONIA"
                confidence = pneumonia_prob
            else:
                predicted_class = "NORMAL"
                confidence = 1.0 - pneumonia_prob

        elapsed = (time.perf_counter() - start) * 1000
        return predicted_class, confidence

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """
        Full prediction pipeline using the existing engine.

        Args:
            image: PIL Image

        Returns:
            Tuple of (predicted_class, confidence)
        """
        predicted_class, confidence, _, _ = self.engine.predict(image)
        return predicted_class, confidence

    def get_model_info(self) -> dict:
        """Get model information."""
        return self.engine.get_info()
