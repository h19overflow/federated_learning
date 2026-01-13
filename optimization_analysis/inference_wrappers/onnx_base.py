"""
Shared base functionality for ONNX inference wrappers.

Contains common methods used by both FP32 and FP16 ONNX inference wrappers.
"""

import logging
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from optimization_analysis.inference_wrappers.base_inference import BaseInferenceWrapper

logger = logging.getLogger(__name__)


class ONNXBaseInferenceWrapper(BaseInferenceWrapper):
    """Base class for ONNX inference wrappers with shared functionality."""

    def __init__(self, name: str, checkpoint_path: str = None, onnx_path: str = None):
        """
        Initialize ONNX inference wrapper base.

        Args:
            name: Wrapper name (e.g., "ONNX_FP32", "ONNX_FP16")
            checkpoint_path: Path to PyTorch checkpoint (for export if needed)
            onnx_path: Path to ONNX model (optional, auto-generated if not provided)
        """
        if checkpoint_path is None:
            checkpoint_path = Path(__file__).parent.parent.parent / \
                "federated_pneumonia_detection" / \
                "src" / "control" / "model_inferance" / "pneumonia_model_07_0.928.ckpt"

        # Keep as Path for internal use
        self._checkpoint_path = Path(checkpoint_path)

        if onnx_path is None:
            onnx_path = self._checkpoint_path.with_suffix('.onnx')

        self.onnx_path = Path(onnx_path)

        self.session = None
        self.input_name = None
        self.output_name = None
        self._transform = None

        super().__init__(name=name, checkpoint_path=str(self._checkpoint_path))

    def _export_to_onnx(self):
        """
        Export PyTorch model to ONNX format.

        Shared method for both FP32 and FP16 wrappers.
        The precision is determined by how the ONNX model is loaded/inferenced,
        not by the export process itself.
        """
        try:
            from federated_pneumonia_detection.src.control.model_inferance.inference_engine import (
                InferenceEngine
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                engine = InferenceEngine(checkpoint_path=self._checkpoint_path)
                model = engine.model
                model.eval()

                dummy_input = torch.randn(1, 3, 224, 224)

                logger.info(f"Exporting PyTorch model to {self.onnx_path}")
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(self.onnx_path),
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )

            logger.info(f"ONNX export successful")

        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")
            raise

    def _setup_transforms(self):
        """
        Setup image preprocessing transforms.

        Matches PyTorch baseline exactly:
        - Resize to (224, 224)
        - CenterCrop to (224, 224)
        - Convert to tensor
        - Normalize with ImageNet stats
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

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for ONNX inference.

        Args:
            image: PIL Image

        Returns:
            Preprocessed numpy array (1, 3, 224, 224)
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self._transform(image)
        numpy_array = tensor.numpy()

        return numpy_array

    def extract_features(self, preprocessed_data: np.ndarray) -> np.ndarray:
        """
        Extract features using ONNX model.

        Args:
            preprocessed_data: Preprocessed numpy array (1, 3, 224, 224)

        Returns:
            Feature/logit tensor
        """
        if isinstance(preprocessed_data, torch.Tensor):
            preprocessed_data = preprocessed_data.numpy()

        if preprocessed_data.ndim == 3:
            preprocessed_data = preprocessed_data[np.newaxis, ...]

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: preprocessed_data.astype(np.float32)}
        )

        return outputs[0]

    def classify(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Classify features into prediction.

        Args:
            features: Feature/logit array from extract_features

        Returns:
            Tuple of (predicted_class, confidence)
        """
        if isinstance(features, torch.Tensor):
            features = features.numpy()

        logit = features.item()
        pneumonia_prob = 1.0 / (1.0 + np.exp(-logit))

        if pneumonia_prob >= 0.5:
            predicted_class = "PNEUMONIA"
            confidence = pneumonia_prob
        else:
            predicted_class = "NORMAL"
            confidence = 1.0 - pneumonia_prob

        return predicted_class, confidence
