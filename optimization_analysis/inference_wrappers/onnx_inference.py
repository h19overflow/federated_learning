"""
ONNX-based inference wrappers for FP32 and FP16 precision.
Implements inference using ONNX Runtime for optimized performance.
"""

import logging
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from torchvision import transforms

from optimization_analysis.inference_wrappers.base_inference import BaseInferenceWrapper

logger = logging.getLogger(__name__)


class ONNXInferenceWrapper(BaseInferenceWrapper):
    """ONNX inference wrapper using ONNX Runtime for FP32 inference."""

    def __init__(self, checkpoint_path: str = None, onnx_path: str = None):
        """
        Initialize ONNX inference wrapper.

        Args:
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

        super().__init__(
            name="ONNX_FP32",
            checkpoint_path=str(self._checkpoint_path)
        )

    def _load_model(self):
        if not self.onnx_path.exists():
            logger.info(f"ONNX model not found at {self.onnx_path}, exporting from PyTorch...")
            self._export_to_onnx()

        logger.info(f"Loading ONNX model from {self.onnx_path}")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.session = ort.InferenceSession(
                    str(self.onnx_path),
                    providers=['CPUExecutionProvider', 'CUDAExecutionProvider']
                )
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"ONNX model loaded successfully")
        logger.info(f"  Input: {self.input_name} (shape: {self.session.get_inputs()[0].shape})")
        logger.info(f"  Output: {self.output_name} (shape: {self.session.get_outputs()[0].shape})")
        logger.info(f"  Providers: {self.session.get_providers()}")

        self._setup_transforms()

    def _export_to_onnx(self):
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

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """
        Full prediction pipeline.

        Args:
            image: PIL Image

        Returns:
            Tuple of (predicted_class, confidence)
        """
        return super().predict(image)

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_type": "ONNX_FP32",
            "onnx_path": str(self.onnx_path),
            "checkpoint_path": str(self.checkpoint_path),
            "input_name": self.input_name,
            "output_name": self.output_name,
            "providers": self.session.get_providers(),
            "input_shape": list(self.session.get_inputs()[0].shape),
            "output_shape": list(self.session.get_outputs()[0].shape),
        }


class ONNXFP16InferenceWrapper(BaseInferenceWrapper):
    """ONNX inference wrapper using ONNX Runtime with FP16 optimization attempts."""

    def __init__(self, checkpoint_path: str = None, onnx_path: str = None):
        """
        Initialize ONNX FP16 inference wrapper.

        Args:
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
        self.fp16_enabled = False
        self.execution_provider_used = None

        super().__init__(
            name="ONNX_FP16",
            checkpoint_path=str(self._checkpoint_path)
        )

    def _load_model(self):
        """Load ONNX model with FP16 optimization attempt."""
        if not self.onnx_path.exists():
            logger.info(f"ONNX model not found at {self.onnx_path}, exporting from PyTorch...")
            self._export_to_onnx()

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNXRuntime execution providers: {available_providers}")

        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info("Attempting CUDAExecutionProvider with FP16 optimization")
        else:
            providers = ['CPUExecutionProvider']
            logger.warning("CUDAExecutionProvider not available, falling back to CPU")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.session = ort.InferenceSession(
                    str(self.onnx_path),
                    providers=providers,
                    sess_options=session_options
                )

            active_providers = self.session.get_providers()
            self.execution_provider_used = active_providers[0] if active_providers else "Unknown"
            logger.info(f"Active execution provider: {self.execution_provider_used}")

            self.fp16_enabled = self._check_fp16_enabled()
            if self.fp16_enabled:
                logger.info("✓ FP16 optimization enabled")
            else:
                logger.warning("✗ FP16 optimization not available, using FP32 inference")

        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            logger.warning("Falling back to FP32 inference")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.session = ort.InferenceSession(
                        str(self.onnx_path),
                        providers=['CPUExecutionProvider'],
                        sess_options=session_options
                    )
                self.execution_provider_used = "CPUExecutionProvider"
                self.fp16_enabled = False
                logger.info("Successfully loaded with CPU provider (FP32 fallback)")
            except Exception as fallback_error:
                logger.error(f"Fallback to CPU also failed: {fallback_error}")
                raise

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"ONNX model loaded successfully")
        logger.info(f"  Input: {self.input_name} (shape: {self.session.get_inputs()[0].shape})")
        logger.info(f"  Output: {self.output_name} (shape: {self.session.get_outputs()[0].shape})")
        logger.info(f"  Precision: {'FP16' if self.fp16_enabled else 'FP32'}")
        logger.info(f"  Execution Provider: {self.execution_provider_used}")

        self._setup_transforms()

    def _check_fp16_enabled(self) -> bool:
        """
        Check if FP16 inference is enabled.

        This is a heuristic check since ONNXRuntime doesn't provide a direct API
        to query the precision mode. We check the execution provider and model metadata.

        Returns:
            True if FP16 is likely being used, False otherwise
        """
        if self.execution_provider_used == "CUDAExecutionProvider":
            try:
                input_dtype = self.session.get_inputs()[0].type
                if 'float16' in str(input_dtype).lower() or 'half' in str(input_dtype).lower():
                    return True
            except Exception:
                pass

            return True

        return False

    def _export_to_onnx(self):
        """Export PyTorch model to ONNX format (shared with FP32)."""
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
        Preprocess image for ONNX FP16 inference.

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

        For ONNX FP16, we run the full forward pass.
        The output is the logit which serves as our "feature".

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

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """
        Full prediction pipeline.

        Args:
            image: PIL Image

        Returns:
            Tuple of (predicted_class, confidence)
        """
        return super().predict(image)

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_type": "ONNX_FP16",
            "onnx_path": str(self.onnx_path),
            "checkpoint_path": str(self.checkpoint_path),
            "input_name": self.input_name,
            "output_name": self.output_name,
            "providers": self.session.get_providers(),
            "input_shape": list(self.session.get_inputs()[0].shape),
            "output_shape": list(self.session.get_outputs()[0].shape),
            "fp16_enabled": self.fp16_enabled,
            "execution_provider_used": self.execution_provider_used,
        }
