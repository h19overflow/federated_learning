"""
ONNX FP32 inference wrapper using ONNX Runtime.

Provides FP32 precision inference with ONNX Runtime optimization.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort

from optimization_analysis.inference_wrappers.onnx_base import ONNXBaseInferenceWrapper

logger = logging.getLogger(__name__)


class ONNXInferenceWrapper(ONNXBaseInferenceWrapper):
    """ONNX inference wrapper using ONNX Runtime for FP32 inference."""

    def __init__(self, checkpoint_path: str = None, onnx_path: str = None):
        """
        Initialize ONNX inference wrapper.

        Args:
            checkpoint_path: Path to PyTorch checkpoint (for export if needed)
            onnx_path: Path to ONNX model (optional, auto-generated if not provided)
        """
        super().__init__(
            name="ONNX_FP32",
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path
        )

    def _load_model(self):
        """Load ONNX model with FP32 precision."""
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
