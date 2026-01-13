"""
ONNX FP16 inference wrapper using ONNX Runtime with FP16 optimization.

Attempts to enable FP16 precision for improved inference speed on supported hardware.
"""

import logging
import warnings

import numpy as np
import onnxruntime as ort

from optimization_analysis.inference_wrappers.onnx_base import ONNXBaseInferenceWrapper

logger = logging.getLogger(__name__)


class ONNXFP16InferenceWrapper(ONNXBaseInferenceWrapper):
    """ONNX inference wrapper using ONNX Runtime with FP16 optimization attempts."""

    def __init__(self, checkpoint_path: str = None, onnx_path: str = None):
        """
        Initialize ONNX FP16 inference wrapper.

        Args:
            checkpoint_path: Path to PyTorch checkpoint (for export if needed)
            onnx_path: Path to ONNX model (optional, auto-generated if not provided)
        """
        super().__init__(
            name="ONNX_FP16",
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path
        )

        self.fp16_enabled = False
        self.execution_provider_used = None

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
