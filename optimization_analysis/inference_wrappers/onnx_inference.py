"""
ONNX-based inference wrappers for FP32 and FP16 precision.

Provides backward-compatible imports from refactored module structure.

Refactored from monolithic file to:
- onnx_base.py: Shared base functionality
- onnx_fp32.py: FP32 implementation
- onnx_fp16.py: FP16 implementation
"""

# Backward-compatible imports
from optimization_analysis.inference_wrappers.onnx_fp32 import ONNXInferenceWrapper
from optimization_analysis.inference_wrappers.onnx_fp16 import ONNXFP16InferenceWrapper

__all__ = [
    "ONNXInferenceWrapper",
    "ONNXFP16InferenceWrapper",
]
