"""
Optimization Analysis Module

Benchmark infrastructure for measuring preprocessing, feature extraction,
and classification performance on X-ray images.
"""

__version__ = "0.1.0"
__author__ = "FYP2 Team"

from .benchmark.stage_timer import StageTimer
from .benchmark.results_collector import ResultsCollector, BenchmarkResult
from .benchmark.benchmark_suite import BenchmarkSuite
from .inference_wrappers.base_inference import BaseInferenceWrapper
from .inference_wrappers.pytorch_inference import PyTorchInferenceWrapper
from .utils.dataset_loader import DatasetLoader
from .metrics.performance_metrics import calculate_classification_metrics, calculate_stage_statistics

__all__ = [
    "StageTimer",
    "ResultsCollector",
    "BenchmarkResult",
    "BenchmarkSuite",
    "BaseInferenceWrapper",
    "PyTorchInferenceWrapper",
    "DatasetLoader",
    "calculate_classification_metrics",
    "calculate_stage_statistics",
]
