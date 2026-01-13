"""
Benchmark suite for optimization analysis.
Orchestrates benchmark runs across multiple inference approaches.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

from optimization_analysis.benchmark.stage_timer import StageTimer
from optimization_analysis.benchmark.results_collector import (
    ResultsCollector,
    BenchmarkResult
)
from optimization_analysis.inference_wrappers.base_inference import BaseInferenceWrapper
from optimization_analysis.inference_wrappers.pytorch_inference import PyTorchInferenceWrapper
from optimization_analysis.utils.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """Main benchmark orchestrator for optimization analysis."""
    
    def __init__(
        self,
        image_dir: str,
        checkpoint_path: str = None,
        num_warmup: int = 10,
        num_samples: int = 1000
    ):
        """
        Initialize benchmark suite.
        
        Args:
            image_dir: Directory containing test images
            checkpoint_path: Path to model checkpoint
            num_warmup: Number of warmup runs before timing
            num_samples: Number of samples to benchmark
        """
        self.image_dir = Path(image_dir)
        self.checkpoint_path = checkpoint_path
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        
        self.dataset_loader = DatasetLoader(image_dir, max_images=num_samples)
        self.results_collector = ResultsCollector()
        
        logger.info(f"BenchmarkSuite initialized with {num_samples} samples")
    
    def load_images(self) -> List[Image.Image]:
        """Load test images."""
        logger.info("Loading test images...")
        images = self.dataset_loader.load_images(count=self.num_samples)
        logger.info(f"Loaded {len(images)} images")
        return images
    
    def run_benchmark(
        self,
        wrapper: BaseInferenceWrapper,
        images: List[Image.Image],
        labels: Optional[List[int]] = None
    ) -> BenchmarkResult:
        """
        Run benchmark for a single inference wrapper.
        
        Args:
            wrapper: Inference wrapper to benchmark
            images: List of PIL Images
            labels: Optional true labels for accuracy calculation
            
        Returns:
            BenchmarkResult object
        """
        logger.info(f"Running benchmark for {wrapper.get_name()}...")
        timer = StageTimer()
        
        # Warmup runs
        logger.info(f"Running {self.num_warmup} warmup iterations...")
        for i, img in enumerate(images[:self.num_warmup]):
            try:
                wrapper.predict(img)
            except Exception as e:
                logger.warning(f"Warmup failed on image {i}: {e}")
        
        # Actual benchmark runs
        logger.info(f"Running benchmark on {len(images)} images...")
        predictions = []
        
        for i, img in enumerate(images):
            try:
                with timer.time_stage('preprocessing'):
                    preprocessed = wrapper.preprocess(img)
                
                with timer.time_stage('feature_extraction'):
                    features = wrapper.extract_features(preprocessed)
                
                with timer.time_stage('classification'):
                    pred_class, confidence = wrapper.classify(features)
                
                with timer.time_stage('total'):
                    pass
                
                predictions.append((pred_class, confidence))
                
            except Exception as e:
                logger.error(f"Failed on image {i}: {e}")
                continue
        
        # Calculate stage statistics
        stage_stats = {
            'preprocessing': timer.get_statistics('preprocessing'),
            'feature_extraction': timer.get_statistics('feature_extraction'),
            'classification': timer.get_statistics('classification'),
        }
        
        # Calculate total time statistics from individual stage means
        # Total time is sum of individual stages
        if all(stats.get('count', 0) > 0 for stats in stage_stats.values()):
            stage_stats['total'] = {
                'mean': stage_stats['preprocessing'].get('mean', 0) +
                        stage_stats['feature_extraction'].get('mean', 0) +
                        stage_stats['classification'].get('mean', 0),
                'median': stage_stats['preprocessing'].get('median', 0) +
                         stage_stats['feature_extraction'].get('median', 0) +
                         stage_stats['classification'].get('median', 0),
                'p50': stage_stats['preprocessing'].get('p50', 0) +
                       stage_stats['feature_extraction'].get('p50', 0) +
                       stage_stats['classification'].get('p50', 0),
                'p95': stage_stats['preprocessing'].get('p95', 0) +
                       stage_stats['feature_extraction'].get('p95', 0) +
                       stage_stats['classification'].get('p95', 0),
                'p99': stage_stats['preprocessing'].get('p99', 0) +
                       stage_stats['feature_extraction'].get('p99', 0) +
                       stage_stats['classification'].get('p99', 0),
                'min': stage_stats['preprocessing'].get('min', 0) +
                      stage_stats['feature_extraction'].get('min', 0) +
                      stage_stats['classification'].get('min', 0),
                'max': stage_stats['preprocessing'].get('max', 0) +
                      stage_stats['feature_extraction'].get('max', 0) +
                      stage_stats['classification'].get('max', 0),
                'count': stage_stats['preprocessing'].get('count', 0),
                'stddev': (stage_stats['preprocessing'].get('stddev', 0) ** 2 +
                         stage_stats['feature_extraction'].get('stddev', 0) ** 2 +
                         stage_stats['classification'].get('stddev', 0) ** 2) ** 0.5
            }
        else:
            stage_stats['total'] = {
                'mean': 0,
                'median': 0,
                'p50': 0,
                'p95': 0,
                'p99': 0,
                'min': 0,
                'max': 0,
                'stddev': 0,
                'count': 0
            }
        
        # Calculate accuracy if labels provided
        accuracy_metrics = {}
        if labels is not None and len(labels) == len(predictions):
            from optimization_analysis.metrics.performance_metrics import (
                calculate_classification_metrics
            )
            pred_labels = [1 if pred == "PNEUMONIA" else 0 
                          for pred, _ in predictions]
            accuracy_metrics = calculate_classification_metrics(
                true_labels=labels,
                pred_labels=pred_labels
            )
        
        # Create result object
        result = BenchmarkResult(
            approach_name=wrapper.get_name(),
            num_samples=len(predictions),
            stage_stats=stage_stats,
            accuracy_metrics=accuracy_metrics,
            total_time_avg=stage_stats['total']['mean'],
            timestamp=self._get_timestamp()
        )
        
        logger.info(f"Benchmark completed for {wrapper.get_name()}")
        logger.info(f"  Total avg: {result.total_time_avg:.2f}ms")
        logger.info(f"  Preprocess: {stage_stats['preprocessing']['mean']:.2f}ms")
        logger.info(f"  Feature: {stage_stats['feature_extraction']['mean']:.2f}ms")
        logger.info(f"  Classify: {stage_stats['classification']['mean']:.2f}ms")
        
        return result
    
    def run_comparison(
        self,
        wrappers: List[BaseInferenceWrapper],
        images: List[Image.Image],
        labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Run benchmarks for multiple wrappers and compare.
        
        Args:
            wrappers: List of inference wrappers
            images: List of PIL Images
            labels: Optional true labels
            
        Returns:
            Comparison results
        """
        logger.info(f"Running comparison across {len(wrappers)} approaches...")
        
        for wrapper in wrappers:
            result = self.run_benchmark(wrapper, images, labels)
            self.results_collector.add_result(result)
        
        comparison = self.results_collector.compare_approaches()
        
        logger.info("\n=== COMPARISON SUMMARY ===")
        for name, stats in comparison.items():
            logger.info(f"\n{name}:")
            logger.info(f"  Total time: {stats['total_time_avg']:.2f}ms")
            if stats['accuracy']:
                logger.info(f"  Accuracy: {stats['accuracy'].get('accuracy', 'N/A'):.4f}")
        
        return comparison
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_results(self, output_dir: str = "optimization_results"):
        """Save benchmark results to file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = self._get_timestamp().replace(':', '-')[:19]
        filepath = output_path / f"benchmark_results_{timestamp}.json"
        
        self.results_collector.save_to_file(str(filepath))
        logger.info(f"Results saved to {filepath}")


def create_default_suite(image_dir: str, num_samples: int = 1000) -> BenchmarkSuite:
    """Create a benchmark suite with default wrappers."""
    suite = BenchmarkSuite(
        image_dir=image_dir,
        num_samples=num_samples
    )
    return suite


def run_baseline_benchmark(
    image_dir: str,
    checkpoint_path: str = None,
    num_samples: int = 1000,
    output_dir: str = "optimization_results"
) -> Dict[str, Any]:
    """
    Run baseline benchmark with PyTorch wrapper.
    
    Args:
        image_dir: Directory containing test images
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples
        output_dir: Output directory for results
        
    Returns:
        Benchmark results
    """
    suite = create_default_suite(image_dir, num_samples=num_samples)
    images = suite.load_images()
    
    # Create PyTorch wrapper
    wrapper = PyTorchInferenceWrapper(checkpoint_path=checkpoint_path)
    
    # Run benchmark
    result = suite.run_benchmark(wrapper, images)
    suite.results_collector.add_result(result)
    
    # Save results
    suite.save_results(output_dir)
    
    return suite.results_collector.get_summary()
