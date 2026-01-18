# Optimization Analysis Module

## Overview

Modular benchmarking infrastructure for measuring and optimizing pneumonia detection inference performance. Provides stage-by-stage profiling (preprocessing, feature extraction, classification) with comprehensive statistical metrics.

**Purpose:** Measure and optimize inference latency without affecting model accuracy, following the optimization strategy outlined in `optimization_strategy.md`.

---

## Architecture

```
optimization_analysis/
├── benchmark/                      # Benchmark orchestration
│   ├── benchmark_suite.py          # Main benchmark orchestrator
│   ├── stage_timer.py            # Stage timing utilities
│   └── results_collector.py      # Results aggregation
├── inference_wrappers/             # Pluggable inference implementations
│   ├── base_inference.py         # Abstract base class
│   └── pytorch_inference.py      # PyTorch FP32 baseline
├── metrics/                       # Metrics calculation
│   └── performance_metrics.py     # Statistical calculations
└── utils/                        # Utilities
    └── dataset_loader.py         # Image loading
```

---

## Quick Start

### 1. Run Baseline Benchmark

```python
from optimization_analysis import run_baseline_benchmark

results = run_baseline_benchmark(
    image_dir="path/to/test/images",
    checkpoint_path="federated_pneumonia_detection/src/control/model_inferance/pneumonia_model_07_0.928.ckpt",
    num_samples=1000,
    output_dir="optimization_results"
)

print(f"Total time: {results['total_time_avg']:.2f}ms")
print(f"Accuracy: {results['accuracy_metrics']['accuracy']:.4f}")
```

### 2. Custom Benchmark Suite

```python
from optimization_analysis import BenchmarkSuite, PyTorchInferenceWrapper

# Initialize suite
suite = BenchmarkSuite(
    image_dir="path/to/images",
    num_samples=1000,
    num_warmup=10  # Warmup iterations before timing
)

# Load images
images = suite.load_images()

# Create inference wrapper
wrapper = PyTorchInferenceWrapper(
    checkpoint_path="path/to/model.ckpt"
)

# Run benchmark (with optional labels for accuracy)
result = suite.run_benchmark(
    wrapper=wrapper,
    images=images,
    labels=[0, 1, 0, 1, ...]  # Optional true labels
)

# Save results
suite.save_results("my_benchmark_results")
```

### 3. Compare Multiple Approaches

```python
from optimization_analysis import BenchmarkSuite, PyTorchInferenceWrapper

suite = BenchmarkSuite(image_dir="path/to/images", num_samples=1000)
images = suite.load_images()

# Multiple wrappers (PyTorch, ONNX, TensorRT, etc.)
wrappers = [
    PyTorchInferenceWrapper(checkpoint_path="model1.ckpt"),
    PyTorchInferenceWrapper(checkpoint_path="model2.ckpt"),
    # Add other wrappers as implemented
]

# Run comparison
comparison = suite.run_comparison(
    wrappers=wrappers,
    images=images,
    labels=[...]  # Optional ground truth
)

# Print comparison
for approach, stats in comparison.items():
    print(f"\n{approach}:")
    print(f"  Total time: {stats['total_time_avg']:.2f}ms")
    print(f"  Accuracy: {stats['accuracy']:.4f}")
```

---

## Components

### BenchmarkSuite

Main orchestrator for running benchmarks.

**Methods:**
- `load_images()` - Load test images from directory
- `run_benchmark(wrapper, images, labels)` - Run single benchmark
- `run_comparison(wrappers, images, labels)` - Compare multiple approaches
- `save_results(output_dir)` - Save results to JSON

**Example:**
```python
from optimization_analysis import BenchmarkSuite

suite = BenchmarkSuite(image_dir="path/to/images", num_samples=1000)
images = suite.load_images()

# Warmup runs (configurable, default 10) ensure model is fully initialized
# Then timed runs on all images
result = suite.run_benchmark(wrapper, images)
```

### StageTimer

Context manager for timing individual pipeline stages.

**Stages Tracked:**
- `preprocessing` - Image preprocessing time
- `feature_extraction` - Feature extraction time
- `classification` - Classification time
- `total` - Total pipeline time

**Statistics Calculated:**
- `mean`, `median`, `p50`, `p95`, `p99`, `min`, `max`, `stddev`

**Example:**
```python
from optimization_analysis import StageTimer

timer = StageTimer()

with timer.time_stage('preprocessing'):
    # Preprocess image
    pass

stats = timer.get_statistics('preprocessing')
print(f"Mean: {stats['mean']:.2f}ms")
print(f"p95: {stats['p95']:.2f}ms")
```

### ResultsCollector

Collects and aggregates benchmark results.

**Methods:**
- `add_result(result)` - Add a benchmark result
- `get_summary()` - Get summary of all results
- `compare_approaches()` - Compare different approaches
- `save_to_file(filepath)` - Save to JSON

**Example:**
```python
from optimization_analysis import ResultsCollector

collector = ResultsCollector()
collector.add_result(result)

summary = collector.get_summary()
comparison = collector.compare_approaches()

collector.save_to_file("results.json")
```

### PyTorchInferenceWrapper

PyTorch FP32 baseline inference wrapper.

**Features:**
- Wraps existing InferenceEngine
- Stage-wise inference (preprocess, extract_features, classify)
- Compatible with existing preprocessing pipeline
- Provides baseline for optimization comparison

**Example:**
```python
from optimization_analysis import PyTorchInferenceWrapper

wrapper = PyTorchInferenceWrapper(checkpoint_path="model.ckpt")

# Individual stages
tensor = wrapper.preprocess(image)
features = wrapper.extract_features(tensor)
class_name, confidence = wrapper.classify(features)

# Or full pipeline
class_name, confidence = wrapper.predict(image)
```

### DatasetLoader

Utility for loading test images.

**Methods:**
- `load_images(count)` - Load images into memory
- `get_sample_paths(count)` - Get paths without loading

**Example:**
```python
from optimization_analysis import DatasetLoader

loader = DatasetLoader(image_dir="path/to/images", max_images=1000)
images = loader.load_images(count=100)
```

### Performance Metrics

Calculate classification metrics using sklearn.

**Functions:**
- `calculate_classification_metrics(true_labels, pred_labels, pred_probs)` - Full metrics
- `calculate_stage_statistics(timings)` - Timing statistics

**Metrics Calculated:**
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix (TN, FP, FN, TP)
- Specificity, Sensitivity
- AUROC (if probabilities provided)

**Example:**
```python
from optimization_analysis import calculate_classification_metrics

metrics = calculate_classification_metrics(
    true_labels=[0, 1, 1, 0, ...],
    pred_labels=[0, 1, 1, 1, ...],
    pred_probs=[0.1, 0.9, 0.8, 0.6, ...]
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
```

---

## Benchmark Workflow

```
1. INITIALIZATION
   └─ Create BenchmarkSuite with image_dir, num_samples, num_warmup

2. IMAGE LOADING
   └─ load_images() → List[PIL.Image]

3. WARMUP PHASE (10 iterations default)
   └─ Runs warmup without timing
       └─ Ensures model is fully initialized

4. BENCHMARK PHASE
   ├─ For each image:
   │   ├─ time_stage('preprocessing')
   │   ├─ time_stage('feature_extraction')
   │   ├─ time_stage('classification')
   │   └─ Collect predictions
   └─ Calculate statistics (mean, p50, p95, p99, etc.)

5. ACCURACY CALCULATION (if labels provided)
   └─ Uses sklearn for accuracy, precision, recall, F1, AUROC

6. RESULTS COLLECTION
   ├─ Add to ResultsCollector
   └─ Save to JSON file

7. COMPARISON (optional)
   └─ Run multiple wrappers and compare results
```

---

## Output Format

### BenchmarkResult

```python
@dataclass
class BenchmarkResult:
    approach_name: str              # e.g., "PyTorch_FP32"
    num_samples: int               # Number of samples processed
    stage_stats: Dict              # Stage statistics
    accuracy_metrics: Dict         # Accuracy metrics (if labels provided)
    total_time_avg: float          # Average total time (ms)
    timestamp: str                # ISO timestamp
```

### Stage Statistics

```python
{
    'preprocessing': {
        'mean': 1.50,
        'median': 1.45,
        'p50': 1.45,
        'p95': 2.10,
        'p99': 2.50,
        'min': 1.20,
        'max': 3.00,
        'stddev': 0.30,
        'count': 1000
    },
    'feature_extraction': { ... },
    'classification': { ... },
    'total': { ... }
}
```

### Accuracy Metrics

```python
{
    'accuracy': 0.9280,
    'precision': 0.9100,
    'recall': 0.9280,
    'f1': 0.9190,
    'auroc': 0.9750,
    'true_negatives': 850,
    'false_positives': 50,
    'false_negatives': 72,
    'true_positives': 928,
    'specificity': 0.9444,
    'sensitivity': 0.9280
}
```

---

## Adding New Inference Approaches

### 1. Create New Wrapper

```python
from optimization_analysis import BaseInferenceWrapper

class ONNXInferenceWrapper(BaseInferenceWrapper):
    """ONNX Runtime inference wrapper."""
    
    def __init__(self, checkpoint_path: str):
        super().__init__(name="ONNX_FP32", checkpoint_path=checkpoint_path)
        # Load ONNX model
        import onnxruntime as ort
        self.session = ort.InferenceSession(checkpoint_path)
    
    def _load_model(self):
        pass  # Model loaded in __init__
    
    def preprocess(self, image):
        # ONNX-specific preprocessing
        pass
    
    def extract_features(self, preprocessed_data):
        # Extract features
        pass
    
    def classify(self, features):
        # Classify
        pass
```

### 2. Use in Benchmark

```python
from optimization_analysis import BenchmarkSuite, ONNXInferenceWrapper

suite = BenchmarkSuite(image_dir="path/to/images", num_samples=1000)
images = suite.load_images()

onnx_wrapper = ONNXInferenceWrapper("model.onnx")
result = suite.run_benchmark(onnx_wrapper, images)
```

---

## Configuration

### Warmup Iterations

```python
suite = BenchmarkSuite(
    image_dir="path/to/images",
    num_warmup=10  # 10 warmup iterations (default)
)
```

### Sample Count

```python
suite = BenchmarkSuite(
    image_dir="path/to/images",
    num_samples=1000  # 1000 samples (default)
)
```

### Checkpoint Path

```python
from optimization_analysis import PyTorchInferenceWrapper

wrapper = PyTorchInferenceWrapper(
    checkpoint_path="path/to/model.ckpt"
)
```

---

## Example: Complete Workflow

```python
import logging
from optimization_analysis import BenchmarkSuite, PyTorchInferenceWrapper

# Setup logging
logging.basicConfig(level=logging.INFO)

# Create benchmark suite
suite = BenchmarkSuite(
    image_dir="path/to/test/images",
    num_samples=1000,
    num_warmup=10
)

# Load images
images = suite.load_images()
print(f"Loaded {len(images)} images")

# Create PyTorch wrapper (FP32 baseline)
wrapper = PyTorchInferenceWrapper(
    checkpoint_path="federated_pneumonia_detection/src/control/model_inferance/pneumonia_model_07_0.928.ckpt"
)

# Run benchmark
print("\nRunning baseline benchmark...")
result = suite.run_benchmark(wrapper, images)

# Print results
print(f"\n=== BENCHMARK RESULTS ===")
print(f"Approach: {result.approach_name}")
print(f"Samples: {result.num_samples}")
print(f"\nStage Breakdown (ms):")
for stage, stats in result.stage_stats.items():
    print(f"  {stage}: {stats['mean']:.2f} (p95: {stats['p95']:.2f})")

print(f"\nTotal Time: {result.total_time_avg:.2f}ms")

if result.accuracy_metrics:
    print(f"\nAccuracy Metrics:")
    print(f"  Accuracy: {result.accuracy_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {result.accuracy_metrics['f1']:.4f}")
    print(f"  Recall: {result.accuracy_metrics['recall']:.4f}")
    print(f"  AUROC: {result.accuracy_metrics.get('auroc', 'N/A')}")

# Save results
suite.save_results("baseline_benchmark_results")
print("\nResults saved!")
```

---

## Next Steps

### Phase 1: Baseline Measurement ✅
- [x] Create modular benchmarking infrastructure
- [x] Implement PyTorch FP32 baseline wrapper
- [ ] Run baseline benchmark on 1000 images
- [ ] Document baseline performance

### Phase 2: Optimization Approaches
- [ ] Implement ONNX FP32 wrapper
- [ ] Implement ONNX FP16 wrapper
- [ ] Implement TensorRT wrapper (if GPU available)
- [ ] Implement quantization wrappers (INT8)

### Phase 3: Comparative Analysis
- [ ] Run all optimization approaches
- [ ] Compare speedups vs baseline
- [ ] Validate accuracy maintained
- [ ] Generate comparison report

### Phase 4: Production Integration
- [ ] Integrate best-performing approach into InferenceEngine
- [ ] Update API endpoints
- [ ] Monitor in production

---

## References

- **Optimization Strategy**: `optimization_strategy.md`
- **Current Inference**: `federated_pneumonia_detection/src/control/model_inferance/inference_engine.py`
- **Model Architecture**: `federated_pneumonia_detection/src/control/dl_model/utils/model/lit_resnet_enhanced.py`
- **Preprocessing**: `federated_pneumonia_detection/src/control/dl_model/utils/model/xray_data_module.py`

---

## Version

**Current Version:** 0.1.0
**Author:** FYP2 Team
**Date:** 2026-01-13

---

## License

MIT License - See project LICENSE file for details.
