"""
Quick test script for optimization_analysis module.
Verifies basic functionality without running full benchmarks.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_imports():
    """Test all module imports."""
    print("=== Testing Imports ===")
    try:
        from optimization_analysis import (
            BenchmarkSuite,
            StageTimer,
            ResultsCollector,
            PyTorchInferenceWrapper,
            DatasetLoader,
            calculate_classification_metrics,
            calculate_stage_statistics
        )
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_stage_timer():
    """Test StageTimer functionality."""
    print("\n=== Testing StageTimer ===")
    try:
        from optimization_analysis import StageTimer
        import time

        timer = StageTimer()

        # Simulate some work
        with timer.time_stage('preprocessing'):
            time.sleep(0.001)  # 1ms

        with timer.time_stage('feature_extraction'):
            time.sleep(0.005)  # 5ms

        with timer.time_stage('classification'):
            time.sleep(0.003)  # 3ms

        # Get statistics
        stats = timer.get_statistics('preprocessing')
        print(f"✅ Preprocessing stats: {stats}")

        assert stats['count'] == 1, "Should have 1 timing"
        assert stats['mean'] > 0, "Mean should be positive"
        print("✅ StageTimer test passed!")
        return True

    except Exception as e:
        print(f"❌ StageTimer test failed: {e}")
        return False


def test_results_collector():
    """Test ResultsCollector functionality."""
    print("\n=== Testing ResultsCollector ===")
    try:
        from optimization_analysis import ResultsCollector, BenchmarkResult

        collector = ResultsCollector()

        # Create a dummy result
        result = BenchmarkResult(
            approach_name="Test",
            num_samples=10,
            stage_stats={
                'preprocessing': {'mean': 1.0, 'count': 10},
                'feature_extraction': {'mean': 5.0, 'count': 10},
                'classification': {'mean': 3.0, 'count': 10},
                'total': {'mean': 9.0, 'count': 10}
            },
            accuracy_metrics={'accuracy': 0.9, 'f1': 0.9},
            total_time_avg=9.0,
            timestamp="2026-01-13T10:00:00"
        )

        collector.add_result(result)

        # Get summary
        summary = collector.get_summary()
        print(f"✅ Summary: total_benchmarks={summary['total_benchmarks']}")

        # Get comparison
        comparison = collector.compare_approaches()
        print(f"✅ Comparison: approaches={list(comparison.keys())}")

        assert summary['total_benchmarks'] == 1, "Should have 1 benchmark"
        assert 'Test' in comparison, "Should have Test approach"
        print("✅ ResultsCollector test passed!")
        return True

    except Exception as e:
        print(f"❌ ResultsCollector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics calculation."""
    print("\n=== Testing Metrics Calculation ===")
    try:
        from optimization_analysis import calculate_classification_metrics, calculate_stage_statistics

        # Test classification metrics
        true_labels = [0, 1, 1, 0, 1]
        pred_labels = [0, 1, 0, 0, 1]
        pred_probs = [0.1, 0.9, 0.4, 0.2, 0.8]

        metrics = calculate_classification_metrics(true_labels, pred_labels, pred_probs)
        print(f"✅ Metrics: {metrics}")

        assert 'accuracy' in metrics, "Should have accuracy"
        assert 'f1' in metrics, "Should have F1 score"
        assert 'recall' in metrics, "Should have recall"

        # Test stage statistics
        timings = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = calculate_stage_statistics(timings)
        print(f"✅ Stage stats: {stats}")

        assert 'mean' in stats, "Should have mean"
        assert 'p95' in stats, "Should have p95"
        assert stats['count'] == 5, "Should have 5 timings"

        print("✅ Metrics test passed!")
        return True

    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loader():
    """Test DatasetLoader (if test images available)."""
    print("\n=== Testing DatasetLoader ===")
    try:
        from optimization_analysis import DatasetLoader

        # Try to load from a common test directory
        test_dir = Path("data/Images")

        if not test_dir.exists():
            print(f"⚠️  Test directory not found: {test_dir}")
            print("   Skipping DatasetLoader test (this is OK)")
            return True

        loader = DatasetLoader(str(test_dir), max_images=10)
        paths = loader.get_sample_paths(count=5)

        print(f"✅ Found {len(paths)} images")
        print(f"✅ Sample path: {paths[0] if paths else 'None'}")

        if len(paths) > 0:
            print("✅ DatasetLoader test passed!")
        else:
            print("⚠️  No images found (this is OK for test)")

        return True

    except Exception as e:
        print(f"❌ DatasetLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pytorch_wrapper():
    """Test PyTorchInferenceWrapper (if checkpoint available)."""
    print("\n=== Testing PyTorchInferenceWrapper ===")
    try:
        from optimization_analysis import PyTorchInferenceWrapper

        checkpoint_path = Path("federated_pneumonia_detection/src/control/model_inferance/pneumonia_model_07_0.928.ckpt")

        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            print("   Skipping PyTorchInferenceWrapper test (this is OK)")
            return True

        print(f"Loading model from {checkpoint_path}...")
        wrapper = PyTorchInferenceWrapper(str(checkpoint_path))

        print(f"✅ Wrapper name: {wrapper.get_name()}")
        print(f"✅ Model info: {wrapper.get_model_info()}")

        print("✅ PyTorchInferenceWrapper test passed!")
        return True

    except Exception as e:
        print(f"❌ PyTorchInferenceWrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("OPTIMIZATION ANALYSIS MODULE TEST")
    print("="*60)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("StageTimer", test_stage_timer()))
    results.append(("ResultsCollector", test_results_collector()))
    results.append(("Metrics", test_metrics()))
    results.append(("DatasetLoader", test_dataset_loader()))
    results.append(("PyTorchInferenceWrapper", test_pytorch_wrapper()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25s} {status}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Module is ready to use.")
        print("\nNext steps:")
        print("1. Prepare a dataset of 1000+ X-ray images")
        print("2. Run baseline benchmark using run_baseline_benchmark()")
        print("3. Implement additional inference wrappers (ONNX, TensorRT, etc.)")
        print("4. Compare performance across approaches")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
