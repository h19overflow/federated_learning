"""
Benchmark package for performance analysis.
"""

from optimization_analysis.benchmark.benchmark_suite import (
    BenchmarkSuite,
    create_default_suite,
    run_baseline_benchmark
)

__all__ = [
    'BenchmarkSuite',
    'create_default_suite',
    'run_baseline_benchmark'
]
