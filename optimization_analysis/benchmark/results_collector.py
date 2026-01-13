from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BenchmarkResult:
    approach_name: str
    num_samples: int
    stage_stats: Dict[str, Dict[str, float]]
    accuracy_metrics: Dict[str, float]
    total_time_avg: float
    timestamp: str


class ResultsCollector:
    """Collect and aggregate benchmark results."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_benchmarks': len(self.results),
            'approaches_tested': list(set(r.approach_name for r in self.results)),
            'results': [self._format_result(r) for r in self.results]
        }

    def compare_approaches(self) -> Dict[str, Dict]:
        comparison = {}
        for result in self.results:
            if result.approach_name not in comparison:
                comparison[result.approach_name] = {
                    'total_time_avg': result.total_time_avg,
                    'accuracy': result.accuracy_metrics,
                    'stage_breakdown': result.stage_stats
                }
        return comparison

    def _format_result(self, result: BenchmarkResult) -> Dict:
        return {
            'approach': result.approach_name,
            'samples': result.num_samples,
            'timestamp': result.timestamp,
            'total_time_ms': result.total_time_avg,
            'accuracy': result.accuracy_metrics,
            'stages': result.stage_stats
        }

    def save_to_file(self, filepath: str):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
