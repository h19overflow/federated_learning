import time
from contextlib import contextmanager
from typing import Dict, List
import statistics


class StageTimer:
    """Timer for measuring individual pipeline stages."""

    def __init__(self):
        self.timings = {
            'preprocessing': [],
            'feature_extraction': [],
            'classification': [],
            'total': []
        }

    @contextmanager
    def time_stage(self, stage_name: str):
        start = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - start) * 1000  # ms
        self.timings[stage_name].append(elapsed)

    def get_statistics(self, stage_name: str) -> Dict:
        times = self.timings[stage_name]
        if not times:
            return {}

        times_sorted = sorted(times)
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'p50': times_sorted[int(len(times) * 0.5)],
            'p95': times_sorted[int(len(times) * 0.95)],
            'p99': times_sorted[int(len(times) * 0.99)],
            'min': min(times),
            'max': max(times),
            'stddev': statistics.stdev(times) if len(times) > 1 else 0,
            'count': len(times)
        }

    def reset(self):
        self.timings = {k: [] for k in self.timings}
