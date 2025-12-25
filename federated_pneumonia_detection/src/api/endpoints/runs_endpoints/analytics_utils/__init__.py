"""
Analytics utilities package.

Provides metric extraction, aggregation, and run detail formatting
for analytics endpoints.
"""

from .aggregators import calculate_mode_statistics, safe_average
from .extractors import extract_federated_metrics, extract_centralized_metrics

__all__ = [
    "calculate_mode_statistics",
    "safe_average",
    "extract_federated_metrics",
    "extract_centralized_metrics",
]
