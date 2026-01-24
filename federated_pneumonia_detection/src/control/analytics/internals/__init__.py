"""Internal implementation of analytics services.

This module contains all service implementations, utilities, and infrastructure
for the analytics facade. These should not be imported directly by endpoints.
"""

from .backfill_service import BackfillService
from .cache import CacheProvider
from .definitions import *  # All types and protocols
from .export_service import ExportService
from .final_epoch_stats_service import FinalEpochStatsService
from .metric_extractors import (
    CentralizedMetricExtractor,
    FederatedMetricExtractor,
    MetricExtractor,
)
from .metrics_service import MetricsService
from .ranking_service import RankingService
from .summary_service import SummaryService
from .transformers import (
    calculate_summary_statistics,
    find_best_epoch,
    transform_run_to_results,
)

__all__ = [
    # Services
    "BackfillService",
    "CacheProvider",
    "ExportService",
    "FinalEpochStatsService",
    "MetricsService",
    "RankingService",
    "SummaryService",
    # Extractors
    "CentralizedMetricExtractor",
    "FederatedMetricExtractor",
    "MetricExtractor",
    # Transformers
    "calculate_summary_statistics",
    "find_best_epoch",
    "transform_run_to_results",
]
