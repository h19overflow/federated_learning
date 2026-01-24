"""Internal implementation of analytics services.

This module contains all service implementations, utilities, and infrastructure
for the analytics facade. These should not be imported directly by endpoints.

Backward Compatibility:
    This module maintains backward compatibility by re-exporting all services,
    extractors, exporters, and utilities from their new locations in subdirectories.
"""

# Services (from services/ subdirectory) - These are the CONCRETE implementations
# NOTE: We do NOT import from .definitions to avoid Protocol name conflicts
from .services import (
    BackfillService,
    ExportService,
    FinalEpochStatsService,
    MetricsService,
    RankingService,
    SummaryService,
)

# Infrastructure (from infrastructure/ subdirectory)
from .infrastructure import CacheProvider

# Extractors (from extractors/ subdirectory)
from .extractors import (
    CentralizedMetricExtractor,
    FederatedMetricExtractor,
    MetricExtractor,
)

# Exporters (from exporters/ subdirectory)
from .exporters import (
    CSVExporter,
    DataExporter,
    JSONExporter,
    TextReportExporter,
)

# Utils (from utils/ subdirectory)
from .utils import (
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
    # Exporters
    "CSVExporter",
    "DataExporter",
    "JSONExporter",
    "TextReportExporter",
    # Transformers
    "calculate_summary_statistics",
    "find_best_epoch",
    "transform_run_to_results",
]
