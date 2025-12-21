"""
Pydantic schemas for comparative analysis results.

Combines centralized and federated experiment results with statistical analysis.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from analysis.schemas.experiment import AggregatedMetrics, ExperimentResult
from analysis.schemas.statistics import StatisticalAnalysisResult


class ComparativeResult(BaseModel):
    """Complete comparative analysis result."""

    centralized_runs: List[ExperimentResult] = Field(
        description="Individual centralized training runs"
    )
    federated_runs: List[ExperimentResult] = Field(
        description="Individual federated training runs"
    )
    centralized_aggregated: AggregatedMetrics = Field(
        description="Aggregated centralized metrics"
    )
    federated_aggregated: AggregatedMetrics = Field(
        description="Aggregated federated metrics"
    )
    statistical_analysis: Optional[StatisticalAnalysisResult] = Field(
        default=None, description="Statistical comparison results"
    )
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    config_used: Dict = Field(default_factory=dict, description="Analysis configuration")

    def get_metric_comparison(self, metric_name: str) -> Dict:
        """Get comparison summary for a specific metric."""
        cent_stats = self.centralized_aggregated.metrics.get(metric_name)
        fed_stats = self.federated_aggregated.metrics.get(metric_name)

        if not cent_stats or not fed_stats:
            return {}

        diff = fed_stats.mean - cent_stats.mean
        pct_diff = (diff / cent_stats.mean * 100) if cent_stats.mean != 0 else 0

        return {
            "metric": metric_name,
            "centralized_mean": cent_stats.mean,
            "centralized_std": cent_stats.std,
            "federated_mean": fed_stats.mean,
            "federated_std": fed_stats.std,
            "difference": diff,
            "percent_difference": pct_diff,
        }

    def summary_dict(self) -> Dict:
        """Generate summary dictionary for all metrics."""
        metrics = list(self.centralized_aggregated.metrics.keys())
        return {m: self.get_metric_comparison(m) for m in metrics}
