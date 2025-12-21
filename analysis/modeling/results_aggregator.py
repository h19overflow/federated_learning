"""
Results aggregator for comparative analysis.

Aggregates metrics across multiple runs and prepares data
for statistical analysis and visualization.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from analysis.schemas.experiment import AggregatedMetrics, ExperimentResult, MetricStats
from analysis.schemas.comparison import ComparativeResult


class ResultsAggregator:
    """
    Aggregates experiment results for statistical comparison.

    Combines results from centralized and federated experiments
    into a unified format for analysis.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize aggregator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self._comparative_result: Optional[ComparativeResult] = None

    def aggregate(
        self,
        centralized_results: List[ExperimentResult],
        federated_results: List[ExperimentResult],
    ) -> ComparativeResult:
        """
        Aggregate results from both training approaches.

        Args:
            centralized_results: List of centralized experiment results
            federated_results: List of federated experiment results

        Returns:
            ComparativeResult with aggregated metrics
        """
        centralized_agg = AggregatedMetrics.from_results(
            centralized_results, "centralized"
        )
        federated_agg = AggregatedMetrics.from_results(
            federated_results, "federated"
        )

        self._comparative_result = ComparativeResult(
            centralized_runs=centralized_results,
            federated_runs=federated_results,
            centralized_aggregated=centralized_agg,
            federated_aggregated=federated_agg,
        )

        self.logger.info(
            f"Aggregated {len(centralized_results)} centralized and "
            f"{len(federated_results)} federated runs"
        )

        return self._comparative_result

    def get_metrics_for_comparison(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Get metrics organized for statistical tests.

        Returns:
            Dict with metric names as keys, containing centralized and federated values
        """
        if self._comparative_result is None:
            raise ValueError("No results aggregated. Call aggregate() first.")

        metrics = {}
        metric_names = list(self._comparative_result.centralized_aggregated.metrics.keys())

        for metric in metric_names:
            cent_stats = self._comparative_result.centralized_aggregated.metrics[metric]
            fed_stats = self._comparative_result.federated_aggregated.metrics[metric]

            metrics[metric] = {
                "centralized": cent_stats.values,
                "federated": fed_stats.values,
            }

        return metrics

    def get_summary_table(self) -> List[Dict]:
        """
        Generate summary table data for all metrics.

        Returns:
            List of dicts suitable for tabular display
        """
        if self._comparative_result is None:
            raise ValueError("No results aggregated. Call aggregate() first.")

        rows = []
        metric_names = list(self._comparative_result.centralized_aggregated.metrics.keys())

        for metric in metric_names:
            comparison = self._comparative_result.get_metric_comparison(metric)
            rows.append({
                "metric": metric,
                "centralized_mean": f"{comparison['centralized_mean']:.4f}",
                "centralized_std": f"{comparison['centralized_std']:.4f}",
                "federated_mean": f"{comparison['federated_mean']:.4f}",
                "federated_std": f"{comparison['federated_std']:.4f}",
                "difference": f"{comparison['difference']:+.4f}",
                "percent_diff": f"{comparison['percent_difference']:+.2f}%",
            })

        return rows

    def save_results(self, output_path: Path) -> None:
        """
        Save aggregated results to JSON.

        Args:
            output_path: Path to save JSON file
        """
        if self._comparative_result is None:
            raise ValueError("No results aggregated. Call aggregate() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(
                self._comparative_result.model_dump(mode="json"),
                f,
                indent=2,
                default=str,
            )

        self.logger.info(f"Saved aggregated results to {output_path}")

    def load_results(self, input_path: Path) -> ComparativeResult:
        """
        Load previously saved results.

        Args:
            input_path: Path to JSON file

        Returns:
            Loaded ComparativeResult
        """
        with open(input_path) as f:
            data = json.load(f)

        self._comparative_result = ComparativeResult.model_validate(data)
        self.logger.info(f"Loaded results from {input_path}")
        return self._comparative_result

    @property
    def comparative_result(self) -> Optional[ComparativeResult]:
        """Get aggregated comparative result."""
        return self._comparative_result
