"""
Pydantic schemas for experiment results.

Defines data structures for single experiment runs and aggregated metrics.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class MetricStats(BaseModel):
    """Statistical summary for a single metric across runs."""

    mean: float
    std: float
    min: float
    max: float
    values: List[float] = Field(description="Individual run values")

    @classmethod
    def from_values(cls, values: List[float]) -> "MetricStats":
        """Create MetricStats from a list of values."""
        import numpy as np

        arr = np.array(values)
        return cls(
            mean=float(np.mean(arr)),
            std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            values=values,
        )


class ExperimentResult(BaseModel):
    """Result from a single training run."""

    run_id: int = Field(description="Run identifier (1-indexed)")
    training_mode: Literal["centralized", "federated"]
    seed: int = Field(description="Random seed used for this run")
    final_metrics: Dict[str, float] = Field(
        description="Final epoch metrics: accuracy, precision, recall, f1, auroc"
    )
    metrics_history: List[Dict[str, float]] = Field(
        description="Per-epoch metrics history"
    )
    training_duration_seconds: float = Field(description="Total training time")
    best_epoch: int = Field(description="Epoch with best validation metric")
    best_model_path: Optional[str] = Field(default=None, description="Path to best checkpoint")
    config_snapshot: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration used for this run"
    )
    timestamp: datetime = Field(default_factory=datetime.now)

    def get_metric_series(self, metric_name: str) -> List[float]:
        """Extract a metric series from history."""
        return [epoch.get(metric_name, 0.0) for epoch in self.metrics_history]


class AggregatedMetrics(BaseModel):
    """Aggregated metrics across multiple runs for one training approach."""

    training_mode: Literal["centralized", "federated"]
    n_runs: int
    metrics: Dict[str, MetricStats] = Field(
        description="Statistics for each metric across runs"
    )
    mean_training_duration: float
    std_training_duration: float

    @classmethod
    def from_results(
        cls, results: List[ExperimentResult], training_mode: str
    ) -> "AggregatedMetrics":
        """Aggregate multiple experiment results."""
        import numpy as np

        if not results:
            raise ValueError("Cannot aggregate empty results list")

        metric_names = list(results[0].final_metrics.keys())
        metrics = {}

        for metric in metric_names:
            values = [r.final_metrics.get(metric, 0.0) for r in results]
            metrics[metric] = MetricStats.from_values(values)

        durations = [r.training_duration_seconds for r in results]

        return cls(
            training_mode=training_mode,
            n_runs=len(results),
            metrics=metrics,
            mean_training_duration=float(np.mean(durations)),
            std_training_duration=float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0,
        )
