"""Visualization module for publication-quality figures."""

from analysis.visualization.style import set_publication_style
from analysis.visualization.learning_curves import LearningCurvePlotter
from analysis.visualization.metric_bars import MetricBarPlotter
from analysis.visualization.distributions import DistributionPlotter
from analysis.visualization.roc_curves import ROCCurvePlotter

__all__ = [
    "set_publication_style",
    "LearningCurvePlotter",
    "MetricBarPlotter",
    "DistributionPlotter",
    "ROCCurvePlotter",
]
