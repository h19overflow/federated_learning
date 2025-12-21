"""
Metric comparison bar charts for analysis results.

Generates grouped bar charts with error bars and
significance indicators for comparing approaches.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from analysis.schemas.comparison import ComparativeResult
from analysis.schemas.statistics import StatisticalAnalysisResult
from analysis.visualization.style import (
    add_significance_stars,
    create_figure,
    get_color,
    set_publication_style,
)


class MetricBarPlotter:
    """
    Generates bar chart comparisons of metrics.

    Creates:
    - Grouped bar charts with error bars
    - Significance stars for statistical tests
    - Publication-ready formatting
    """

    def __init__(
        self,
        output_dir: Path,
        dpi: int = 300,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize plotter.

        Args:
            output_dir: Directory to save figures
            dpi: Figure resolution
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_publication_style()

    def plot_comparison(
        self,
        comparative_result: ComparativeResult,
        statistical_result: Optional[StatisticalAnalysisResult] = None,
        metrics: Optional[List[str]] = None,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot grouped bar chart comparing metrics.

        Args:
            comparative_result: Aggregated comparison results
            statistical_result: Optional statistical test results
            metrics: Metrics to include (default: all)
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = list(comparative_result.centralized_aggregated.metrics.keys())

        fig, ax = create_figure("wide")

        x = np.arange(len(metrics))
        width = 0.35

        cent_means = []
        cent_stds = []
        fed_means = []
        fed_stds = []

        for metric in metrics:
            cent_stats = comparative_result.centralized_aggregated.metrics[metric]
            fed_stats = comparative_result.federated_aggregated.metrics[metric]

            cent_means.append(cent_stats.mean)
            cent_stds.append(cent_stats.std)
            fed_means.append(fed_stats.mean)
            fed_stds.append(fed_stats.std)

        bars1 = ax.bar(
            x - width/2, cent_means, width,
            yerr=cent_stds,
            label="Centralized",
            color=get_color("centralized"),
            edgecolor="black",
            linewidth=1,
            capsize=4,
        )

        bars2 = ax.bar(
            x + width/2, fed_means, width,
            yerr=fed_stds,
            label="Federated",
            color=get_color("federated"),
            edgecolor="black",
            linewidth=1,
            capsize=4,
        )

        if statistical_result:
            for i, metric in enumerate(metrics):
                if metric in statistical_result.metrics:
                    p_value = statistical_result.metrics[metric].comparison_test.p_value
                    max_height = max(cent_means[i] + cent_stds[i], fed_means[i] + fed_stds[i])
                    add_significance_stars(
                        ax, i - width/2, i + width/2, max_height * 1.05, p_value
                    )

        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title("Centralized vs Federated Learning: Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([m.title() for m in metrics], rotation=45, ha="right")
        ax.legend(loc="upper left")

        ax.set_ylim(0, max(max(cent_means), max(fed_means)) * 1.25)

        plt.tight_layout()

        if save:
            path = self.output_dir / "metric_comparison.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            self.logger.info(f"Saved metric comparison to {path}")

        return fig

    def plot_difference_chart(
        self,
        comparative_result: ComparativeResult,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot difference chart (federated - centralized).

        Args:
            comparative_result: Comparison results
            save: Whether to save

        Returns:
            Matplotlib figure
        """
        fig, ax = create_figure("double_column")

        metrics = list(comparative_result.centralized_aggregated.metrics.keys())
        differences = []
        colors = []

        for metric in metrics:
            comparison = comparative_result.get_metric_comparison(metric)
            diff = comparison["difference"]
            differences.append(diff)
            colors.append(get_color("federated") if diff >= 0 else get_color("highlight"))

        y_pos = np.arange(len(metrics))
        ax.barh(y_pos, differences, color=colors, edgecolor="black", linewidth=1)

        ax.axvline(x=0, color="black", linestyle="-", linewidth=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([m.title() for m in metrics])
        ax.set_xlabel("Difference (Federated - Centralized)")
        ax.set_title("Performance Difference by Metric")

        for i, (diff, metric) in enumerate(zip(differences, metrics)):
            x_pos = diff + 0.005 if diff >= 0 else diff - 0.005
            ha = "left" if diff >= 0 else "right"
            ax.text(x_pos, i, f"{diff:+.4f}", va="center", ha=ha, fontsize=10)

        plt.tight_layout()

        if save:
            path = self.output_dir / "metric_difference.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            self.logger.info(f"Saved difference chart to {path}")

        return fig
