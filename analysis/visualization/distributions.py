"""
Distribution plots for metric analysis.

Generates box plots and violin plots to visualize
the distribution of metrics across runs.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from analysis.schemas.comparison import ComparativeResult
from analysis.visualization.style import (
    create_figure,
    get_color,
    set_publication_style,
)


class DistributionPlotter:
    """
    Generates distribution visualizations.

    Creates:
    - Box plots for metric comparison
    - Violin plots with individual points
    - Combined distribution overview
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

    def plot_boxplots(
        self,
        comparative_result: ComparativeResult,
        metrics: Optional[List[str]] = None,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot box plots for each metric.

        Args:
            comparative_result: Comparison results
            metrics: Metrics to include
            save: Whether to save

        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = list(comparative_result.centralized_aggregated.metrics.keys())

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]

            cent_values = comparative_result.centralized_aggregated.metrics[metric].values
            fed_values = comparative_result.federated_aggregated.metrics[metric].values

            bp = ax.boxplot(
                [cent_values, fed_values],
                labels=["Centralized", "Federated"],
                patch_artist=True,
                widths=0.6,
            )

            colors = [get_color("centralized"), get_color("federated")]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            for j, (values, color) in enumerate([(cent_values, colors[0]), (fed_values, colors[1])]):
                x = np.random.normal(j + 1, 0.04, len(values))
                ax.scatter(x, values, color=color, alpha=0.6, s=30, zorder=3, edgecolors="black")

            ax.set_title(metric.title())
            ax.set_ylabel("Value")

        fig.suptitle("Metric Distributions", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save:
            path = self.output_dir / "metric_distributions.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            self.logger.info(f"Saved distribution plots to {path}")

        return fig

    def plot_violin(
        self,
        comparative_result: ComparativeResult,
        metric: str = "accuracy",
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot violin plot for a single metric.

        Args:
            comparative_result: Comparison results
            metric: Metric to plot
            save: Whether to save

        Returns:
            Matplotlib figure
        """
        fig, ax = create_figure("square")

        cent_values = comparative_result.centralized_aggregated.metrics[metric].values
        fed_values = comparative_result.federated_aggregated.metrics[metric].values

        parts = ax.violinplot(
            [cent_values, fed_values],
            positions=[1, 2],
            showmeans=True,
            showmedians=True,
        )

        colors = [get_color("centralized"), get_color("federated")]
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        for j, (values, color) in enumerate([(cent_values, colors[0]), (fed_values, colors[1])]):
            x = np.random.normal(j + 1, 0.04, len(values))
            ax.scatter(x, values, color=color, alpha=0.8, s=50, zorder=3, edgecolors="black")

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Centralized", "Federated"])
        ax.set_ylabel(metric.title())
        ax.set_title(f"{metric.title()} Distribution Comparison")

        plt.tight_layout()

        if save:
            path = self.output_dir / f"violin_{metric}.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            self.logger.info(f"Saved violin plot to {path}")

        return fig

    def plot_all_distributions(
        self,
        comparative_result: ComparativeResult,
    ) -> List[Path]:
        """
        Generate all distribution visualizations.

        Returns:
            List of generated file paths
        """
        generated = []

        self.plot_boxplots(comparative_result)
        generated.append(self.output_dir / "metric_distributions.png")

        metrics = list(comparative_result.centralized_aggregated.metrics.keys())
        for metric in metrics:
            self.plot_violin(comparative_result, metric)
            generated.append(self.output_dir / f"violin_{metric}.png")

        return generated
