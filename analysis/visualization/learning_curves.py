"""
Learning curve visualization for training analysis.

Generates plots showing training and validation metrics
across epochs/rounds with confidence bands.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from analysis.schemas.experiment import ExperimentResult
from analysis.visualization.style import (
    create_figure,
    get_color,
    set_publication_style,
)


class LearningCurvePlotter:
    """
    Generates learning curve visualizations.

    Creates plots showing:
    - Loss curves with shaded std regions
    - Metric progression over epochs/rounds
    - Comparison between approaches
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

    def plot_loss_curves(
        self,
        centralized_results: List[ExperimentResult],
        federated_results: List[ExperimentResult],
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot training/validation loss curves for both approaches.

        Args:
            centralized_results: Centralized experiment results
            federated_results: Federated experiment results
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = create_figure("wide", nrows=1, ncols=2)

        self._plot_single_approach_loss(
            axes[0],
            centralized_results,
            "Centralized Learning",
            get_color("centralized"),
        )

        self._plot_single_approach_loss(
            axes[1],
            federated_results,
            "Federated Learning",
            get_color("federated"),
        )

        fig.suptitle("Training Loss Curves", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save:
            path = self.output_dir / "learning_curves_loss.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            self.logger.info(f"Saved loss curves to {path}")

        return fig

    def _plot_single_approach_loss(
        self,
        ax,
        results: List[ExperimentResult],
        title: str,
        color: str,
    ) -> None:
        """Plot loss curves for a single approach."""
        all_train_loss = []
        all_val_loss = []

        max_epochs = max(len(r.metrics_history) for r in results)

        for result in results:
            train_loss = [e.get("train_loss", np.nan) for e in result.metrics_history]
            val_loss = [e.get("val_loss", np.nan) for e in result.metrics_history]

            while len(train_loss) < max_epochs:
                train_loss.append(np.nan)
                val_loss.append(np.nan)

            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

        all_train_loss = np.array(all_train_loss)
        all_val_loss = np.array(all_val_loss)

        epochs = np.arange(1, max_epochs + 1)

        train_mean = np.nanmean(all_train_loss, axis=0)
        train_std = np.nanstd(all_train_loss, axis=0)
        val_mean = np.nanmean(all_val_loss, axis=0)
        val_std = np.nanstd(all_val_loss, axis=0)

        ax.plot(epochs, train_mean, color=color, linestyle="-", label="Training", linewidth=2)
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                        color=color, alpha=0.2)

        ax.plot(epochs, val_mean, color=color, linestyle="--", label="Validation", linewidth=2)
        ax.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                        color=color, alpha=0.1)

        ax.set_xlabel("Epoch/Round")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.set_xlim(1, max_epochs)

    def plot_metric_progression(
        self,
        centralized_results: List[ExperimentResult],
        federated_results: List[ExperimentResult],
        metric: str = "val_acc",
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot metric progression comparison.

        Args:
            centralized_results: Centralized results
            federated_results: Federated results
            metric: Metric to plot
            save: Whether to save

        Returns:
            Matplotlib figure
        """
        fig, ax = create_figure("double_column")

        self._plot_metric_band(
            ax, centralized_results, metric,
            get_color("centralized"), "Centralized"
        )
        self._plot_metric_band(
            ax, federated_results, metric,
            get_color("federated"), "Federated"
        )

        metric_label = metric.replace("val_", "").replace("_", " ").title()
        ax.set_xlabel("Epoch/Round")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} Progression")
        ax.legend(loc="lower right")

        plt.tight_layout()

        if save:
            path = self.output_dir / f"learning_curves_{metric}.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            self.logger.info(f"Saved metric progression to {path}")

        return fig

    def _plot_metric_band(
        self,
        ax,
        results: List[ExperimentResult],
        metric: str,
        color: str,
        label: str,
    ) -> None:
        """Plot metric with confidence band."""
        all_values = []
        max_epochs = max(len(r.metrics_history) for r in results)

        for result in results:
            values = [e.get(metric, np.nan) for e in result.metrics_history]
            while len(values) < max_epochs:
                values.append(np.nan)
            all_values.append(values)

        all_values = np.array(all_values)
        epochs = np.arange(1, max_epochs + 1)

        mean = np.nanmean(all_values, axis=0)
        std = np.nanstd(all_values, axis=0)

        ax.plot(epochs, mean, color=color, label=label, linewidth=2)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)

    def generate_all(
        self,
        centralized_results: List[ExperimentResult],
        federated_results: List[ExperimentResult],
    ) -> List[Path]:
        """
        Generate all learning curve visualizations.

        Returns:
            List of generated file paths
        """
        generated = []

        self.plot_loss_curves(centralized_results, federated_results)
        generated.append(self.output_dir / "learning_curves_loss.png")

        for metric in ["val_acc", "val_recall", "val_f1"]:
            self.plot_metric_progression(centralized_results, federated_results, metric)
            generated.append(self.output_dir / f"learning_curves_{metric}.png")

        return generated
