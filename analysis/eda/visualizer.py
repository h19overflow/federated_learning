"""
EDA visualization module for research paper figures.

Generates publication-quality visualizations for Section 5.3,
including class distributions, sample images, and data quality plots.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from analysis.preprocessing.data_loader import AnalysisDataLoader


class EDAVisualizer:
    """
    Generates EDA visualizations for research documentation.

    Produces figures for:
    - Class distribution histograms
    - Sample image grids
    - Data quality visualizations
    """

    def __init__(
        self,
        data_loader: AnalysisDataLoader,
        output_dir: Path,
        dpi: int = 300,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize visualizer.

        Args:
            data_loader: AnalysisDataLoader with loaded data
            output_dir: Directory to save figures
            dpi: Figure resolution
            logger: Optional logger instance
        """
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._set_style()

    def _set_style(self) -> None:
        """Set matplotlib style for publication figures."""
        plt.rcParams.update({
            "font.size": 12,
            "font.family": "serif",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
            "axes.grid": True,
            "grid.alpha": 0.3,
        })

    def plot_class_distribution(self, save: bool = True) -> plt.Figure:
        """
        Create class distribution bar chart.

        Args:
            save: Whether to save figure to disk

        Returns:
            Matplotlib figure
        """
        df = self.data_loader.dataframe
        if df is None:
            df = self.data_loader.load_full_dataset()

        target_col = self.data_loader.config.get("columns.target")
        class_counts = df[target_col].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(8, 6))

        labels = ["Normal (0)", "Pneumonia (1)"]
        colors = ["#2ecc71", "#e74c3c"]
        bars = ax.bar(labels, class_counts.values, color=colors, edgecolor="black", linewidth=1.2)

        for bar, count in zip(bars, class_counts.values):
            pct = count / len(df) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + len(df) * 0.01,
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        ax.set_xlabel("Class")
        ax.set_ylabel("Number of Samples")
        ax.set_title("Class Distribution in Dataset")
        ax.set_ylim(0, max(class_counts.values) * 1.15)

        plt.tight_layout()

        if save:
            path = self.output_dir / "class_distribution.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            self.logger.info(f"Saved class distribution plot to {path}")

        return fig

    def plot_sample_images(
        self,
        n_samples: int = 4,
        save: bool = True,
    ) -> plt.Figure:
        """
        Create grid of sample images from each class.

        Args:
            n_samples: Number of samples per class
            save: Whether to save figure to disk

        Returns:
            Matplotlib figure
        """
        df = self.data_loader.dataframe
        if df is None:
            df = self.data_loader.load_full_dataset()

        image_dir = self.data_loader.image_dir
        if image_dir is None:
            self.data_loader.extract_and_validate()
            image_dir = self.data_loader.image_dir

        target_col = self.data_loader.config.get("columns.target")
        filename_col = self.data_loader.config.get("columns.filename")

        fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))

        for row_idx, (label, class_name) in enumerate([("0", "Normal"), ("1", "Pneumonia")]):
            class_df = df[df[target_col] == label].sample(n=min(n_samples, len(df[df[target_col] == label])), random_state=42)

            for col_idx, (_, row) in enumerate(class_df.iterrows()):
                if col_idx >= n_samples:
                    break

                ax = axes[row_idx, col_idx]
                img_path = image_dir / row[filename_col]

                try:
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                    ax.set_title(f"{class_name}", fontsize=12)
                except Exception as e:
                    ax.text(0.5, 0.5, "Image\nNot Found", ha="center", va="center", transform=ax.transAxes)
                    self.logger.warning(f"Could not load image: {img_path}")

                ax.axis("off")

        plt.suptitle("Sample X-Ray Images by Class", fontsize=16, y=1.02)
        plt.tight_layout()

        if save:
            path = self.output_dir / "sample_images.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            self.logger.info(f"Saved sample images to {path}")

        return fig

    def plot_data_summary(self, save: bool = True) -> plt.Figure:
        """
        Create data summary visualization with multiple panels.

        Args:
            save: Whether to save figure to disk

        Returns:
            Matplotlib figure
        """
        df = self.data_loader.dataframe
        if df is None:
            df = self.data_loader.load_full_dataset()

        stats = self.data_loader.get_statistics()
        target_col = self.data_loader.config.get("columns.target")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Class distribution pie chart
        ax1 = axes[0]
        class_counts = df[target_col].value_counts().sort_index()
        colors = ["#2ecc71", "#e74c3c"]
        wedges, texts, autotexts = ax1.pie(
            class_counts.values,
            labels=["Normal", "Pneumonia"],
            autopct="%1.1f%%",
            colors=colors,
            explode=(0.02, 0.02),
            startangle=90,
        )
        ax1.set_title("Class Distribution")

        # Panel 2: Dataset statistics text
        ax2 = axes[1]
        ax2.axis("off")
        stats_text = (
            f"Dataset Statistics\n"
            f"{'─' * 30}\n"
            f"Total Samples: {stats['total_samples']:,}\n"
            f"Normal Cases: {class_counts.get('0', 0):,}\n"
            f"Pneumonia Cases: {class_counts.get('1', 0):,}\n"
            f"Balance Ratio: {stats.get('class_balance_ratio', 1.0):.3f}\n"
            f"Missing Values: {sum(stats['missing_values'].values())}"
        )
        ax2.text(
            0.5, 0.5, stats_text,
            ha="center", va="center",
            fontsize=14,
            family="monospace",
            transform=ax2.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3),
        )

        # Panel 3: Class imbalance bar
        ax3 = axes[2]
        imbalance_ratio = stats.get("class_balance_ratio", 1.0)
        ax3.barh(["Balance\nRatio"], [imbalance_ratio], color="#3498db", height=0.5)
        ax3.axvline(x=0.5, color="red", linestyle="--", label="Imbalance threshold")
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("Ratio (minority/majority)")
        ax3.set_title("Class Balance Analysis")
        ax3.legend(loc="lower right")

        plt.suptitle("Dataset Overview", fontsize=16, y=1.02)
        plt.tight_layout()

        if save:
            path = self.output_dir / "data_summary.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            self.logger.info(f"Saved data summary to {path}")

        return fig

    def generate_all(self) -> List[Path]:
        """
        Generate all EDA visualizations.

        Returns:
            List of paths to generated figures
        """
        generated = []

        self.plot_class_distribution()
        generated.append(self.output_dir / "class_distribution.png")

        self.plot_sample_images()
        generated.append(self.output_dir / "sample_images.png")

        self.plot_data_summary()
        generated.append(self.output_dir / "data_summary.png")

        self.logger.info(f"Generated {len(generated)} EDA figures")
        return generated
