"""
Result exporters for various output formats.

Handles exporting analysis results to JSON files
and managing figure exports.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis.schemas.comparison import ComparativeResult
from analysis.schemas.statistics import StatisticalAnalysisResult


class ResultsExporter:
    """
    Exports analysis results to various formats.

    Handles:
    - JSON raw results export
    - Figure organization and copying
    - Complete output package creation
    """

    def __init__(
        self,
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize exporter.

        Args:
            output_dir: Base output directory
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(
        self,
        comparative_result: ComparativeResult,
        statistical_result: Optional[StatisticalAnalysisResult] = None,
        data_profile: Optional[Dict] = None,
        filename: str = "raw_results.json",
    ) -> Path:
        """
        Export complete results to JSON.

        Args:
            comparative_result: Comparison results
            statistical_result: Optional statistical results
            data_profile: Optional data profile
            filename: Output filename

        Returns:
            Path to exported file
        """
        export_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0",
            },
            "comparative_result": comparative_result.model_dump(mode="json"),
        }

        if statistical_result:
            export_data["statistical_result"] = statistical_result.model_dump(mode="json")

        if data_profile:
            export_data["data_profile"] = data_profile

        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Exported results to {output_path}")
        return output_path

    def export_individual_runs(
        self,
        comparative_result: ComparativeResult,
    ) -> List[Path]:
        """
        Export individual run results to separate files.

        Args:
            comparative_result: Comparison results

        Returns:
            List of exported file paths
        """
        runs_dir = self.output_dir / "experiments"
        runs_dir.mkdir(parents=True, exist_ok=True)

        exported = []

        for result in comparative_result.centralized_runs:
            path = runs_dir / f"centralized_run_{result.run_id}.json"
            with open(path, "w") as f:
                json.dump(result.model_dump(mode="json"), f, indent=2, default=str)
            exported.append(path)

        for result in comparative_result.federated_runs:
            path = runs_dir / f"federated_run_{result.run_id}.json"
            with open(path, "w") as f:
                json.dump(result.model_dump(mode="json"), f, indent=2, default=str)
            exported.append(path)

        self.logger.info(f"Exported {len(exported)} individual run files")
        return exported

    def export_summary_csv(
        self,
        comparative_result: ComparativeResult,
        filename: str = "metrics_summary.csv",
    ) -> Path:
        """
        Export metrics summary to CSV.

        Args:
            comparative_result: Comparison results
            filename: Output filename

        Returns:
            Path to exported file
        """
        import csv

        output_path = self.output_dir / filename
        metrics = list(comparative_result.centralized_aggregated.metrics.keys())

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Metric",
                "Centralized Mean",
                "Centralized Std",
                "Federated Mean",
                "Federated Std",
                "Difference",
                "Percent Change",
            ])

            for metric in metrics:
                comparison = comparative_result.get_metric_comparison(metric)
                writer.writerow([
                    metric,
                    f"{comparison['centralized_mean']:.6f}",
                    f"{comparison['centralized_std']:.6f}",
                    f"{comparison['federated_mean']:.6f}",
                    f"{comparison['federated_std']:.6f}",
                    f"{comparison['difference']:.6f}",
                    f"{comparison['percent_difference']:.4f}",
                ])

        self.logger.info(f"Exported summary CSV to {output_path}")
        return output_path

    def organize_figures(
        self,
        source_dirs: List[Path],
        figures_subdir: str = "figures",
    ) -> Path:
        """
        Collect and organize all figures into one directory.

        Args:
            source_dirs: List of directories containing figures
            figures_subdir: Subdirectory name for figures

        Returns:
            Path to organized figures directory
        """
        figures_dir = self.output_dir / figures_subdir
        figures_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for source_dir in source_dirs:
            if not source_dir.exists():
                continue

            for fig_path in source_dir.glob("*.png"):
                dest = figures_dir / fig_path.name
                shutil.copy2(fig_path, dest)
                copied += 1

            for fig_path in source_dir.glob("*.pdf"):
                dest = figures_dir / fig_path.name
                shutil.copy2(fig_path, dest)
                copied += 1

        self.logger.info(f"Organized {copied} figures to {figures_dir}")
        return figures_dir

    def create_output_package(
        self,
        comparative_result: ComparativeResult,
        statistical_result: StatisticalAnalysisResult,
        data_profile: Optional[Dict] = None,
        figure_dirs: Optional[List[Path]] = None,
    ) -> Dict[str, Path]:
        """
        Create complete output package with all exports.

        Args:
            comparative_result: Comparison results
            statistical_result: Statistical results
            data_profile: Optional data profile
            figure_dirs: Optional list of figure directories

        Returns:
            Dictionary mapping output types to paths
        """
        outputs = {}

        outputs["raw_json"] = self.export_json(
            comparative_result, statistical_result, data_profile
        )

        outputs["individual_runs"] = self.output_dir / "experiments"
        self.export_individual_runs(comparative_result)

        outputs["summary_csv"] = self.export_summary_csv(comparative_result)

        if figure_dirs:
            outputs["figures"] = self.organize_figures(figure_dirs)

        self.logger.info(f"Created complete output package at {self.output_dir}")
        return outputs
