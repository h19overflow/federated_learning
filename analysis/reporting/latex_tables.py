"""
LaTeX table generation for research papers.

Generates publication-ready LaTeX tables for metrics
comparison and statistical analysis results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from analysis.schemas.comparison import ComparativeResult
from analysis.schemas.statistics import StatisticalAnalysisResult


class LatexTableGenerator:
    """
    Generates LaTeX tables for academic papers.

    Creates:
    - Metrics comparison tables (mean +/- std)
    - Statistical significance tables
    - Effect size summary tables
    """

    def __init__(
        self,
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize generator.

        Args:
            output_dir: Directory to save .tex files
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_metrics_table(
        self,
        comparative_result: ComparativeResult,
        caption: str = "Performance Comparison: Centralized vs Federated Learning",
        label: str = "tab:metrics_comparison",
        save: bool = True,
    ) -> str:
        """
        Generate metrics comparison table.

        Args:
            comparative_result: Comparison results
            caption: Table caption
            label: LaTeX label
            save: Whether to save to file

        Returns:
            LaTeX table string
        """
        metrics = list(comparative_result.centralized_aggregated.metrics.keys())

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Metric & Centralized & Federated & Difference & \% Change \\",
            r"\midrule",
        ]

        for metric in metrics:
            comparison = comparative_result.get_metric_comparison(metric)

            cent = f"{comparison['centralized_mean']:.4f} $\\pm$ {comparison['centralized_std']:.4f}"
            fed = f"{comparison['federated_mean']:.4f} $\\pm$ {comparison['federated_std']:.4f}"
            diff = f"{comparison['difference']:+.4f}"
            pct = f"{comparison['percent_difference']:+.2f}\\%"

            metric_name = metric.replace("_", " ").title()
            lines.append(rf"{metric_name} & {cent} & {fed} & {diff} & {pct} \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        table = "\n".join(lines)

        if save:
            path = self.output_dir / "metrics_table.tex"
            with open(path, "w") as f:
                f.write(table)
            self.logger.info(f"Saved metrics table to {path}")

        return table

    def generate_statistical_table(
        self,
        statistical_result: StatisticalAnalysisResult,
        caption: str = "Statistical Analysis Results",
        label: str = "tab:statistical_analysis",
        save: bool = True,
    ) -> str:
        """
        Generate statistical analysis table.

        Args:
            statistical_result: Statistical test results
            caption: Table caption
            label: LaTeX label
            save: Whether to save to file

        Returns:
            LaTeX table string
        """
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Metric & Test & p-value & Significant & Cohen's d & Effect Size \\",
            r"\midrule",
        ]

        for metric_name, result in statistical_result.metrics.items():
            test_name = result.comparison_test.test_name
            p_value = result.comparison_test.p_value
            significant = "Yes" if result.comparison_test.is_significant else "No"
            cohens_d = result.effect_size.cohens_d
            magnitude = result.effect_size.magnitude.title()

            p_str = f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001"
            if result.comparison_test.is_significant:
                p_str = rf"\textbf{{{p_str}}}"
                significant = r"\textbf{Yes}"

            metric_display = metric_name.replace("_", " ").title()
            lines.append(
                rf"{metric_display} & {test_name} & {p_str} & {significant} & {cohens_d:.3f} & {magnitude} \\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            rf"\\ \footnotesize{{Note: Significance level $\alpha$ = {statistical_result.alpha}}}",
            r"\end{table}",
        ])

        table = "\n".join(lines)

        if save:
            path = self.output_dir / "statistical_summary.tex"
            with open(path, "w") as f:
                f.write(table)
            self.logger.info(f"Saved statistical table to {path}")

        return table

    def generate_effect_size_table(
        self,
        statistical_result: StatisticalAnalysisResult,
        caption: str = "Effect Size Analysis with Confidence Intervals",
        label: str = "tab:effect_sizes",
        save: bool = True,
    ) -> str:
        """
        Generate effect size table with confidence intervals.

        Args:
            statistical_result: Statistical results
            caption: Table caption
            label: LaTeX label
            save: Whether to save to file

        Returns:
            LaTeX table string
        """
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Metric & Cohen's $d$ & 95\% CI & Interpretation \\",
            r"\midrule",
        ]

        for metric_name, result in statistical_result.metrics.items():
            es = result.effect_size
            ci = f"[{es.ci_lower:.3f}, {es.ci_upper:.3f}]"

            metric_display = metric_name.replace("_", " ").title()
            lines.append(
                rf"{metric_display} & {es.cohens_d:.3f} & {ci} & {es.magnitude.title()} \\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        table = "\n".join(lines)

        if save:
            path = self.output_dir / "effect_sizes.tex"
            with open(path, "w") as f:
                f.write(table)
            self.logger.info(f"Saved effect size table to {path}")

        return table

    def generate_all(
        self,
        comparative_result: ComparativeResult,
        statistical_result: StatisticalAnalysisResult,
    ) -> List[Path]:
        """
        Generate all LaTeX tables.

        Returns:
            List of generated file paths
        """
        generated = []

        self.generate_metrics_table(comparative_result)
        generated.append(self.output_dir / "metrics_table.tex")

        self.generate_statistical_table(statistical_result)
        generated.append(self.output_dir / "statistical_summary.tex")

        self.generate_effect_size_table(statistical_result)
        generated.append(self.output_dir / "effect_sizes.tex")

        return generated
