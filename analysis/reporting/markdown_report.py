"""
Markdown report generation for analysis results.

Creates comprehensive analysis reports in Markdown format
with tables, figures, and statistical interpretations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from analysis.schemas.comparison import ComparativeResult
from analysis.schemas.statistics import StatisticalAnalysisResult


class MarkdownReportGenerator:
    """
    Generates Markdown analysis reports.

    Creates comprehensive reports including:
    - Executive summary
    - Methodology description
    - Results tables
    - Statistical analysis
    - Conclusions
    """

    def __init__(
        self,
        output_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize generator.

        Args:
            output_dir: Directory to save report
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        comparative_result: ComparativeResult,
        statistical_result: StatisticalAnalysisResult,
        data_profile: Optional[Dict] = None,
        figures_dir: Optional[Path] = None,
        save: bool = True,
    ) -> str:
        """
        Generate complete analysis report.

        Args:
            comparative_result: Comparison results
            statistical_result: Statistical analysis results
            data_profile: Optional data profile information
            figures_dir: Path to generated figures
            save: Whether to save to file

        Returns:
            Markdown report string
        """
        sections = []

        sections.append(self._generate_header())
        sections.append(self._generate_executive_summary(comparative_result, statistical_result))
        sections.append(self._generate_methodology(comparative_result))

        if data_profile:
            sections.append(self._generate_data_section(data_profile))

        sections.append(self._generate_results_section(comparative_result))
        sections.append(self._generate_statistical_section(statistical_result))
        sections.append(self._generate_conclusions(comparative_result, statistical_result))

        if figures_dir:
            sections.append(self._generate_figures_section(figures_dir))

        report = "\n\n".join(sections)

        if save:
            path = self.output_dir / "analysis_report.md"
            with open(path, "w") as f:
                f.write(report)
            self.logger.info(f"Saved report to {path}")

        return report

    def _generate_header(self) -> str:
        """Generate report header."""
        return f"""# Comparative Analysis: Federated vs Centralized Learning

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---"""

    def _generate_executive_summary(
        self,
        comparative_result: ComparativeResult,
        statistical_result: StatisticalAnalysisResult,
    ) -> str:
        """Generate executive summary."""
        n_cent = len(comparative_result.centralized_runs)
        n_fed = len(comparative_result.federated_runs)

        significant_metrics = statistical_result.significant_metrics()
        n_significant = len(significant_metrics)
        n_total = len(statistical_result.metrics)

        acc_cent = comparative_result.centralized_aggregated.metrics.get("accuracy")
        acc_fed = comparative_result.federated_aggregated.metrics.get("accuracy")

        summary = f"""## Executive Summary

This report presents a comparative analysis between **centralized** and **federated**
learning approaches for pneumonia detection from chest X-rays.

### Key Findings

- **Experiments:** {n_cent} centralized runs, {n_fed} federated runs
- **Significant Differences:** {n_significant} of {n_total} metrics show significant difference (p < 0.05)
"""

        if acc_cent and acc_fed:
            diff = acc_fed.mean - acc_cent.mean
            direction = "higher" if diff > 0 else "lower"
            summary += f"""- **Accuracy:** Centralized {acc_cent.mean:.4f} ± {acc_cent.std:.4f} vs Federated {acc_fed.mean:.4f} ± {acc_fed.std:.4f}
- **Finding:** Federated approach shows {abs(diff):.4f} {direction} accuracy
"""

        summary += f"\n**Conclusion:** {statistical_result.overall_conclusion}"

        return summary

    def _generate_methodology(self, comparative_result: ComparativeResult) -> str:
        """Generate methodology section."""
        cent_run = comparative_result.centralized_runs[0] if comparative_result.centralized_runs else None
        fed_run = comparative_result.federated_runs[0] if comparative_result.federated_runs else None

        section = """## Methodology

### Experimental Design

- **Comparison Approach:** Sequential execution (centralized first, then federated)
- **Statistical Validity:** 5 runs per approach with different random seeds
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, AUROC

### Centralized Training Configuration
"""
        if cent_run and cent_run.config_snapshot:
            for key, value in cent_run.config_snapshot.items():
                section += f"- **{key.title()}:** {value}\n"

        section += "\n### Federated Training Configuration\n"
        if fed_run and fed_run.config_snapshot:
            for key, value in fed_run.config_snapshot.items():
                section += f"- **{key.title()}:** {value}\n"

        return section

    def _generate_data_section(self, data_profile: Dict) -> str:
        """Generate data description section."""
        overview = data_profile.get("dataset_overview", {})
        class_dist = data_profile.get("class_distribution", {})

        section = f"""## Dataset Description

### Overview
- **Total Samples:** {overview.get('total_samples', 'N/A'):,}
- **Number of Classes:** {class_dist.get('num_classes', 'N/A')}
- **Class Balance Ratio:** {class_dist.get('balance_ratio', 'N/A'):.4f}

### Class Distribution
"""
        classes = class_dist.get("classes", {})
        for label, info in classes.items():
            section += f"- **Class {label}:** {info.get('count', 0):,} ({info.get('percentage', 0):.1f}%)\n"

        return section

    def _generate_results_section(self, comparative_result: ComparativeResult) -> str:
        """Generate results section with tables."""
        metrics = list(comparative_result.centralized_aggregated.metrics.keys())

        section = """## Results

### Performance Comparison

| Metric | Centralized | Federated | Difference | % Change |
|--------|-------------|-----------|------------|----------|
"""
        for metric in metrics:
            comparison = comparative_result.get_metric_comparison(metric)
            section += (
                f"| {metric.title()} | "
                f"{comparison['centralized_mean']:.4f} ± {comparison['centralized_std']:.4f} | "
                f"{comparison['federated_mean']:.4f} ± {comparison['federated_std']:.4f} | "
                f"{comparison['difference']:+.4f} | "
                f"{comparison['percent_difference']:+.2f}% |\n"
            )

        return section

    def _generate_statistical_section(self, statistical_result: StatisticalAnalysisResult) -> str:
        """Generate statistical analysis section."""
        section = """## Statistical Analysis

### Hypothesis Testing

| Metric | Test Used | p-value | Significant | Cohen's d | Effect |
|--------|-----------|---------|-------------|-----------|--------|
"""
        for metric_name, result in statistical_result.metrics.items():
            sig = "**Yes**" if result.comparison_test.is_significant else "No"
            p_str = f"{result.comparison_test.p_value:.4f}" if result.comparison_test.p_value >= 0.0001 else "<0.0001"

            section += (
                f"| {metric_name.title()} | "
                f"{result.comparison_test.test_name} | "
                f"{p_str} | "
                f"{sig} | "
                f"{result.effect_size.cohens_d:.3f} | "
                f"{result.effect_size.magnitude.title()} |\n"
            )

        section += f"\n*Note: Significance level α = {statistical_result.alpha}*"

        return section

    def _generate_conclusions(
        self,
        comparative_result: ComparativeResult,
        statistical_result: StatisticalAnalysisResult,
    ) -> str:
        """Generate conclusions section."""
        significant = statistical_result.significant_metrics()

        section = """## Conclusions

### Summary of Findings

"""
        if not significant:
            section += """The analysis reveals **no statistically significant differences** between
centralized and federated learning approaches across all evaluated metrics.
This suggests that federated learning can achieve comparable performance to
centralized learning while providing the privacy benefits of decentralized training.
"""
        else:
            section += f"""The analysis reveals **statistically significant differences** in the following
metrics: {', '.join(m.title() for m in significant)}.

"""
            for metric in significant:
                result = statistical_result.metrics[metric]
                section += f"- **{metric.title()}:** {result.comparison_test.interpretation}\n"

        section += """
### Implications

1. **Privacy-Performance Trade-off:** Results indicate the feasibility of using
   federated learning for medical image analysis with minimal performance degradation.

2. **Practical Considerations:** The observed differences (if any) should be
   weighed against the privacy benefits of federated learning.

3. **Future Work:** Further investigation with larger datasets and more
   federated clients is recommended.
"""

        return section

    def _generate_figures_section(self, figures_dir: Path) -> str:
        """Generate figures reference section."""
        section = """## Figures

The following visualizations are available in the output directory:

"""
        if figures_dir.exists():
            for fig_path in sorted(figures_dir.glob("*.png")):
                section += f"- `{fig_path.name}`\n"

        return section
