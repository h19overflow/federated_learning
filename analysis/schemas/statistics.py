"""
Pydantic schemas for statistical test results.

Defines structures for normality tests, comparison tests, effect sizes, and bootstrap analysis.
"""

from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field


class NormalityTestResult(BaseModel):
    """Result from Shapiro-Wilk normality test."""

    test_name: str = "Shapiro-Wilk"
    statistic: float
    p_value: float
    is_normal: bool = Field(description="True if p > alpha (data appears normal)")
    sample_size: int
    group: str = Field(description="Group tested: centralized or federated")


class PairedTestResult(BaseModel):
    """Result from paired comparison test (t-test or Wilcoxon)."""

    test_name: str = Field(description="Test used: paired t-test or Wilcoxon signed-rank")
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float = 0.05
    interpretation: str = Field(description="Human-readable interpretation")


class EffectSizeResult(BaseModel):
    """Cohen's d effect size with confidence interval."""

    cohens_d: float
    ci_lower: float
    ci_upper: float
    magnitude: Literal["negligible", "small", "medium", "large"] = Field(
        description="Effect size interpretation"
    )

    @classmethod
    def interpret_magnitude(cls, d: float) -> str:
        """Interpret Cohen's d magnitude."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"


class BootstrapResult(BaseModel):
    """Bootstrap confidence interval result."""

    mean: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    n_bootstrap: int = 10000
    method: str = "percentile"


class ANOVAResult(BaseModel):
    """One-way ANOVA result for multi-run comparison."""

    f_statistic: float
    p_value: float
    is_significant: bool
    df_between: int
    df_within: int
    interpretation: str


class MetricStatisticalResult(BaseModel):
    """Complete statistical analysis for a single metric."""

    metric_name: str
    normality_centralized: NormalityTestResult
    normality_federated: NormalityTestResult
    comparison_test: PairedTestResult
    effect_size: EffectSizeResult
    bootstrap_difference: BootstrapResult
    anova: Optional[ANOVAResult] = None


class StatisticalAnalysisResult(BaseModel):
    """Complete statistical analysis across all metrics."""

    metrics: Dict[str, MetricStatisticalResult] = Field(
        description="Statistical results per metric"
    )
    alpha: float = 0.05
    confidence_level: float = 0.95
    n_bootstrap: int = 10000
    overall_conclusion: str = Field(
        description="Summary conclusion about centralized vs federated"
    )

    def significant_metrics(self) -> List[str]:
        """Return list of metrics with significant differences."""
        return [
            name
            for name, result in self.metrics.items()
            if result.comparison_test.is_significant
        ]

    def to_summary_table(self) -> List[Dict]:
        """Generate summary table data for all metrics."""
        rows = []
        for name, result in self.metrics.items():
            rows.append(
                {
                    "metric": name,
                    "test": result.comparison_test.test_name,
                    "p_value": result.comparison_test.p_value,
                    "significant": result.comparison_test.is_significant,
                    "cohens_d": result.effect_size.cohens_d,
                    "effect_magnitude": result.effect_size.magnitude,
                }
            )
        return rows
