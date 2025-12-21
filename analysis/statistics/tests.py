"""
Statistical tests for comparing centralized vs federated learning.

Implements publication-ready statistical tests including normality tests,
paired comparisons, and ANOVA for multi-run analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from analysis.schemas.statistics import (
    ANOVAResult,
    NormalityTestResult,
    PairedTestResult,
)


class StatisticalTests:
    """
    Statistical test suite for comparative analysis.

    Provides:
    - Shapiro-Wilk normality tests
    - Paired t-test (parametric)
    - Wilcoxon signed-rank test (non-parametric)
    - One-way ANOVA
    """

    def __init__(
        self,
        alpha: float = 0.05,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize test suite.

        Args:
            alpha: Significance level
            logger: Optional logger instance
        """
        self.alpha = alpha
        self.logger = logger or logging.getLogger(__name__)

    def shapiro_wilk(
        self,
        data: List[float],
        group_name: str,
    ) -> NormalityTestResult:
        """
        Perform Shapiro-Wilk normality test.

        Args:
            data: Sample data
            group_name: Name of group (centralized/federated)

        Returns:
            NormalityTestResult
        """
        if len(data) < 3:
            return NormalityTestResult(
                statistic=0.0,
                p_value=1.0,
                is_normal=True,
                sample_size=len(data),
                group=group_name,
            )

        statistic, p_value = stats.shapiro(data)
        is_normal = p_value > self.alpha

        return NormalityTestResult(
            statistic=float(statistic),
            p_value=float(p_value),
            is_normal=is_normal,
            sample_size=len(data),
            group=group_name,
        )

    def paired_ttest(
        self,
        centralized: List[float],
        federated: List[float],
        metric_name: str,
    ) -> PairedTestResult:
        """
        Perform paired t-test for dependent samples.

        Args:
            centralized: Centralized results
            federated: Federated results
            metric_name: Name of metric being tested

        Returns:
            PairedTestResult
        """
        if len(centralized) != len(federated):
            raise ValueError("Sample sizes must match for paired test")

        statistic, p_value = stats.ttest_rel(centralized, federated)
        is_significant = p_value < self.alpha

        cent_mean = np.mean(centralized)
        fed_mean = np.mean(federated)
        diff = fed_mean - cent_mean

        interpretation = self._interpret_comparison(
            is_significant, diff, metric_name, "paired t-test"
        )

        return PairedTestResult(
            test_name="paired t-test",
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            alpha=self.alpha,
            interpretation=interpretation,
        )

    def wilcoxon(
        self,
        centralized: List[float],
        federated: List[float],
        metric_name: str,
    ) -> PairedTestResult:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative).

        Args:
            centralized: Centralized results
            federated: Federated results
            metric_name: Name of metric being tested

        Returns:
            PairedTestResult
        """
        if len(centralized) != len(federated):
            raise ValueError("Sample sizes must match for paired test")

        try:
            statistic, p_value = stats.wilcoxon(centralized, federated)
        except ValueError:
            statistic, p_value = 0.0, 1.0

        is_significant = p_value < self.alpha

        cent_mean = np.mean(centralized)
        fed_mean = np.mean(federated)
        diff = fed_mean - cent_mean

        interpretation = self._interpret_comparison(
            is_significant, diff, metric_name, "Wilcoxon signed-rank"
        )

        return PairedTestResult(
            test_name="Wilcoxon signed-rank",
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            alpha=self.alpha,
            interpretation=interpretation,
        )

    def one_way_anova(
        self,
        centralized: List[float],
        federated: List[float],
        metric_name: str,
    ) -> ANOVAResult:
        """
        Perform one-way ANOVA comparing training approaches.

        Args:
            centralized: Centralized results
            federated: Federated results
            metric_name: Name of metric being tested

        Returns:
            ANOVAResult
        """
        f_statistic, p_value = stats.f_oneway(centralized, federated)
        is_significant = p_value < self.alpha

        df_between = 1
        df_within = len(centralized) + len(federated) - 2

        if is_significant:
            interpretation = (
                f"Significant difference in {metric_name} between approaches "
                f"(F={f_statistic:.3f}, p={p_value:.4f})"
            )
        else:
            interpretation = (
                f"No significant difference in {metric_name} between approaches "
                f"(F={f_statistic:.3f}, p={p_value:.4f})"
            )

        return ANOVAResult(
            f_statistic=float(f_statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            df_between=df_between,
            df_within=df_within,
            interpretation=interpretation,
        )

    def select_appropriate_test(
        self,
        centralized: List[float],
        federated: List[float],
        metric_name: str,
    ) -> Tuple[PairedTestResult, str]:
        """
        Select and run appropriate comparison test based on normality.

        Args:
            centralized: Centralized results
            federated: Federated results
            metric_name: Name of metric

        Returns:
            Tuple of (test result, rationale for test selection)
        """
        norm_cent = self.shapiro_wilk(centralized, "centralized")
        norm_fed = self.shapiro_wilk(federated, "federated")

        both_normal = norm_cent.is_normal and norm_fed.is_normal

        if both_normal:
            result = self.paired_ttest(centralized, federated, metric_name)
            rationale = "Paired t-test selected: both groups pass normality test"
        else:
            result = self.wilcoxon(centralized, federated, metric_name)
            rationale = "Wilcoxon test selected: normality assumption violated"

        return result, rationale

    def _interpret_comparison(
        self,
        is_significant: bool,
        difference: float,
        metric_name: str,
        test_name: str,
    ) -> str:
        """Generate human-readable interpretation."""
        if not is_significant:
            return (
                f"No statistically significant difference in {metric_name} "
                f"between centralized and federated learning ({test_name})."
            )

        direction = "higher" if difference > 0 else "lower"
        return (
            f"Federated learning shows significantly {direction} {metric_name} "
            f"compared to centralized learning ({test_name}, diff={difference:+.4f})."
        )
