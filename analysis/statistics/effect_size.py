"""
Effect size calculations for comparative analysis.

Computes Cohen's d and confidence intervals for quantifying
the practical significance of differences between approaches.
"""

import logging
import math
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

from analysis.schemas.statistics import EffectSizeResult


class EffectSizeCalculator:
    """
    Calculator for effect sizes and confidence intervals.

    Provides:
    - Cohen's d with interpretation
    - Confidence intervals (parametric and non-parametric)
    - Effect size magnitude classification
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize calculator.

        Args:
            confidence_level: Confidence level for intervals
            logger: Optional logger instance
        """
        self.confidence_level = confidence_level
        self.logger = logger or logging.getLogger(__name__)

    def cohens_d(
        self,
        group1: List[float],
        group2: List[float],
    ) -> EffectSizeResult:
        """
        Calculate Cohen's d effect size.

        Uses pooled standard deviation for independent samples.

        Args:
            group1: First group (centralized)
            group2: Second group (federated)

        Returns:
            EffectSizeResult with Cohen's d and confidence interval
        """
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            d = 0.0
        else:
            d = (mean2 - mean1) / pooled_std

        ci_lower, ci_upper = self._cohens_d_ci(d, n1, n2)
        magnitude = EffectSizeResult.interpret_magnitude(d)

        return EffectSizeResult(
            cohens_d=float(d),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            magnitude=magnitude,
        )

    def _cohens_d_ci(
        self,
        d: float,
        n1: int,
        n2: int,
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for Cohen's d.

        Uses non-central t-distribution approximation.

        Args:
            d: Cohen's d value
            n1: Sample size group 1
            n2: Sample size group 2

        Returns:
            Tuple of (lower, upper) bounds
        """
        se = math.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
        df = n1 + n2 - 2

        alpha = 1 - self.confidence_level
        t_crit = stats.t.ppf(1 - alpha / 2, df)

        ci_lower = d - t_crit * se
        ci_upper = d + t_crit * se

        return ci_lower, ci_upper

    def confidence_interval(
        self,
        data: List[float],
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for a single sample.

        Args:
            data: Sample data

        Returns:
            Tuple of (lower, upper) bounds
        """
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)

        alpha = 1 - self.confidence_level
        t_crit = stats.t.ppf(1 - alpha / 2, n - 1)

        margin = t_crit * se

        return float(mean - margin), float(mean + margin)

    def difference_ci(
        self,
        group1: List[float],
        group2: List[float],
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for the difference of means.

        Args:
            group1: First group (centralized)
            group2: Second group (federated)

        Returns:
            Tuple of (mean difference, lower bound, upper bound)
        """
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        mean_diff = mean2 - mean1
        se_diff = math.sqrt(var1 / n1 + var2 / n2)

        df = n1 + n2 - 2
        alpha = 1 - self.confidence_level
        t_crit = stats.t.ppf(1 - alpha / 2, df)

        margin = t_crit * se_diff

        return float(mean_diff), float(mean_diff - margin), float(mean_diff + margin)

    def interpret_effect(self, d: float) -> str:
        """
        Provide detailed interpretation of effect size.

        Args:
            d: Cohen's d value

        Returns:
            Human-readable interpretation
        """
        d_abs = abs(d)
        direction = "higher" if d > 0 else "lower"

        if d_abs < 0.2:
            size_desc = "negligible"
            practical = "no practical difference"
        elif d_abs < 0.5:
            size_desc = "small"
            practical = "minimal practical significance"
        elif d_abs < 0.8:
            size_desc = "medium"
            practical = "moderate practical significance"
        else:
            size_desc = "large"
            practical = "substantial practical significance"

        return (
            f"Cohen's d = {d:.3f} indicates a {size_desc} effect. "
            f"Federated approach is {direction}, with {practical}."
        )
