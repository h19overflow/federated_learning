"""
Bootstrap analysis for robust confidence intervals.

Implements bootstrap resampling methods for non-parametric
confidence interval estimation.
"""

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

from analysis.schemas.statistics import BootstrapResult


class BootstrapAnalyzer:
    """
    Bootstrap confidence interval analyzer.

    Provides:
    - Percentile bootstrap CI
    - BCa (bias-corrected accelerated) bootstrap
    - Bootstrap hypothesis testing
    """

    def __init__(
        self,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
        random_state: int = 42,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize analyzer.

        Args:
            n_bootstrap: Number of bootstrap iterations
            confidence_level: Confidence level for intervals
            random_state: Random seed for reproducibility
            logger: Optional logger instance
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.logger = logger or logging.getLogger(__name__)
        self._rng = np.random.RandomState(random_state)

    def percentile_ci(
        self,
        data: List[float],
        statistic: Callable = np.mean,
    ) -> BootstrapResult:
        """
        Calculate percentile bootstrap confidence interval.

        Args:
            data: Original sample data
            statistic: Statistic function to compute (default: mean)

        Returns:
            BootstrapResult with CI
        """
        data_array = np.array(data)
        n = len(data_array)

        bootstrap_stats = np.zeros(self.n_bootstrap)

        for i in range(self.n_bootstrap):
            sample = self._rng.choice(data_array, size=n, replace=True)
            bootstrap_stats[i] = statistic(sample)

        alpha = 1 - self.confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)

        return BootstrapResult(
            mean=float(statistic(data_array)),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            confidence_level=self.confidence_level,
            n_bootstrap=self.n_bootstrap,
            method="percentile",
        )

    def difference_ci(
        self,
        group1: List[float],
        group2: List[float],
    ) -> BootstrapResult:
        """
        Calculate bootstrap CI for difference of means.

        Args:
            group1: First group (centralized)
            group2: Second group (federated)

        Returns:
            BootstrapResult for the difference
        """
        data1 = np.array(group1)
        data2 = np.array(group2)
        n1, n2 = len(data1), len(data2)

        observed_diff = np.mean(data2) - np.mean(data1)

        bootstrap_diffs = np.zeros(self.n_bootstrap)

        for i in range(self.n_bootstrap):
            sample1 = self._rng.choice(data1, size=n1, replace=True)
            sample2 = self._rng.choice(data2, size=n2, replace=True)
            bootstrap_diffs[i] = np.mean(sample2) - np.mean(sample1)

        alpha = 1 - self.confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
        ci_upper = np.percentile(bootstrap_diffs, upper_percentile)

        return BootstrapResult(
            mean=float(observed_diff),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            confidence_level=self.confidence_level,
            n_bootstrap=self.n_bootstrap,
            method="percentile",
        )

    def bca_ci(
        self,
        data: List[float],
        statistic: Callable = np.mean,
    ) -> BootstrapResult:
        """
        Calculate BCa (bias-corrected accelerated) bootstrap CI.

        More accurate than percentile method for skewed distributions.

        Args:
            data: Original sample data
            statistic: Statistic function to compute

        Returns:
            BootstrapResult with BCa CI
        """
        data_array = np.array(data)
        n = len(data_array)
        observed = statistic(data_array)

        bootstrap_stats = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            sample = self._rng.choice(data_array, size=n, replace=True)
            bootstrap_stats[i] = statistic(sample)

        from scipy import stats as scipy_stats

        z0 = scipy_stats.norm.ppf(np.mean(bootstrap_stats < observed))

        jackknife_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data_array, i)
            jackknife_stats[i] = statistic(jack_sample)

        jack_mean = np.mean(jackknife_stats)
        num = np.sum((jack_mean - jackknife_stats) ** 3)
        den = 6 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
        a = num / den if den != 0 else 0

        alpha = 1 - self.confidence_level
        z_alpha_lower = scipy_stats.norm.ppf(alpha / 2)
        z_alpha_upper = scipy_stats.norm.ppf(1 - alpha / 2)

        alpha1 = scipy_stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        alpha2 = scipy_stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

        ci_lower = np.percentile(bootstrap_stats, alpha1 * 100)
        ci_upper = np.percentile(bootstrap_stats, alpha2 * 100)

        return BootstrapResult(
            mean=float(observed),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            confidence_level=self.confidence_level,
            n_bootstrap=self.n_bootstrap,
            method="BCa",
        )

    def hypothesis_test(
        self,
        group1: List[float],
        group2: List[float],
        alternative: str = "two-sided",
    ) -> Tuple[float, bool]:
        """
        Bootstrap hypothesis test for difference of means.

        Args:
            group1: First group
            group2: Second group
            alternative: "two-sided", "greater", or "less"

        Returns:
            Tuple of (p-value, is_significant)
        """
        data1 = np.array(group1)
        data2 = np.array(group2)

        observed_diff = np.mean(data2) - np.mean(data1)

        pooled = np.concatenate([data1, data2])
        pooled_mean = np.mean(pooled)

        data1_centered = data1 - np.mean(data1) + pooled_mean
        data2_centered = data2 - np.mean(data2) + pooled_mean

        bootstrap_diffs = np.zeros(self.n_bootstrap)
        n1, n2 = len(data1), len(data2)

        for i in range(self.n_bootstrap):
            sample1 = self._rng.choice(data1_centered, size=n1, replace=True)
            sample2 = self._rng.choice(data2_centered, size=n2, replace=True)
            bootstrap_diffs[i] = np.mean(sample2) - np.mean(sample1)

        if alternative == "two-sided":
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        elif alternative == "greater":
            p_value = np.mean(bootstrap_diffs >= observed_diff)
        else:
            p_value = np.mean(bootstrap_diffs <= observed_diff)

        alpha = 1 - self.confidence_level
        is_significant = p_value < alpha

        return float(p_value), is_significant
