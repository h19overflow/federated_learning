"""Type definitions and service contracts for analytics.

This module consolidates all type hints and Protocol interfaces
used across analytics services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy.orm import Session


# Type definitions


@dataclass(frozen=True)
class MetricFilter:
    """Filter criteria for metric queries."""

    run_ids: list[int] | None = None
    metric_names: list[str] | None = None
    start_epoch: int | None = None
    end_epoch: int | None = None


@dataclass(frozen=True)
class RunFilter:
    """Filter criteria for run queries."""

    centralized_only: bool = False
    federated_only: bool = False
    active_only: bool = False


@dataclass(frozen=True)
class ExportFormat:
    """Export format specification."""

    FORMAT_CSV: str = "csv"
    FORMAT_JSON: str = "json"
    FORMAT_TXT: str = "txt"


# Service Protocols


class MetricsService(Protocol):
    """Service for retrieving and aggregating metrics."""

    def get_run_metrics(self, db: Session, run_id: int) -> dict[str, Any]:
        """Get all metrics for a specific run.

        Args:
            db: Database session.
            run_id: Run identifier.

        Returns:
            Dictionary containing metric data.
        """
        ...

    def get_analytics_summary(
        self, db: Session, *, filters: dict[str, Any]
    ) -> dict[str, Any]:
        """Get aggregated analytics summary.

        Args:
            db: Database session.
            filters: Optional filter criteria.

        Returns:
            Dictionary containing summary statistics.
        """
        ...

    def get_run_detail(self, db: Session, run_id: int) -> dict[str, Any]:
        """Get detailed information for a specific run.

        Args:
            db: Database session.
            run_id: Run identifier.

        Returns:
            Dictionary containing detailed run information.
        """
        ...


class SummaryService(Protocol):
    """Service for run summaries and listings."""

    def list_run_summaries(
        self,
        db: Session,
        *,
        skip: int,
        limit: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """List run summaries with pagination.

        Args:
            db: Database session.
            skip: Number of records to skip.
            limit: Maximum number of records to return.
            filters: Optional filter criteria.

        Returns:
            List of run summary dictionaries.
        """
        ...

    def get_run_summary(self, db: Session, run_id: int) -> dict[str, Any]:
        """Get summary for a specific run.

        Args:
            db: Database session.
            run_id: Run identifier.

        Returns:
            Dictionary containing run summary.
        """
        ...


class RankingService(Protocol):
    """Service for ranking and ordering runs."""

    def top_runs(
        self,
        db: Session,
        *,
        metric: str,
        n: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Get top N runs by a specific metric.

        Args:
            db: Database session.
            metric: Metric name to sort by.
            n: Number of top runs to return.
            filters: Optional filter criteria.

        Returns:
            List of run dictionaries sorted by metric.
        """
        ...


class ExportService(Protocol):
    """Service for exporting run data."""

    def export_run(
        self, db: Session, run_id: int, *, format: str
    ) -> tuple[bytes, str, str]:
        """Export run data in specified format.

        Args:
            db: Database session.
            run_id: Run identifier.
            format: Export format (csv, json, txt).

        Returns:
            Tuple of (content_bytes, media_type, filename).
        """
        ...


class BackfillService(Protocol):
    """Service for backfilling run data from external sources."""

    def backfill_from_json(
        self, db: Session, run_id: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Backfill run data from JSON payload.

        Args:
            db: Database session.
            run_id: Run identifier.
            payload: JSON payload containing run data.

        Returns:
            Dictionary containing backfill results.
        """
        ...
