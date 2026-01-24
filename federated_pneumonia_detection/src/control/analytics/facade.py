"""Analytics Facade - Unified interface for all analytics services.

Provides a single entry point for endpoints to access analytics business logic
through dependency injection. Services are initialized at startup and cached
in app.state for performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from federated_pneumonia_detection.src.control.analytics.internals import (
        BackfillService,
        ExportService,
        MetricsService,
        RankingService,
        SummaryService,
    )


class AnalyticsFacade:
    """Facade providing unified access to analytics services.

    Orchestrates multiple analytics services (metrics, summaries, rankings, etc.)
    and provides a clean interface for API endpoints via dependency injection.

    All services are initialized once at startup with shared caching to optimize
    performance for read-heavy operations like the SavedExperiments page.
    """

    def __init__(
        self,
        *,
        metrics: Optional[MetricsService] = None,
        summary: Optional[SummaryService] = None,
        ranking: Optional[RankingService] = None,
        export: Optional[ExportService] = None,
        backfill: Optional[BackfillService] = None,
    ):
        """Initialize facade with analytics services (all optional for partial initialization).

        Args:
            metrics: MetricsService for metric extraction and aggregation.
            summary: SummaryService for run summaries.
            ranking: RankingService for top-N rankings.
            export: ExportService for data exports.
            backfill: BackfillService for JSON backfill operations.

        Note:
            Services can be None if initialization failed at startup. The facade
            will still function with available services, and endpoints should handle
            None values gracefully by falling back to direct CRUD operations.
        """
        self.metrics = metrics
        self.summary = summary
        self.ranking = ranking
        self.export = export
        self.backfill = backfill
