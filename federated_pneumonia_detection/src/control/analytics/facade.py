"""Analytics Facade - Unified interface for all analytics services.

Provides a single entry point for endpoints to access analytics business logic
through dependency injection. Services are initialized at startup and cached
in app.state for performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

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
        """Initialize facade with analytics services (all optional for partial initialization).  # noqa: E501

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

    def get_run_summary(self, session: Session, run_id: int) -> Any:
        """Retrieve and process run summary data.

        Delegates to SummaryService and returns a result object matching
        the expected analytics format.
        """
        if not self.summary:
            raise ValueError("Summary service not available")

        raw_summary = self.summary.get_run_summary(session, run_id)

        # Transform to match the expected test format:
        # result.metrics['best_accuracy'], result.status, result.mode
        class Result:
            def __init__(self, data):
                self.metrics = {"best_accuracy": data.get("best_val_accuracy", 0.0)}
                self.status = data.get("status")
                self.mode = data.get("training_mode")

        return Result(raw_summary)
