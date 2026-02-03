import logging

from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    server_evaluation_crud,
)
from federated_pneumonia_detection.src.control.analytics import AnalyticsFacade
from federated_pneumonia_detection.src.control.analytics.internals import (
    BackfillService,
    CacheProvider,
    ExportService,
    MetricsService,
    RankingService,
    SummaryService,
)

logger = logging.getLogger(__name__)

def initialize_analytics_services(app) -> AnalyticsFacade | None:
    """
    Initialize analytics services at startup for caching and performance.

    Creates CacheProvider (TTL=600s) and all 5 analytics services.
    Stored in app.state.analytics for endpoint access via dependency injection.

    Args:
        app: FastAPI app instance for storing services in app.state

    Returns:
        AnalyticsFacade instance or None if initialization fails catastrophically.
        Returns a partial facade if some services fail initialization.
    """
    logger.info("[Startup] Initializing analytics services...")

    # Track which services initialized successfully
    services_status: dict[str, bool] = {}
    cache: CacheProvider | None = None
    metrics_service: MetricsService | None = None
    summary_service: SummaryService | None = None
    ranking_service: RankingService | None = None
    export_service: ExportService | None = None
    backfill_service: BackfillService | None = None

    # CacheProvider is critical - if this fails, entire analytics subsystem fails
    try:
        cache = CacheProvider(ttl=600, maxsize=1000)
        logger.info("[Startup/Analytics] CacheProvider initialized successfully")
        services_status["cache"] = True
    except Exception as e:
        logger.error(f"[Startup/Analytics] CacheProvider initialization failed: {e}")
        logger.warning(
            "[Startup/Analytics] Cannot initialize analytics services without cache provider"  # noqa: E501
        )
        return None

    # MetricsService - optional
    try:
        metrics_service = MetricsService(
            cache=cache,
            run_crud_obj=run_crud,
            run_metric_crud_obj=run_metric_crud,
            server_evaluation_crud_obj=server_evaluation_crud,
        )
        logger.info("[Startup/Analytics] MetricsService initialized successfully")
        services_status["metrics"] = True
    except Exception as e:
        logger.warning(
            f"[Startup/Analytics] MetricsService initialization failed: {e} (metrics endpoints will be degraded)"  # noqa: E501
        )
        services_status["metrics"] = False

    # SummaryService - optional
    try:
        summary_service = SummaryService(
            cache=cache,
            run_crud_obj=run_crud,
            run_metric_crud_obj=run_metric_crud,
            server_evaluation_crud_obj=server_evaluation_crud,
        )
        logger.info("[Startup/Analytics] SummaryService initialized successfully")
        services_status["summary"] = True
    except Exception as e:
        logger.warning(
            f"[Startup/Analytics] SummaryService initialization failed: {e} (summary endpoints will be degraded)"  # noqa: E501
        )
        services_status["summary"] = False

    # RankingService - optional
    try:
        ranking_service = RankingService(
            cache=cache,
            run_crud_obj=run_crud,
        )
        logger.info("[Startup/Analytics] RankingService initialized successfully")
        services_status["ranking"] = True
    except Exception as e:
        logger.warning(
            f"[Startup/Analytics] RankingService initialization failed: {e} (ranking endpoints will be degraded)"  # noqa: E501
        )
        services_status["ranking"] = False

    # ExportService - optional
    try:
        export_service = ExportService(
            cache=cache,
            run_crud_obj=run_crud,
            run_metric_crud_obj=run_metric_crud,
        )
        logger.info("[Startup/Analytics] ExportService initialized successfully")
        services_status["export"] = True
    except Exception as e:
        logger.warning(
            f"[Startup/Analytics] ExportService initialization failed: {e} (export endpoints will be degraded)"  # noqa: E501
        )
        services_status["export"] = False

    # BackfillService - optional
    try:
        backfill_service = BackfillService(
            run_crud_obj=run_crud,
            server_evaluation_crud_obj=server_evaluation_crud,
        )
        logger.info("[Startup/Analytics] BackfillService initialized successfully")
        services_status["backfill"] = True
    except Exception as e:
        logger.warning(
            f"[Startup/Analytics] BackfillService initialization failed: {e} (backfill endpoints will be degraded)"  # noqa: E501
        )
        services_status["backfill"] = False

    # Create facade with available services (at least cache must be working)
    try:
        facade = AnalyticsFacade(
            metrics=metrics_service,
            summary=summary_service,
            ranking=ranking_service,
            export=export_service,
            backfill=backfill_service,
        )
        logger.info(
            f"[Startup/Analytics] Analytics facade created with "
            f"{sum(1 for v in services_status.values() if v)}/{len(services_status)} services available"  # noqa: E501
        )

        # Log unavailable services for visibility
        unavailable = [
            name
            for name, available in services_status.items()
            if not available and name != "cache"
        ]
        if unavailable:
            logger.warning(
                f"[Startup/Analytics] Unavailable services: {', '.join(unavailable)}. "
                "Endpoints using these services will fall back to direct CRUD calls."
            )

        # Store in app.state
        if app is not None:
            app.state.analytics = facade
            logger.info("[Startup] Analytics services initialized and cached")

        return facade

    except Exception as e:
        logger.error(f"[Startup/Analytics] Failed to create analytics facade: {e}")
        logger.warning("[Startup/Analytics] Analytics subsystem will be unavailable")
        return None
