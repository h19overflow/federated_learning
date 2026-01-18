"""Report generation endpoints package."""

from federated_pneumonia_detection.src.api.endpoints.reports.report_endpoints import (
    router as report_router,
)

__all__ = ["report_router"]
