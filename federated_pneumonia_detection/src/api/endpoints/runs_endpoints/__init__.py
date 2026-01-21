"""
Runs endpoints module - aggregates all run-related API endpoints.

Provides a single router for retrieving training run results and metrics
from the database. Includes functionality for listing runs, debugging,
fetching metrics, and downloading results in various formats.
"""

from fastapi import APIRouter

from . import (
    runs_analytics,
    runs_download,
    runs_federated_rounds,
    runs_list,
    runs_metrics,
    runs_server_evaluation,
)

# Create main router for runs endpoints
router = APIRouter(prefix="/api/runs", tags=["runs", "results"])

# Include sub-routers from each module
router.include_router(runs_list.router)
router.include_router(runs_metrics.router)
router.include_router(runs_download.router)
router.include_router(runs_federated_rounds.router)
router.include_router(runs_server_evaluation.router)
router.include_router(runs_analytics.router)

__all__ = ["router"]
