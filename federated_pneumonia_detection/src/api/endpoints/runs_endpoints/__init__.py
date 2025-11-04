"""
Runs endpoints module - aggregates all run-related API endpoints.

Provides a single router for retrieving training run results and metrics
from the database. Includes functionality for listing runs, debugging,
fetching metrics, and downloading results in various formats.
"""

from fastapi import APIRouter
from . import runs_list, runs_debug, runs_metrics, runs_download, runs_federated_rounds

# Create main router for runs endpoints
router = APIRouter(prefix="/api/runs", tags=["runs", "results"])

# Include sub-routers from each module
router.include_router(runs_list.router)
router.include_router(runs_debug.router)
router.include_router(runs_metrics.router)
router.include_router(runs_download.router)
router.include_router(runs_federated_rounds.router)

__all__ = ["router"]
