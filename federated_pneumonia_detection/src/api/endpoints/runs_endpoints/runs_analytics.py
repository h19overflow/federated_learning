"""
Analytics endpoints for aggregated training run statistics.

Provides endpoints for comparing centralized vs federated training performance
with aggregated metrics and top-performing runs.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
from .analytics_utils import (
    calculate_mode_statistics,
    extract_run_details,
    create_empty_response
)

router = APIRouter()
logger = get_logger(__name__)


@router.get("/analytics/summary")
async def get_analytics_summary(
    status: Optional[str] = Query("completed", description="Filter by run status"),
    training_mode: Optional[str] = Query(None, description="Filter by training mode"),
    days: Optional[int] = Query(None, description="Filter by last N days")
) -> Dict[str, Any]:
    """
    Get aggregated statistics for centralized vs federated training comparison.

    Args:
        status: Filter by run status (default: "completed")
        training_mode: Filter by mode ("centralized", "federated", or None for all)
        days: Filter by last N days (None for all time)

    Returns:
        Dictionary with aggregated statistics and top runs

    Response format:
        {
            "total_runs": int,
            "success_rate": float,
            "centralized": {...},
            "federated": {...},
            "top_runs": [...]
        }
    """
    db = get_session()

    try:
        # Fetch runs with status filter
        runs = run_crud.get_by_status(db, status)

        # Apply training_mode filter if specified
        if training_mode:
            runs = [r for r in runs if r.training_mode == training_mode]

        # Apply time filter if specified
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            runs = [r for r in runs if r.start_time and r.start_time >= cutoff_date]

        if not runs:
            logger.warning(
                f"No runs found with filters: status={status}, "
                f"mode={training_mode}, days={days}"
            )
            return create_empty_response()

        # Split runs by training mode
        centralized_runs = [r for r in runs if r.training_mode == "centralized"]
        federated_runs = [r for r in runs if r.training_mode == "federated"]

        # Calculate aggregated statistics
        centralized_stats = calculate_mode_statistics(db, centralized_runs)
        federated_stats = calculate_mode_statistics(db, federated_runs)

        # Extract details for all runs
        all_run_details = []
        for run in runs:
            run_detail = extract_run_details(db, run)
            if run_detail:
                all_run_details.append(run_detail)

        # Sort by best_accuracy descending and take top 10
        top_runs = sorted(
            all_run_details,
            key=lambda x: x.get("best_accuracy", 0.0),
            reverse=True
        )[:10]

        # Calculate success rate
        total_runs = len(runs)
        all_status_runs = run_crud.get_by_status(db, status)
        success_rate = total_runs / len(all_status_runs) if len(all_status_runs) > 0 else 0.0

        logger.info(
            f"Analytics summary generated: {total_runs} runs "
            f"({len(centralized_runs)} centralized, {len(federated_runs)} federated)"
        )

        return {
            "total_runs": total_runs,
            "success_rate": round(success_rate, 4),
            "centralized": centralized_stats,
            "federated": federated_stats,
            "top_runs": top_runs
        }

    except Exception as e:
        logger.error(f"Error generating analytics summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
