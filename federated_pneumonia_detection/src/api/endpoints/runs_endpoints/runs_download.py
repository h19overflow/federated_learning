"""
Download endpoints for exporting training run results.

Provides endpoints to download run results in multiple formats (JSON, CSV, text).
Uses shared exporter modules and download service following SOLID principles.
"""

from fastapi import APIRouter, HTTPException

from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

from .utils import _transform_run_to_results
from .shared.exporters import (
    JSONExporter,
    CSVExporter,
    TextReportExporter,
    DownloadService
)

router = APIRouter()
logger = get_logger(__name__)


@router.get("/{run_id}/download/json")
async def download_metrics_json(run_id: int):
    """
    Download complete training metrics as JSON file.

    Args:
        run_id: Database run ID

    Returns:
        Streaming JSON file download
    """
    db = get_session()

    try:
        run = run_crud.get_with_metrics(db, run_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        results = _transform_run_to_results(run)
        return DownloadService.prepare_download(
            data=results,
            run_id=run_id,
            prefix="metrics",
            exporter=JSONExporter()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading JSON for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate JSON: {str(e)}")
    finally:
        db.close()


@router.get("/{run_id}/download/csv")
async def download_metrics_csv(run_id: int):
    """
    Download training history metrics as CSV file.

    Args:
        run_id: Database run ID

    Returns:
        Streaming CSV file download
    """
    db = get_session()

    try:
        run = run_crud.get_with_metrics(db, run_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        results = _transform_run_to_results(run)

        if not results.get("training_history"):
            raise HTTPException(status_code=404, detail="No training history available")

        return DownloadService.prepare_download(
            data=results,
            run_id=run_id,
            prefix="metrics",
            exporter=CSVExporter()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading CSV for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate CSV: {str(e)}")
    finally:
        db.close()


@router.get("/{run_id}/download/summary")
async def download_summary_report(run_id: int):
    """
    Download formatted summary report as text file.

    Args:
        run_id: Database run ID

    Returns:
        Streaming text file download
    """
    db = get_session()

    try:
        run = run_crud.get_with_metrics(db, run_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        results = _transform_run_to_results(run)
        return DownloadService.prepare_download(
            data=results,
            run_id=run_id,
            prefix="summary",
            exporter=TextReportExporter()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating summary for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")
    finally:
        db.close()
