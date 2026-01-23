"""Download endpoints for exporting training run results."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.api.deps import get_db
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

from .shared.exporters import CSVExporter, DownloadService
from .shared.utils import _transform_run_to_results

router = APIRouter()
logger = get_logger(__name__)


@router.get("/{run_id}/download/csv")
async def download_metrics_csv(
    run_id: int,
    db: Session = Depends(get_db),
):
    """
    Download training history metrics as CSV file.

    Args:
        run_id: Database run ID

    Returns:
        Streaming CSV file download
    """
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
            exporter=CSVExporter(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading CSV for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate CSV: {str(e)}")
