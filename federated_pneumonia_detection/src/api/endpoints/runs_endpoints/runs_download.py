"""Download endpoints for exporting training run results."""

import io
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.api.deps import get_analytics, get_db
from federated_pneumonia_detection.src.control.analytics.facade import AnalyticsFacade
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/{run_id}/download/csv")
async def download_metrics_csv(
    run_id: int,
    db: Session = Depends(get_db),
    analytics: Optional[AnalyticsFacade] = Depends(get_analytics),
):
    """
    Download training history metrics as CSV file.

    Args:
        run_id: Database run ID
        analytics: AnalyticsFacade dependency (optional)

    Returns:
        Streaming CSV file download
    """
    if analytics is None:
        logger.warning("Analytics service not available for download endpoint")
        raise HTTPException(
            status_code=503,
            detail="Analytics service unavailable. Please check server logs.",
        )

    try:
        content, media_type, filename = analytics.export.export_run(
            db, run_id, format="csv"
        )
        return StreamingResponse(
            io.BytesIO(content),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading CSV for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate CSV: {str(e)}")
