"""
Download endpoints for exporting training run results.

Provides endpoints to download run results in multiple formats:
JSON, CSV, and formatted text summary reports.
"""

import json
import csv
from io import StringIO
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any

from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
from .utils import _transform_run_to_results

router = APIRouter()
logger = get_logger(__name__)


@router.get("/{run_id}/download/json")
async def download_metrics_json(run_id: int):
    """
    Download complete training metrics as JSON file.

    Args:
        run_id: Database run ID

    Returns:
        JSON file download
    """
    db = get_session()

    try:
        run = run_crud.get_with_metrics(db, run_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Get complete results
        results = _transform_run_to_results(run)

        # Convert to JSON string with pretty formatting
        json_str = json.dumps(results, indent=2, default=str)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{run_id}_metrics_{timestamp}.json"

        # Return as downloadable file
        return StreamingResponse(
            iter([json_str]),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/json"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading JSON for run {run_id}: {e}")
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
        CSV file download
    """
    db = get_session()

    try:
        run = run_crud.get_with_metrics(db, run_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Get results
        results = _transform_run_to_results(run)
        training_history = results.get("training_history", [])

        if not training_history:
            raise HTTPException(status_code=404, detail="No training history available")

        # Create CSV in memory
        output = StringIO()

        # Get all unique keys from all epochs
        all_keys = set()
        for entry in training_history:
            all_keys.update(entry.keys())

        # Sort keys for consistent column order
        fieldnames = sorted(all_keys)

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(training_history)

        # Get CSV content
        csv_content = output.getvalue()
        output.close()

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{run_id}_metrics_{timestamp}.csv"

        # Return as downloadable file
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "text/csv"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading CSV for run {run_id}: {e}")
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
        Text file with formatted summary
    """
    db = get_session()

    try:
        run = run_crud.get_with_metrics(db, run_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Get results
        results = _transform_run_to_results(run)

        # Build formatted summary report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("TRAINING RUN SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Metadata section
        metadata = results.get("metadata", {})
        report_lines.append("EXPERIMENT INFORMATION")
        report_lines.append("-" * 80)
        report_lines.append(f"Run ID:              {run_id}")
        report_lines.append(f"Experiment Name:     {metadata.get('experiment_name', 'N/A')}")
        report_lines.append(f"Status:              {results.get('status', 'N/A')}")
        report_lines.append(f"Start Time:          {metadata.get('start_time', 'N/A')}")
        report_lines.append(f"End Time:            {metadata.get('end_time', 'N/A')}")
        report_lines.append(f"Total Epochs:        {metadata.get('total_epochs', 'N/A')}")
        report_lines.append(f"Best Epoch:          {metadata.get('best_epoch', 'N/A')}")
        report_lines.append("")

        # Final metrics section
        final_metrics = results.get("final_metrics", {})
        report_lines.append("FINAL PERFORMANCE METRICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Accuracy:            {final_metrics.get('accuracy', 0):.4f} ({final_metrics.get('accuracy', 0)*100:.2f}%)")
        report_lines.append(f"Precision:           {final_metrics.get('precision', 0):.4f} ({final_metrics.get('precision', 0)*100:.2f}%)")
        report_lines.append(f"Recall:              {final_metrics.get('recall', 0):.4f} ({final_metrics.get('recall', 0)*100:.2f}%)")
        report_lines.append(f"F1 Score:            {final_metrics.get('f1_score', 0):.4f} ({final_metrics.get('f1_score', 0)*100:.2f}%)")
        report_lines.append(f"AUC-ROC:             {final_metrics.get('auc', 0):.4f} ({final_metrics.get('auc', 0)*100:.2f}%)")
        report_lines.append(f"Loss:                {final_metrics.get('loss', 0):.4f}")
        report_lines.append("")

        # Best metrics section
        report_lines.append("BEST METRICS ACROSS ALL EPOCHS")
        report_lines.append("-" * 80)
        report_lines.append(f"Best Accuracy:       {metadata.get('best_val_accuracy', 0):.4f} ({metadata.get('best_val_accuracy', 0)*100:.2f}%)")
        report_lines.append(f"Best Recall:         {metadata.get('best_val_recall', 0):.4f} ({metadata.get('best_val_recall', 0)*100:.2f}%)")
        report_lines.append(f"Best F1 Score:       {metadata.get('best_val_f1', 0):.4f} ({metadata.get('best_val_f1', 0)*100:.2f}%)")
        report_lines.append(f"Best AUC-ROC:        {metadata.get('best_val_auroc', 0):.4f} ({metadata.get('best_val_auroc', 0)*100:.2f}%)")
        report_lines.append(f"Best (Lowest) Loss:  {metadata.get('best_val_loss', 0):.4f}")
        report_lines.append("")

        # Training history summary
        training_history = results.get("training_history", [])
        if training_history:
            report_lines.append("TRAINING HISTORY SUMMARY")
            report_lines.append("-" * 80)
            report_lines.append(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Train Acc':<12} {'Val Acc':<12} {'Val F1':<12}")
            report_lines.append("-" * 80)

            for entry in training_history:
                epoch = entry.get('epoch', 1)
                train_loss = entry.get('train_loss', 0)
                val_loss = entry.get('val_loss', 0)
                train_acc = entry.get('train_acc', 0)
                val_acc = entry.get('val_acc', 0)
                val_f1 = entry.get('val_f1', 0)

                report_lines.append(
                    f"{epoch:<8} {train_loss:<12.4f} {val_loss:<12.4f} "
                    f"{train_acc*100:<11.2f}% {val_acc*100:<11.2f}% {val_f1*100:<11.2f}%"
                )

        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)

        # Join all lines
        report_content = "\n".join(report_lines)

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{run_id}_summary_{timestamp}.txt"

        # Return as downloadable file
        return StreamingResponse(
            iter([report_content]),
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating summary for run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")
    finally:
        db.close()
