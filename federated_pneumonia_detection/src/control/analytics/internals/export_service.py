"""
Export service module for training run results.

Provides abstracted exporter classes for multiple formats (JSON, CSV, Text)
and a service to orchestrate export operations with caching.

Exports:
- DataExporter: Abstract base for all exporters
- JSONExporter: JSON serialization
- CSVExporter: CSV format with dynamic field discovery
- TextReportExporter: Formatted text summary reports
- ExportService: Orchestrates export operations with caching
"""

import csv
import json
from abc import ABC, abstractmethod
from datetime import datetime
from io import StringIO
from typing import Any, Dict

from sqlalchemy.orm import Session

from .cache import CacheProvider


class DataExporter(ABC):
    """Abstract base for data exporters."""

    @abstractmethod
    def export(self, data: Dict[str, Any]) -> str:
        """Convert data to export format."""
        pass

    @abstractmethod
    def get_media_type(self) -> str:
        """Return MIME type for HTTP response."""
        pass

    @abstractmethod
    def get_file_extension(self) -> str:
        """Return file extension (e.g., 'json')."""
        pass


class JSONExporter(DataExporter):
    """Exports complete results as formatted JSON."""

    def export(self, data: Dict[str, Any]) -> str:
        """Serialize data to pretty-printed JSON."""
        return json.dumps(data, indent=2, default=str)

    def get_media_type(self) -> str:
        return "application/json"

    def get_file_extension(self) -> str:
        return "json"


class CSVExporter(DataExporter):
    """Exports training history metrics as CSV."""

    def export(self, data: Dict[str, Any]) -> str:
        """Convert training history to CSV format."""
        training_history = data.get("training_history", [])

        if not training_history:
            return ""

        # Discover all unique fields from history entries
        all_keys = set()
        for entry in training_history:
            all_keys.update(entry.keys())

        fieldnames = sorted(all_keys)

        # Write to StringIO buffer
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(training_history)

        csv_content = output.getvalue()
        output.close()

        return csv_content

    def get_media_type(self) -> str:
        return "text/csv"

    def get_file_extension(self) -> str:
        return "csv"


class TextReportExporter(DataExporter):
    """Exports formatted text summary report."""

    def export(self, data: Dict[str, Any]) -> str:
        """Generate formatted text report."""
        lines = ["=" * 80, "TRAINING RUN SUMMARY REPORT", "=" * 80, ""]
        metadata = data.get("metadata", {})
        lines += ["EXPERIMENT INFORMATION", "-" * 80]
        lines.append(f"Experiment: {metadata.get('experiment_name', 'N/A')}")
        lines.append(
            f"Status: {data.get('status', 'N/A')} | Epochs: {metadata.get('total_epochs', 'N/A')}",
        )
        final = data.get("final_metrics", {})
        lines += ["", "FINAL METRICS", "-" * 80]
        for k, v in [
            ("Acc", "accuracy"),
            ("Recall", "recall"),
            ("F1", "f1_score"),
            ("AUC", "auc"),
        ]:
            val = final.get(v, 0)
            lines.append(f"{k:<8} {val:.4f} ({val * 100:.1f}%)")
        history = data.get("training_history", [])
        if history:
            lines += ["", "HISTORY", "-" * 80]
            for e in history[:5]:
                lines.append(
                    f"E{e.get('epoch', 1):<2} | TL:{e.get('train_loss', 0):.3f} "
                    f"VL:{e.get('val_loss', 0):.3f} VA:{e.get('val_acc', 0) * 100:.1f}%",
                )
        lines += [
            "",
            "=" * 80,
            f"Report: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
        ]
        return "\n".join(lines)

    def get_media_type(self) -> str:
        return "text/plain"

    def get_file_extension(self) -> str:
        return "txt"


class ExportService:
    """Orchestrates export operations with caching support."""

    def __init__(
        self,
        *,
        cache: CacheProvider,
        run_crud_obj,
        run_metric_crud_obj,
    ):
        """Initialize ExportService with cache and CRUD objects.

        Args:
            cache: CacheProvider instance for caching computed data
            run_crud_obj: CRUD object for run queries
            run_metric_crud_obj: CRUD object for metric queries
        """
        self._cache = cache
        self._run_crud = run_crud_obj
        self._run_metric_crud = run_metric_crud_obj
        self._exporters = {
            "csv": CSVExporter(),
            "json": JSONExporter(),
            "text": TextReportExporter(),
        }

    def export_run(
        self,
        db: Session,
        run_id: int,
        *,
        format: str,
    ) -> tuple[bytes, str, str]:
        """Export a training run in the specified format.

        Args:
            db: Database session
            run_id: ID of the run to export
            format: Export format ('csv', 'json', or 'text')

        Returns:
            Tuple of (content_bytes, media_type, filename)

        Raises:
            ValueError: If format is not supported or run not found
        """
        if format not in self._exporters:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported: {', '.join(self._exporters.keys())}"
            )

        # Get or compute run data (cache the dict, not the bytes)
        cache_key = f"run_export_data:{run_id}"
        run_data = self._cache.get(cache_key)

        if run_data is None:
            run_data = self._build_run_data(db, run_id)
            self._cache.set(cache_key, run_data)

        # Export using appropriate exporter
        exporter = self._exporters[format]
        content_str = exporter.export(run_data)
        content_bytes = content_str.encode("utf-8")

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{run_id}_export_{timestamp}.{exporter.get_file_extension()}"

        return content_bytes, exporter.get_media_type(), filename

    def _build_run_data(self, db: Session, run_id: int) -> Dict[str, Any]:
        """Build complete run data dictionary from database.

        Args:
            db: Database session
            run_id: ID of the run

        Returns:
            Dictionary containing run metadata, metrics, and history

        Raises:
            ValueError: If run not found
        """
        # Fetch run record
        run = self._run_crud.get(db, run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        # Fetch metrics for this run
        metrics = self._run_metric_crud.get_by_run_id(db, run_id)

        # Build training history from metrics
        training_history = []
        if metrics:
            for metric in metrics:
                history_entry = {
                    "epoch": getattr(metric, "epoch", 0),
                    "train_loss": getattr(metric, "train_loss", 0.0),
                    "val_loss": getattr(metric, "val_loss", 0.0),
                    "train_acc": getattr(metric, "train_acc", 0.0),
                    "val_acc": getattr(metric, "val_acc", 0.0),
                    "recall": getattr(metric, "recall", 0.0),
                    "f1_score": getattr(metric, "f1_score", 0.0),
                    "auc": getattr(metric, "auc", 0.0),
                }
                training_history.append(history_entry)

        # Extract final metrics (last entry or defaults)
        final_metrics = training_history[-1] if training_history else {}

        # Build complete data structure
        run_data = {
            "run_id": run_id,
            "status": getattr(run, "status", "unknown"),
            "metadata": {
                "experiment_name": getattr(run, "experiment_name", "N/A"),
                "total_epochs": getattr(run, "total_epochs", 0),
                "created_at": str(getattr(run, "created_at", "")),
                "completed_at": str(getattr(run, "completed_at", "")),
            },
            "final_metrics": {
                "accuracy": final_metrics.get("train_acc", 0.0),
                "recall": final_metrics.get("recall", 0.0),
                "f1_score": final_metrics.get("f1_score", 0.0),
                "auc": final_metrics.get("auc", 0.0),
            },
            "training_history": training_history,
        }

        return run_data
