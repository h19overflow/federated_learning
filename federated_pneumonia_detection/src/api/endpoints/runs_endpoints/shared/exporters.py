"""
Export service module for training run results.

Provides abstracted exporter classes for multiple formats (JSON, CSV, Text)
and a service to orchestrate download operations.

Exports:
- DataExporter: Abstract base for all exporters
- JSONExporter: JSON serialization
- CSVExporter: CSV format with dynamic field discovery
- TextReportExporter: Formatted text summary reports
- DownloadService: Orchestrates export and file preparation
"""

import csv
import json
from abc import ABC, abstractmethod
from datetime import datetime
from io import StringIO
from typing import Any, Dict

from fastapi.responses import StreamingResponse


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


class DownloadService:
    """Orchestrates export and download file preparation."""

    @staticmethod
    def prepare_download(
        data: Dict[str, Any],
        run_id: int,
        prefix: str,
        exporter: DataExporter,
    ) -> StreamingResponse:
        """Prepare downloadable file with specified exporter."""
        content = exporter.export(data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_{run_id}_{prefix}_{timestamp}.{exporter.get_file_extension()}"
        return StreamingResponse(
            iter([content]),
            media_type=exporter.get_media_type(),
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": exporter.get_media_type(),
            },
        )
