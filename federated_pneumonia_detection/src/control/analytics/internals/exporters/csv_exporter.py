"""CSV exporter for training history metrics."""

import csv
from io import StringIO
from typing import Any, Dict

from .base import DataExporter


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
