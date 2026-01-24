"""JSON exporter for training run results."""

import json
from typing import Any, Dict

from .base import DataExporter


class JSONExporter(DataExporter):
    """Exports complete results as formatted JSON."""

    def export(self, data: Dict[str, Any]) -> str:
        """Serialize data to pretty-printed JSON."""
        return json.dumps(data, indent=2, default=str)

    def get_media_type(self) -> str:
        return "application/json"

    def get_file_extension(self) -> str:
        return "json"
