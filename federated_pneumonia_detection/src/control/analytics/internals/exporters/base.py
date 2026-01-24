"""Abstract base class for data exporters."""

from abc import ABC, abstractmethod
from typing import Any, Dict


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
