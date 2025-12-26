"""Nested dictionary access operations using dot notation.

Provides utilities for getting and setting values in nested dictionaries
using dot-separated key paths (e.g., 'experiment.learning_rate').
"""

from typing import Any, Dict
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger


class NestedAccessor:
    """Handles get/set operations on nested dictionaries with dot notation."""

    def __init__(self):
        """Initialize the accessor with logger."""
        self.logger = get_logger(__name__)

    def get(self, config: Dict[str, Any], key_path: str) -> Any:
        """
        Get a nested value from config using dot notation.

        Args:
            config: The configuration dictionary
            key_path: Dot-separated key path (e.g., 'experiment.learning_rate')

        Returns:
            The value at the specified path

        Raises:
            KeyError: If the key path doesn't exist
        """
        keys = key_path.split(".")
        value = config

        for key in keys:
            if not isinstance(value, dict) or key not in value:
                error_msg = f"Key path '{key_path}' not found in configuration"
                self.logger.error(error_msg)
                raise KeyError(error_msg)
            value = value[key]

        return value

    def set(self, config: Dict[str, Any], key_path: str, value: Any) -> None:
        """
        Set a nested value in config using dot notation.

        Args:
            config: The configuration dictionary
            key_path: Dot-separated key path (e.g., 'experiment.learning_rate')
            value: The value to set

        Raises:
            ValueError: If a non-dict intermediate key blocks the path
        """
        keys = key_path.split(".")
        current = config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                error_msg = (
                    f"Cannot set nested value: '{key}' is not a dictionary"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            current = current[key]

        # Set the value
        current[keys[-1]] = value

    def exists(self, config: Dict[str, Any], key_path: str) -> bool:
        """
        Check if a key path exists in the configuration.

        Args:
            config: The configuration dictionary
            key_path: Dot-separated key path

        Returns:
            True if the key exists, False otherwise
        """
        try:
            self.get(config, key_path)
            return True
        except KeyError:
            return False
