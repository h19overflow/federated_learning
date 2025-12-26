"""Configuration flattening utilities.

Converts nested dictionaries to dot-notation keys and vice versa for
easier manipulation of configuration structures.
"""

from typing import Any, Dict
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger


class ConfigFlattener:
    """Handles flattening and unflattening of nested configuration dictionaries."""

    def __init__(self):
        """Initialize the flattener with logger."""
        self.logger = get_logger(__name__)

    def flatten(
        self, config_dict: Dict[str, Any], prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Convert nested config dictionary to dot-notation keys.

        Recursively flattens nested dictionaries into dot-separated keys,
        suitable for use with configuration get/set operations.

        Args:
            config_dict: The configuration dictionary to flatten
            prefix: The prefix to use for keys (used for recursion)

        Returns:
            Dictionary with flattened dot-notation keys and non-None values

        Example:
            nested = {
                "experiment": {"learning_rate": 0.002},
                "system": {"batch_size": 256}
            }
            flattened = flattener.flatten(nested)
            # Returns: {
            #     "experiment.learning_rate": 0.002,
            #     "system.batch_size": 256
            # }
        """
        flattened = {}
        for key, value in config_dict.items():
            if value is None:
                continue
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self.flatten(value, full_key))
            else:
                flattened[full_key] = value
        return flattened

    def collect_keys(
        self, config: Dict[str, Any], prefix: str = ""
    ) -> list:
        """
        Collect all keys from a configuration dictionary in dot notation.

        Args:
            config: The configuration dictionary to traverse
            prefix: The prefix to use for keys (used for recursion)

        Returns:
            List of all key paths in dot notation

        Example:
            config = {
                "experiment": {"learning_rate": 0.002, "epochs": 20},
                "system": {"batch_size": 256}
            }
            keys = flattener.collect_keys(config)
            # Returns: [
            #     "experiment.learning_rate",
            #     "experiment.epochs",
            #     "system.batch_size"
            # ]
        """
        keys = []
        if not isinstance(config, dict):
            return keys

        for key, value in config.items():
            current_path = f"{prefix}.{key}" if prefix else key
            keys.append(current_path)
            if isinstance(value, dict):
                keys.extend(self.collect_keys(value, current_path))

        return keys
