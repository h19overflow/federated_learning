"""YAML configuration file I/O operations.

Handles loading and saving YAML configuration files with proper error
handling and data integrity checks.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger


class YamlConfigLoader:
    """Handles YAML configuration file loading and saving."""

    def __init__(self):
        """Initialize the YAML loader with logger."""
        self.logger = get_logger(__name__)

    def load(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Dictionary containing the loaded configuration

        Raises:
            FileNotFoundError: If the configuration file does not exist
            yaml.YAMLError: If the YAML file is malformed
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                self.logger.debug(f"Loaded configuration from {config_path}")
                return config if config is not None else {}
        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse YAML file {config_path}: {e}")
            raise

    def save(self, config_path: str, config: Dict[str, Any]) -> None:
        """
        Save configuration to YAML file with fsync for durability.

        Args:
            config_path: Path to save the YAML configuration file
            config: Configuration dictionary to save

        Raises:
            IOError: If the file cannot be written
        """
        config_path = Path(config_path)
        try:
            with open(config_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(
                    config, file, default_flow_style=False, indent=2
                )
                file.flush()
                os.fsync(file.fileno())
            self.logger.info(f"Saved configuration to {config_path}")
        except IOError as e:
            self.logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
