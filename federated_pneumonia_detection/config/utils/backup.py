"""Configuration backup and restore operations.

Provides utilities for creating and managing configuration backups
with timestamped filenames and restoration capabilities.
"""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger


class ConfigBackup:
    """Handles backup and restore operations for configuration files."""

    def __init__(self):
        """Initialize the backup manager with logger."""
        self.logger = get_logger(__name__)

    def create(
        self, config: Dict[str, Any], config_path: str, backup_path: str = None
    ) -> str:
        """
        Create a backup of the current configuration.

        Args:
            config: The configuration dictionary to backup
            config_path: Path to the original configuration file
            backup_path: Optional custom path for the backup file.
                        If None, creates a timestamped backup in the same directory.

        Returns:
            Path to the created backup file

        Raises:
            IOError: If the backup file cannot be written
        """
        config_path = Path(config_path)

        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = config_path.with_suffix(
                f".backup_{timestamp}.yaml"
            )

        backup_path = Path(backup_path)

        try:
            with open(backup_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(
                    config, file, default_flow_style=False, indent=2
                )
            self.logger.info(f"Created backup at {backup_path}")
            return str(backup_path)
        except IOError as e:
            self.logger.error(f"Failed to create backup at {backup_path}: {e}")
            raise

    def restore(self, backup_path: str) -> Dict[str, Any]:
        """
        Restore configuration from a backup file.

        Args:
            backup_path: Path to the backup file

        Returns:
            Configuration dictionary from the backup

        Raises:
            FileNotFoundError: If the backup file does not exist
            yaml.YAMLError: If the backup file is malformed
        """
        backup_path = Path(backup_path)
        if not backup_path.exists():
            error_msg = f"Backup file not found: {backup_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(backup_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                self.logger.info(f"Restored configuration from {backup_path}")
                return config if config is not None else {}
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse backup file {backup_path}: {e}"
            self.logger.error(error_msg)
            raise
