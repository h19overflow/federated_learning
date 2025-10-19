"""
Configuration Manager for Federated Pneumonia Detection System

This module provides a centralized way to read and modify configuration values
from the YAML configuration files. It supports nested field access using dot notation
and provides type-safe operations for configuration management.

Example usage:
    config_manager = ConfigManager()

    # Get values
    learning_rate = config_manager.get('experiment.learning_rate')
    img_size = config_manager.get('system.img_size')

    # Set values
    config_manager.set('experiment.learning_rate', 0.002)
    config_manager.set('system.batch_size', 256)

    # Save changes
    config_manager.save()
"""

import os
import yaml
from typing import Any, Dict, List, Union
from pathlib import Path
import copy
from federated_pneumonia_detection.src.utils.logger import get_logger


class ConfigManager:
    """Centralized configuration manager for YAML configuration files."""

    def __init__(self, config_path: str = None):
        """
        Initialize the ConfigManager.

        Args:
            config_path: Path to the configuration YAML file.
                        If None, uses default_config.yaml in the same directory.
        """
        if config_path is None:
            # Default to the config file in the same directory
            config_dir = Path(__file__).parent
            config_path = config_dir / "default_config.yaml"
        self.logger = get_logger(__name__)
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        self.config = self._load_config()
        self._original_config = copy.deepcopy(self.config)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _save_config(self) -> None:
        """Save configuration to YAML file."""
        with open(self.config_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(self.config, file, default_flow_style=False, indent=2)

    def _get_nested_value(self, config: Dict[str, Any], key_path: str) -> Any:
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
                self.logger.error(f"Key path '{key_path}' not found in configuration")
                raise KeyError(f"Key path '{key_path}' not found in configuration")
            value = value[key]

        return value

    def _set_nested_value(
        self, config: Dict[str, Any], key_path: str, value: Any
    ) -> None:
        """
        Set a nested value in config using dot notation.

        Args:
            config: The configuration dictionary
            key_path: Dot-separated key path (e.g., 'experiment.learning_rate')
            value: The value to set
        """
        keys = key_path.split(".")
        current = config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                self.logger.error(
                    f"Cannot set nested value: '{key}' is not a dictionary"
                )
                raise ValueError(
                    f"Cannot set nested value: '{key}' is not a dictionary"
                )
            current = current[key]

        # Set the value
        current[keys[-1]] = value

    def _flatten_config(
        self, config_dict: Dict[str, Any], prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Convert nested config dictionary to dot-notation keys.

        Recursively flattens nested dictionaries into dot-separated keys,
        suitable for use with the set() method.

        Args:
            config_dict: The configuration dictionary to flatten
            prefix: The prefix to use for keys (used for recursion)

        Returns:
            Dictionary with flattened dot-notation keys and non-None values

        Example:
            nested = {"experiment": {"learning_rate": 0.002}, "system": {"batch_size": 256}}
            flattened = config.flatten_config(nested)
            # Returns: {"experiment.learning_rate": 0.002, "system.batch_size": 256}
        """
        flattened = {}
        for key, value in config_dict.items():
            if value is None:
                continue
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_config(value, full_key))
            else:
                flattened[full_key] = value
        return flattened

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key_path: Dot-separated key path (e.g., 'experiment.learning_rate')
            default: Default value to return if key doesn't exist

        Returns:
            The configuration value or default if not found

        Example:
            learning_rate = config.get('experiment.learning_rate')
            img_size = config.get('system.img_size', [224, 224])
        """
        try:
            return self._get_nested_value(self.config, key_path)
        except KeyError:
            if default is not None:
                return default
            self.logger.error(f"Key path '{key_path}' not found in configuration")
            raise KeyError(f"Key path '{key_path}' not found in configuration")

    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key_path: Dot-separated key path (e.g., 'experiment.learning_rate')
            value: The value to set

        Example:
            config.set('experiment.learning_rate', 0.002)
            config.set('system.batch_size', 256)
        """
        self._set_nested_value(self.config, key_path, value)

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values at once.

        Args:
            updates: Dictionary of key paths and their new values

        Example:
            config.update({
                'experiment.learning_rate': 0.002,
                'system.batch_size': 256,
                'experiment.epochs': 20
            })
        """
        for key_path, value in updates.items():
            self.set(key_path, value)

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: The section name (e.g., 'experiment', 'system')

        Returns:
            Dictionary containing the section's configuration

        Example:
            experiment_config = config.get_section('experiment')
        """
        return self.get(section, {})

    def set_section(self, section: str, values: Dict[str, Any]) -> None:
        """
        Set an entire configuration section.

        Args:
            section: The section name (e.g., 'experiment', 'system')
            values: Dictionary of values to set in the section

        Example:
            config.set_section('experiment', {
                'learning_rate': 0.002,
                'epochs': 20,
                'batch_size': 256
            })
        """
        self.set(section, values)

    def has_key(self, key_path: str) -> bool:
        """
        Check if a configuration key exists.

        Args:
            key_path: Dot-separated key path

        Returns:
            True if the key exists, False otherwise
        """
        try:
            self._get_nested_value(self.config, key_path)
            return True
        except KeyError:
            self.logger.error(f"Key path '{key_path}' not found in configuration")
            raise KeyError(f"Key path '{key_path}' not found in configuration")
            return False

    def list_keys(self, section: str = None) -> List[str]:
        """
        List all available configuration keys.

        Args:
            section: Optional section to list keys from

        Returns:
            List of key paths
        """

        def _collect_keys(obj, prefix=""):
            keys = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{prefix}.{key}" if prefix else key
                    keys.append(current_path)
                    if isinstance(value, dict):
                        keys.extend(_collect_keys(value, current_path))
            self.logger.error(f"Error collecting keys: {keys}")
            raise ValueError(f"Error collecting keys: {keys}")
            return keys

        if section:
            section_data = self.get_section(section)
            return _collect_keys(section_data, section)
        else:
            return _collect_keys(self.config)

    def save(self) -> None:
        """
        Save the current configuration to file.

        Example:
            config.set('experiment.learning_rate', 0.002)
            config.save()
        """
        self.logger.info(f"Saving configuration to {self.config_path}")
        self._save_config()

    def reset(self) -> None:
        """
        Reset configuration to the original state when the manager was created.
        """
        self.logger.info(f"Resetting configuration to original state")
        self.config = copy.deepcopy(self._original_config)

    def reload(self) -> None:
        """
        Reload configuration from file, discarding any unsaved changes.
        """
        self.config = self._load_config()
        self._original_config = copy.deepcopy(self.config)

    def backup(self, backup_path: str = None) -> str:
        """
        Create a backup of the current configuration.

        Args:
            backup_path: Path for the backup file. If None, creates a timestamped backup.

        Returns:
            Path to the backup file
        """
        if backup_path is None:
            self.logger.info(f"Creating backup of configuration")
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.config_path.with_suffix(f".backup_{timestamp}.yaml")

        backup_path = Path(backup_path)
        with open(backup_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(self.config, file, default_flow_style=False, indent=2)

        return str(backup_path)

    def to_dict(self) -> Dict[str, Any]:
        """
        Get a copy of the entire configuration as a dictionary.

        Returns:
            Deep copy of the configuration dictionary
        """
        return copy.deepcopy(self.config)

    def __getitem__(self, key_path: str) -> Any:
        """Support dictionary-style access: config['experiment.learning_rate']"""
        return self.get(key_path)

    def __setitem__(self, key_path: str, value: Any) -> None:
        """Support dictionary-style assignment: config['experiment.learning_rate'] = 0.002"""
        self.set(key_path, value)

    def __contains__(self, key_path: str) -> bool:
        """Support 'in' operator: 'experiment.learning_rate' in config"""
        return self.has_key(key_path)

    def __str__(self) -> str:
        """String representation of the configuration."""
        return yaml.safe_dump(self.config, default_flow_style=False, indent=2)

    def __repr__(self) -> str:
        """Developer representation of the ConfigManager."""
        return f"ConfigManager(config_path='{self.config_path}')"


# Convenience functions for common operations
def get_config_manager(config_path: str = None) -> ConfigManager:
    """
    Factory function to create a ConfigManager instance.

    Args:
        config_path: Optional path to configuration file

    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)


def quick_get(key_path: str, config_path: str = None, default: Any = None) -> Any:
    """
    Quickly get a configuration value without creating a persistent manager.

    Args:
        key_path: Dot-separated key path
        config_path: Optional path to configuration file
        default: Default value if key doesn't exist

    Returns:
        The configuration value
    """
    manager = ConfigManager(config_path)
    return manager.get(key_path, default)


def quick_set(key_path: str, value: Any, config_path: str = None) -> None:
    """
    Quickly set a configuration value and save it.

    Args:
        key_path: Dot-separated key path
        value: The value to set
        config_path: Optional path to configuration file
    """
    manager = ConfigManager(config_path)
    manager.set(key_path, value)
    manager.save()
