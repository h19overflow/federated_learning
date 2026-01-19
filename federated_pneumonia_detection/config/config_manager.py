"""
Configuration Manager for Federated Pneumonia Detection System

This module provides a centralized way to read and modify configuration values
from YAML files. It supports nested field access using dot notation and provides
type-safe operations for configuration management.

Example usage:
    config_manager = ConfigManager()
    learning_rate = config_manager.get('experiment.learning_rate')
    config_manager.set('experiment.learning_rate', 0.002)
    config_manager.save()
"""

from typing import Any, Dict, List
from pathlib import Path
import copy
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger
from federated_pneumonia_detection.config.internals import (
    YamlConfigLoader,
    NestedAccessor,
    ConfigFlattener,
    ConfigBackup,
)


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
            config_dir = Path(__file__).parent
            config_path = config_dir / "default_config.yaml"

        self.logger = get_logger(__name__)
        self.config_path = Path(config_path)
        self.loader = YamlConfigLoader()
        self.accessor = NestedAccessor()
        self.flattener = ConfigFlattener()
        self.backup_manager = ConfigBackup()

        self.config = self.loader.load(str(self.config_path))
        self._original_config = copy.deepcopy(self.config)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        try:
            return self.accessor.get(self.config, key_path)
        except KeyError:
            if default is not None:
                return default
            raise

    def set(self, key_path: str, value: Any) -> None:
        """Set a configuration value using dot notation."""
        self.accessor.set(self.config, key_path, value)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values at once."""
        for key_path, value in updates.items():
            self.set(key_path, value)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section."""
        return self.get(section, {})

    def set_section(self, section: str, values: Dict[str, Any]) -> None:
        """Set an entire configuration section."""
        self.set(section, values)

    def has_key(self, key_path: str) -> bool:
        """Check if a configuration key exists."""
        return self.accessor.exists(self.config, key_path)

    def list_keys(self, section: str = None) -> List[str]:
        """List all available configuration keys in dot notation."""
        if section:
            section_data = self.get_section(section)
            return self.flattener.collect_keys(section_data, section)
        return self.flattener.collect_keys(self.config)

    def save(self) -> None:
        """Save the current configuration to file."""
        self.logger.info(f"Saving configuration to {self.config_path}")
        self.loader.save(str(self.config_path), self.config)

    def reset(self) -> None:
        """Reset configuration to the original state when manager was created."""
        self.logger.info("Resetting configuration to original state")
        self.config = copy.deepcopy(self._original_config)

    def reload(self) -> None:
        """Reload configuration from file, discarding unsaved changes."""
        self.config = self.loader.load(str(self.config_path))
        self._original_config = copy.deepcopy(self.config)

    def backup(self, backup_path: str = None) -> str:
        """Create a backup of the current configuration."""
        return self.backup_manager.create(self.config, str(self.config_path), backup_path)

    def flatten_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten a nested configuration dictionary to dot-notation keys.

        Args:
            config_dict: The nested configuration dictionary to flatten

        Returns:
            Dictionary with flattened dot-notation keys
        """
        return self.flattener.flatten(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Get a copy of the entire configuration as a dictionary."""
        return copy.deepcopy(self.config)

    def __getitem__(self, key_path: str) -> Any:
        """Support dictionary-style access: config['experiment.learning_rate']"""
        return self.get(key_path)

    def __setitem__(self, key_path: str, value: Any) -> None:
        """Support dictionary-style assignment."""
        self.set(key_path, value)

    def __contains__(self, key_path: str) -> bool:
        """Support 'in' operator: 'experiment.learning_rate' in config"""
        return self.has_key(key_path)

    def __str__(self) -> str:
        """String representation of the configuration."""
        import yaml
        return yaml.safe_dump(self.config, default_flow_style=False, indent=2)

    def __repr__(self) -> str:
        """Developer representation of the ConfigManager."""
        return f"ConfigManager(config_path='{self.config_path}')"


def get_config_manager(config_path: str = None) -> ConfigManager:
    """Factory function to create a ConfigManager instance."""
    return ConfigManager(config_path)


def quick_get(key_path: str, config_path: str = None, default: Any = None) -> Any:
    """Quickly get a configuration value without creating a persistent manager."""
    manager = ConfigManager(config_path)
    return manager.get(key_path, default)


def quick_set(key_path: str, value: Any, config_path: str = None) -> None:
    """Quickly set a configuration value and save it."""
    manager = ConfigManager(config_path)
    manager.set(key_path, value)
    manager.save()
