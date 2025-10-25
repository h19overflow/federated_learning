"""
Configuration loading utilities for the federated pneumonia detection system.
Handles loading from YAML files and environment variables with validation.
"""

import os
from typing import Dict, Any, Optional
import yaml

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger


class ConfigLoader:
    """
    Load and manage configuration from YAML files and environment variables.

    Provides centralized configuration management with validation and defaults.
    """

    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration loader.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir
        self.logger = get_logger(__name__)

    def load_config(
        self,
        config_file: str = "federated_pneumonia_detection/config/default_config.yaml",
    ) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_file: Name of configuration file to load, or full path

        Returns:
            Dictionary containing configuration values

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        # Handle both relative names and full paths
        if os.path.isabs(config_file) or "\\" in config_file or "/" in config_file:
            # It's a path (absolute or relative with separators)
            config_path = config_file
        else:
            # It's just a filename, use config_dir
            config_path = os.path.join(self.config_dir, config_file)

        if not os.path.exists(config_path):
            self.logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            from pathlib import Path
            abs_path = Path(config_path).resolve()

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            self.logger.info(f"Configuration loaded from {config_path} (absolute: {abs_path})")

            # Log key config values for debugging
            if 'experiment' in config:
                exp_config = config['experiment']
                self.logger.info(f"[ConfigLoader] Loaded from YAML - "
                               f"epochs={exp_config.get('epochs', 'NOT SET')}, "
                               f"batch_size={exp_config.get('batch_size', 'NOT SET')}, "
                               f"learning_rate={exp_config.get('learning_rate', 'NOT SET')}")

            return config

        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML config: {e}")
            raise yaml.YAMLError(f"Error parsing YAML config: {e}")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise Exception(f"Error loading configuration: {e}") from e

    def create_system_constants(
        self, config: Optional[Dict[str, Any]] = None
    ) -> SystemConstants:
        """
        Create SystemConstants from configuration.

        Args:
            config: Configuration dictionary (loads default if None)

        Returns:
            SystemConstants instance
        """
        if config is None:
            config = self.load_config()

        system_config = config.get("system", {})
        paths_config = config.get("paths", {})
        columns_config = config.get("columns", {})

        return SystemConstants.create_custom(
            img_size=tuple(system_config.get("img_size", [224, 224])),
            batch_size=system_config.get("batch_size", 128),
            sample_fraction=system_config.get("sample_fraction", 0.10),
            validation_split=system_config.get("validation_split", 0.20),
            seed=system_config.get("seed", 42),
            base_path=paths_config.get("base_path", "."),
            main_images_folder=paths_config.get("main_images_folder", "Images"),
            images_subfolder=paths_config.get("images_subfolder", "Images"),
            metadata_filename=paths_config.get(
                "metadata_filename", "Train_metadata.csv"
            ),
            image_extension=system_config.get("image_extension", ".png"),
        )

    def create_experiment_config(
        self, config: Optional[Dict[str, Any]] = None
    ) -> ExperimentConfig:
        """
        Create ExperimentConfig from configuration.

        Args:
            config: Configuration dictionary (loads default if None)

        Returns:
            ExperimentConfig instance
        """
        if config is None:
            config = self.load_config()

        system_config = config.get("system", {})
        experiment_config = config.get("experiment", {})
        output_config = config.get("output", {})

        # Extract epochs with logging
        epochs_value = experiment_config.get("epochs", 10)
        self.logger.info(f"[ConfigLoader] Creating ExperimentConfig with epochs={epochs_value} (from YAML: {experiment_config.get('epochs', 'NOT FOUND')})")

        return ExperimentConfig(
            # Model parameters
            learning_rate=experiment_config.get("learning_rate", 0.001),
            epochs=epochs_value,
            weight_decay=experiment_config.get("weight_decay", 0.0001),
            freeze_backbone=experiment_config.get("freeze_backbone", True),
            dropout_rate=experiment_config.get("dropout_rate", 0.5),
            fine_tune_layers_count=experiment_config.get("fine_tune_layers_count", 0),
            num_classes=experiment_config.get("num_classes", 1),
            monitor_metric=experiment_config.get("monitor_metric", "val_loss"),
            # Data parameters
            sample_fraction=system_config.get("sample_fraction", 0.10),
            validation_split=system_config.get("validation_split", 0.20),
            batch_size=system_config.get("batch_size", 128),
            # Training parameters
            early_stopping_patience=experiment_config.get("early_stopping_patience", 5),
            reduce_lr_patience=experiment_config.get("reduce_lr_patience", 3),
            reduce_lr_factor=experiment_config.get("reduce_lr_factor", 0.5),
            min_lr=experiment_config.get("min_lr", 1e-7),
            # Federated Learning parameters
            num_rounds=experiment_config.get("num_rounds", 10),
            num_clients=experiment_config.get("num_clients", 5),
            clients_per_round=experiment_config.get("clients_per_round", 3),
            local_epochs=experiment_config.get("local_epochs", 1),
            # System parameters
            seed=system_config.get("seed", 42),
            device=experiment_config.get("device", "auto"),
            num_workers=experiment_config.get("num_workers", 4),
            # Image processing parameters
            color_mode=experiment_config.get("color_mode", "RGB"),
            use_imagenet_norm=experiment_config.get("use_imagenet_norm", True),
            augmentation_strength=experiment_config.get("augmentation_strength", 1.0),
            use_custom_preprocessing=experiment_config.get(
                "use_custom_preprocessing", False
            ),
            validate_images_on_init=experiment_config.get(
                "validate_images_on_init", True
            ),
            pin_memory=experiment_config.get("pin_memory", True),
            persistent_workers=experiment_config.get("persistent_workers", False),
            prefetch_factor=experiment_config.get("prefetch_factor", 2),
            # Custom preprocessing parameters
            contrast_stretch=experiment_config.get("contrast_stretch", True),
            adaptive_histogram=experiment_config.get("adaptive_histogram", False),
            edge_enhancement=experiment_config.get("edge_enhancement", False),
            lower_percentile=experiment_config.get("lower_percentile", 5.0),
            upper_percentile=experiment_config.get("upper_percentile", 95.0),
            clip_limit=experiment_config.get("clip_limit", 2.0),
            edge_strength=experiment_config.get("edge_strength", 1.0),
            # Paths
            base_path=config.get("paths", {}).get("base_path", "."),
            checkpoint_dir=output_config.get("checkpoint_dir", "models/checkpoints"),
            results_dir=output_config.get("results_dir", "results"),
            log_dir=output_config.get("log_dir", "logs"),
        )

    def save_config(self, config: Dict[str, Any], filename: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: Configuration dictionary to save
            filename: Name of file to save to
        """
        filepath = os.path.join(self.config_dir, filename)

        try:
            # Create directory if it doesn't exist
            os.makedirs(self.config_dir, exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)

            self.logger.info(f"Configuration saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise Exception(f"Failed to save configuration: {e}") from e

    def override_from_env(
        self, config: Dict[str, Any], prefix: str = "FPD_"
    ) -> Dict[str, Any]:
        """
        Override configuration values from environment variables.

        Args:
            config: Configuration dictionary to modify
            prefix: Prefix for environment variables

        Returns:
            Modified configuration dictionary
        """
        env_overrides = {}

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix) :].lower()

                # Convert string values to appropriate types
                if value.lower() in ("true", "false"):
                    env_overrides[config_key] = value.lower() == "true"
                elif value.isdigit():
                    env_overrides[config_key] = int(value)
                else:
                    try:
                        env_overrides[config_key] = float(value)
                    except ValueError:
                        env_overrides[config_key] = value

        if env_overrides:
            self.logger.info(
                f"Applied environment overrides: {list(env_overrides.keys())}"
            )
            # Simple merge - could be enhanced for nested dictionaries
            self.logger.info(f"Environment overrides: {env_overrides}")
            config.update(env_overrides)

        return config


if __name__ == "__main__":
    loader = ConfigLoader(
        config_dir=r"C:\Users\User\Projects\FYP2\federated_pneumonia_detection\config"
    )
    print(loader.create_experiment_config(loader.load_config()))
