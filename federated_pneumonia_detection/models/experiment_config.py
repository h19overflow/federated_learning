"""
Experiment configuration for storing all experiment parameters.
Contains learning rate, epochs, FL settings, and other hyperparameters.
"""

from typing import Dict, Any
from dataclasses import dataclass, field
from federated_pneumonia_detection.models.system_constants import SystemConstants


@dataclass
class ExperimentConfig:
    """
    Store all experiment parameters for centralized and federated learning.

    Contains hyperparameters, model settings, and training configuration.
    """

    # Model parameters
    learning_rate: float = 0.001
    epochs: int = 15
    weight_decay: float = 0.0001
    freeze_backbone: bool = True
    dropout_rate: float = 0.3
    fine_tune_layers_count: int = -10  # 0 = freeze all, negative = unfreeze last N layers
    num_classes: int = 1  # 1 for binary classification
    monitor_metric: str = 'val_loss'  # Metric to monitor for LR scheduling

    # Data parameters
    sample_fraction: float = 0.10
    validation_split: float = 0.20
    batch_size: int = 512

    # Training parameters
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7

    # Federated Learning parameters
    num_rounds: int = 2
    num_clients: int = 2
    clients_per_round: int = 2
    local_epochs: int = 15

    # System parameters
    seed: int = 42
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'
    num_workers: int = 4

    # Image processing parameters
    color_mode: str = 'RGB'  # 'RGB' or 'L'
    use_imagenet_norm: bool = True
    augmentation_strength: float = 1.0  # 0.0 to 2.0
    use_custom_preprocessing: bool = False
    validate_images_on_init: bool = True
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2

    # Custom preprocessing parameters
    contrast_stretch: bool = True
    adaptive_histogram: bool = False
    edge_enhancement: bool = False
    lower_percentile: float = 5.0
    upper_percentile: float = 95.0
    clip_limit: float = 2.0
    edge_strength: float = 1.0

    # Paths and file settings
    base_path: str = '../src/entities'
    checkpoint_dir: str = 'models/checkpoints'
    results_dir: str = 'results'
    log_dir: str = 'logs'

    # Additional parameters
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate all configuration parameters."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.epochs <= 0:
            raise ValueError("Number of epochs must be positive")

        if not 0 < self.sample_fraction <= 1:
            raise ValueError("Sample fraction must be between 0 and 1")

        if not 0 < self.validation_split < 1:
            raise ValueError("Validation split must be between 0 and 1")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.num_rounds <= 0:
            raise ValueError("Number of FL rounds must be positive")

        if self.num_clients <= 0:
            raise ValueError("Number of clients must be positive")

        if self.clients_per_round > self.num_clients:
            raise ValueError("Clients per round cannot exceed total number of clients")

        # Validate image processing parameters
        if self.color_mode not in ['RGB', 'L']:
            raise ValueError("color_mode must be 'RGB' or 'L'")

        if not 0.0 <= self.augmentation_strength <= 2.0:
            raise ValueError("augmentation_strength must be between 0.0 and 2.0")

        if not 0.0 <= self.lower_percentile < self.upper_percentile <= 100.0:
            raise ValueError("Percentiles must satisfy: 0 <= lower < upper <= 100")

        if self.clip_limit <= 0:
            raise ValueError("clip_limit must be positive")

        if not 0.0 <= self.edge_strength <= 2.0:
            raise ValueError("edge_strength must be between 0.0 and 2.0")

        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be at least 1")

        # Validate model parameters
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be between 0.0 and 1.0")

        if not isinstance(self.fine_tune_layers_count, int):
            raise ValueError("fine_tune_layers_count must be an integer")

        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")

        if self.monitor_metric not in ['val_loss', 'val_acc', 'val_f1', 'val_auroc']:
            raise ValueError("monitor_metric must be one of: val_loss, val_acc, val_f1, val_auroc")

    @classmethod
    def from_system_constants(cls, constants: SystemConstants, **kwargs) -> 'ExperimentConfig':
        """
        Create experiment config from system constants.

        Args:
            constants: SystemConstants instance
            **kwargs: Additional parameters to override defaults

        Returns:
            ExperimentConfig instance
        """
        return cls(
            sample_fraction=constants.SAMPLE_FRACTION,
            validation_split=constants.VALIDATION_SPLIT,
            batch_size=constants.BATCH_SIZE,
            seed=constants.SEED,
            base_path=constants.BASE_PATH,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'weight_decay': self.weight_decay,
            'freeze_backbone': self.freeze_backbone,
            'dropout_rate': self.dropout_rate,
            'fine_tune_layers_count': self.fine_tune_layers_count,
            'num_classes': self.num_classes,
            'monitor_metric': self.monitor_metric,
            'sample_fraction': self.sample_fraction,
            'validation_split': self.validation_split,
            'batch_size': self.batch_size,
            'early_stopping_patience': self.early_stopping_patience,
            'reduce_lr_patience': self.reduce_lr_patience,
            'reduce_lr_factor': self.reduce_lr_factor,
            'min_lr': self.min_lr,
            'num_rounds': self.num_rounds,
            'num_clients': self.num_clients,
            'clients_per_round': self.clients_per_round,
            'local_epochs': self.local_epochs,
            'seed': self.seed,
            'device': self.device,
            'num_workers': self.num_workers,
            'color_mode': self.color_mode,
            'use_imagenet_norm': self.use_imagenet_norm,
            'augmentation_strength': self.augmentation_strength,
            'use_custom_preprocessing': self.use_custom_preprocessing,
            'validate_images_on_init': self.validate_images_on_init,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers,
            'prefetch_factor': self.prefetch_factor,
            'contrast_stretch': self.contrast_stretch,
            'adaptive_histogram': self.adaptive_histogram,
            'edge_enhancement': self.edge_enhancement,
            'lower_percentile': self.lower_percentile,
            'upper_percentile': self.upper_percentile,
            'clip_limit': self.clip_limit,
            'edge_strength': self.edge_strength,
            'base_path': self.base_path,
            'checkpoint_dir': self.checkpoint_dir,
            'results_dir': self.results_dir,
            'log_dir': self.log_dir,
            'metadata': self.metadata
        }

    def get_custom_preprocessing_config(self) -> Dict[str, Any]:
        """
        Get custom preprocessing configuration as dictionary.

        Returns:
            Dictionary with preprocessing parameters
        """
        return {
            'contrast_stretch': self.contrast_stretch,
            'adaptive_hist': self.adaptive_histogram,
            'edge_enhance': self.edge_enhancement,
            'lower_percentile': self.lower_percentile,
            'upper_percentile': self.upper_percentile,
            'clip_limit': self.clip_limit,
            'edge_strength': self.edge_strength
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            ExperimentConfig instance
        """
        return cls(**config_dict)

if __name__ == "__main__":
    # Example usage
    config = ExperimentConfig()
    print(config)
    print(config.to_dict())