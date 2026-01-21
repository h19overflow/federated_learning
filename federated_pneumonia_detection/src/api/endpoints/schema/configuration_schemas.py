"""Pydantic schemas for configuration endpoint requests."""

from typing import List, Optional

from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    """System configuration schema."""

    img_size: Optional[List[int]] = Field(
        None, description="Image size as [height, width]"
    )
    image_extension: Optional[str] = Field(None, description="Image file extension")
    batch_size: Optional[int] = Field(None, description="Batch size for training")
    sample_fraction: Optional[float] = Field(
        None, description="Fraction of data to sample"
    )
    validation_split: Optional[float] = Field(
        None, description="Validation split ratio"
    )
    seed: Optional[int] = Field(None, description="Random seed")


class PathsConfig(BaseModel):
    """Paths configuration schema."""

    base_path: Optional[str] = Field(None, description="Base path for data")
    main_images_folder: Optional[str] = Field(
        None, description="Main images folder name"
    )
    images_subfolder: Optional[str] = Field(None, description="Images subfolder name")
    metadata_filename: Optional[str] = Field(None, description="Metadata CSV filename")


class ColumnsConfig(BaseModel):
    """Columns configuration schema."""

    patient_id: Optional[str] = Field(None, description="Patient ID column name")
    target: Optional[str] = Field(None, description="Target column name")
    filename: Optional[str] = Field(None, description="Filename column name")


class ExperimentConfig(BaseModel):
    """Experiment configuration schema."""

    # Model parameters
    learning_rate: Optional[float] = Field(None, description="Learning rate")
    epochs: Optional[int] = Field(None, description="Number of epochs")
    batch_size: Optional[int] = Field(None, description="Batch size for training")
    weight_decay: Optional[float] = Field(None, description="Weight decay")
    freeze_backbone: Optional[bool] = Field(
        None, description="Whether to freeze backbone"
    )
    dropout_rate: Optional[float] = Field(None, description="Dropout rate")
    fine_tune_layers_count: Optional[int] = Field(
        None, description="Number of layers to fine-tune"
    )
    num_classes: Optional[int] = Field(None, description="Number of classes")
    monitor_metric: Optional[str] = Field(None, description="Metric to monitor")

    # Training parameters
    early_stopping_patience: Optional[int] = Field(
        None, description="Early stopping patience"
    )
    reduce_lr_patience: Optional[int] = Field(None, description="Reduce LR patience")
    reduce_lr_factor: Optional[float] = Field(None, description="Reduce LR factor")
    min_lr: Optional[float] = Field(None, description="Minimum learning rate")
    validation_split: Optional[float] = Field(
        None, description="Validation split ratio (0.0 to 1.0)"
    )

    # Federated Learning parameters
    num_rounds: Optional[int] = Field(None, description="Number of federated rounds")
    num_clients: Optional[int] = Field(None, description="Number of clients")
    clients_per_round: Optional[int] = Field(None, description="Clients per round")
    local_epochs: Optional[int] = Field(None, description="Local epochs per round")

    # System parameters
    device: Optional[str] = Field(
        None, description="Device to use (cuda, cpu, mps, auto)"
    )
    num_workers: Optional[int] = Field(
        None, description="Number of workers for data loading"
    )

    # Image processing parameters
    color_mode: Optional[str] = Field(None, description="Color mode (RGB or L)")
    use_imagenet_norm: Optional[bool] = Field(
        None, description="Use ImageNet normalization"
    )
    augmentation_strength: Optional[float] = Field(
        None, description="Augmentation strength"
    )
    use_custom_preprocessing: Optional[bool] = Field(
        None, description="Use custom preprocessing"
    )
    validate_images_on_init: Optional[bool] = Field(
        None, description="Validate images on init"
    )
    pin_memory: Optional[bool] = Field(None, description="Pin memory for GPU")
    persistent_workers: Optional[bool] = Field(
        None, description="Use persistent workers"
    )
    prefetch_factor: Optional[int] = Field(
        None, description="Prefetch factor for data loading"
    )

    # Custom preprocessing parameters
    contrast_stretch: Optional[bool] = Field(None, description="Apply contrast stretch")
    adaptive_histogram: Optional[bool] = Field(
        None, description="Apply adaptive histogram"
    )
    edge_enhancement: Optional[bool] = Field(None, description="Apply edge enhancement")
    lower_percentile: Optional[float] = Field(
        None, description="Lower percentile for preprocessing"
    )
    upper_percentile: Optional[float] = Field(
        None, description="Upper percentile for preprocessing"
    )
    clip_limit: Optional[float] = Field(
        None, description="Clip limit for preprocessing"
    )
    edge_strength: Optional[float] = Field(
        None, description="Edge strength for preprocessing"
    )


class OutputConfig(BaseModel):
    """Output directories configuration schema."""

    checkpoint_dir: Optional[str] = Field(None, description="Checkpoint directory")
    results_dir: Optional[str] = Field(None, description="Results directory")
    log_dir: Optional[str] = Field(None, description="Logs directory")


class LoggingConfig(BaseModel):
    """Logging configuration schema."""

    level: Optional[str] = Field(None, description="Logging level")
    format: Optional[str] = Field(None, description="Logging format")
    file_logging: Optional[bool] = Field(None, description="Enable file logging")


class ConfigurationUpdateRequest(BaseModel):
    """Root configuration update request schema."""

    system: Optional[SystemConfig] = Field(None, description="System configuration")
    paths: Optional[PathsConfig] = Field(None, description="Paths configuration")
    columns: Optional[ColumnsConfig] = Field(None, description="Columns configuration")
    experiment: Optional[ExperimentConfig] = Field(
        None, description="Experiment configuration"
    )
    output: Optional[OutputConfig] = Field(None, description="Output configuration")
    logging: Optional[LoggingConfig] = Field(None, description="Logging configuration")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "experiment": {
                    "learning_rate": 0.002,
                    "epochs": 20,
                },
                "system": {
                    "batch_size": 256,
                },
            }
        }
