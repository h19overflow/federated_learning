# Phase 1 API Reference

**Complete API documentation for all Phase 1 components**

## Configuration Classes

### SystemConstants

**File**: `src/entities/system_constants.py`

Immutable global configuration values for the system.

```python
@dataclass(frozen=True)
class SystemConstants:
    IMG_SIZE: Tuple[int, int] = (224, 224)
    IMAGE_EXTENSION: str = '.png'
    BATCH_SIZE: int = 128
    SAMPLE_FRACTION: float = 0.10
    VALIDATION_SPLIT: float = 0.20
    SEED: int = 42
    # ... path and column configurations
```

**Methods**:
- `create_custom(**kwargs) -> SystemConstants`: Create with custom values
- All attributes are immutable after creation

**Example**:
```python
constants = SystemConstants.create_custom(
    img_size=(512, 512),
    batch_size=64,
    seed=999
)
```

### ExperimentConfig

**File**: `src/entities/experiment_config.py`

Comprehensive experiment configuration with validation.

```python
@dataclass
class ExperimentConfig:
    # Model parameters
    learning_rate: float = 0.001
    epochs: int = 10
    weight_decay: float = 0.0001
    dropout_rate: float = 0.5
    fine_tune_layers_count: int = 0
    num_classes: int = 1

    # Image processing parameters
    color_mode: str = 'RGB'
    use_imagenet_norm: bool = True
    augmentation_strength: float = 1.0
    # ... additional parameters
```

**Methods**:
- `to_dict() -> Dict[str, Any]`: Convert to dictionary
- `from_dict(config_dict: Dict) -> ExperimentConfig`: Create from dictionary
- `from_system_constants(constants: SystemConstants, **kwargs) -> ExperimentConfig`
- `get_custom_preprocessing_config() -> Dict[str, Any]`: Get preprocessing parameters

**Validation**: Automatic parameter validation on initialization

## Data Processing

### Core Functions

**File**: `src/utils/data_processing.py`

#### load_metadata()
```python
def load_metadata(
    metadata_path: Union[str, Path],
    constants: SystemConstants,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame
```
Load and prepare metadata CSV with filename column generation.

#### sample_dataframe()
```python
def sample_dataframe(
    df: pd.DataFrame,
    sample_fraction: float,
    target_column: str,
    seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame
```
Sample DataFrame with stratified sampling to maintain class balance.

#### create_train_val_split()
```python
def create_train_val_split(
    df: pd.DataFrame,
    validation_split: float,
    target_column: str,
    seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]
```
Create train/validation split with stratification.

#### load_and_split_data()
```python
def load_and_split_data(
    constants: SystemConstants,
    config: ExperimentConfig,
    metadata_path: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]
```
Complete data processing pipeline.

### Utility Functions

#### validate_image_paths()
```python
def validate_image_paths(
    constants: SystemConstants,
    logger: Optional[logging.Logger] = None
) -> bool
```

#### get_image_directory_path()
```python
def get_image_directory_path(constants: SystemConstants) -> str
```

#### get_data_statistics()
```python
def get_data_statistics(df: pd.DataFrame, target_column: str) -> dict
```

## Dataset Management

### CustomImageDataset

**File**: `src/entities/custom_image_dataset.py`

PyTorch Dataset for X-ray images with validation and error handling.

```python
class CustomImageDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Union[str, Path],
        constants: SystemConstants,
        transform: Optional[Callable] = None,
        color_mode: str = 'RGB',
        validate_images: bool = True
    )
```

**Methods**:
- `__len__() -> int`: Number of valid samples
- `__getitem__(idx: int) -> Tuple[torch.Tensor, torch.Tensor]`: Get image and label
- `get_class_distribution() -> dict`: Class counts
- `get_sample_info(idx: int) -> dict`: Detailed sample information
- `validate_all_images() -> Tuple[int, int, list]`: Validate all images
- `get_memory_usage_estimate() -> dict`: Memory usage estimation

**Error Handling**:
- Invalid images are filtered during initialization
- Graceful handling of missing files
- Comprehensive logging for debugging

## Image Transforms

### XRayPreprocessor

**File**: `src/utils/image_transforms.py`

Specialized preprocessing for X-ray images.

```python
class XRayPreprocessor:
    @staticmethod
    def contrast_stretch_percentile(
        image: Image.Image,
        lower_percentile: float = 5.0,
        upper_percentile: float = 95.0
    ) -> Image.Image

    @staticmethod
    def adaptive_histogram_equalization(
        image: Image.Image,
        clip_limit: float = 2.0
    ) -> Image.Image

    @staticmethod
    def edge_enhancement(
        image: Image.Image,
        strength: float = 1.0
    ) -> Image.Image
```

### TransformBuilder

Configurable transform pipeline creation.

```python
class TransformBuilder:
    def __init__(self, constants: SystemConstants, config: ExperimentConfig)

    def build_training_transforms(
        self,
        enable_augmentation: bool = True,
        augmentation_strength: float = 1.0,
        custom_preprocessing: Optional[dict] = None
    ) -> transforms.Compose

    def build_validation_transforms(
        self,
        custom_preprocessing: Optional[dict] = None
    ) -> transforms.Compose

    def build_test_time_augmentation_transforms(
        self,
        num_augmentations: int = 5
    ) -> list
```

### Convenience Functions

#### get_transforms()
```python
def get_transforms(
    constants: SystemConstants,
    config: ExperimentConfig,
    is_training: bool = True,
    use_custom_preprocessing: bool = False,
    augmentation_strength: float = 1.0,
    **kwargs
) -> transforms.Compose
```

## Model Components

### ResNetWithCustomHead

**File**: `src/entities/resnet_with_custom_head.py`

ResNet50 V2 with custom classification head.

```python
class ResNetWithCustomHead(nn.Module):
    def __init__(
        self,
        constants: SystemConstants,
        config: ExperimentConfig,
        base_model_weights: Optional[ResNet50_Weights] = None,
        num_classes: int = 1,
        dropout_rate: Optional[float] = None,
        fine_tune_layers_count: Optional[int] = None,
        custom_head_sizes: Optional[list] = None
    )
```

**Methods**:
- `forward(x: torch.Tensor) -> torch.Tensor`: Forward pass
- `get_model_info() -> dict`: Comprehensive model information
- `freeze_backbone() -> None`: Freeze backbone parameters
- `unfreeze_backbone() -> None`: Unfreeze backbone parameters
- `set_dropout_rate(new_rate: float) -> None`: Update dropout rate
- `get_feature_maps(x: torch.Tensor, layer_name: Optional[str] = None) -> torch.Tensor`

**Fine-tuning Control**:
- `fine_tune_layers_count = 0`: Freeze all backbone layers
- `fine_tune_layers_count < 0`: Unfreeze last N layers
- `fine_tune_layers_count > 0`: Unfreeze first N layers

## PyTorch Lightning Components

### XRayDataModule

**File**: `src/control/xray_data_module.py`

PyTorch Lightning DataModule for X-ray data management.

```python
class XRayDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        constants: SystemConstants,
        config: ExperimentConfig,
        image_dir: Union[str, Path],
        test_df: Optional[pd.DataFrame] = None,
        color_mode: str = 'RGB',
        # ... additional parameters
    )
```

**Methods**:
- `setup(stage: Optional[str] = None) -> None`: Setup datasets
- `train_dataloader() -> DataLoader`: Training data loader
- `val_dataloader() -> DataLoader`: Validation data loader
- `test_dataloader() -> Optional[DataLoader]`: Test data loader
- `predict_dataloader() -> DataLoader`: Prediction data loader
- `get_data_statistics() -> Dict[str, Any]`: Dataset statistics
- `validate_all_images() -> Dict[str, Any]`: Image validation results
- `get_sample_batch(dataset_type: str, batch_size: Optional[int] = None) -> Dict`

### LitResNet

**File**: `src/control/lit_resnet.py`

PyTorch Lightning module for ResNet training.

```python
class LitResNet(pl.LightningModule):
    def __init__(
        self,
        constants: SystemConstants,
        config: ExperimentConfig,
        base_model_weights: Optional[ResNet50_Weights] = None,
        class_weights_tensor: Optional[torch.Tensor] = None,
        num_classes: int = 1,
        monitor_metric: str = "val_loss"
    )
```

**Methods**:
- `forward(x: torch.Tensor) -> torch.Tensor`: Forward pass
- `training_step(batch, batch_idx) -> torch.Tensor`: Training step
- `validation_step(batch, batch_idx) -> torch.Tensor`: Validation step
- `test_step(batch, batch_idx) -> torch.Tensor`: Test step
- `predict_step(batch, batch_idx) -> torch.Tensor`: Prediction step
- `configure_optimizers() -> Dict[str, Any]`: Optimizer configuration
- `get_model_summary() -> Dict[str, Any]`: Model summary
- `freeze_backbone() -> None`: Freeze backbone
- `unfreeze_backbone() -> None`: Unfreeze backbone
- `set_fine_tuning_mode(enabled: bool = True) -> None`: Fine-tuning mode
- `compute_class_weights(train_dataloader) -> torch.Tensor`: Compute class weights

**Metrics Tracked**:
- Training: accuracy, F1
- Validation: accuracy, precision, recall, F1, AUROC
- Test: accuracy, precision, recall, F1, AUROC

## Configuration Management

### ConfigLoader

**File**: `src/utils/config_loader.py`

YAML-based configuration loading with environment variable support.

```python
class ConfigLoader:
    def __init__(self, config_dir: str = "config")

    def load_config(self, config_file: str = "default_config.yaml") -> Dict[str, Any]

    def create_system_constants(self, config: Optional[Dict[str, Any]] = None) -> SystemConstants

    def create_experiment_config(self, config: Optional[Dict[str, Any]] = None) -> ExperimentConfig

    def save_config(self, config: Dict[str, Any], filename: str) -> None

    def override_from_env(self, config: Dict[str, Any], prefix: str = "FPD_") -> Dict[str, Any]
```

## Common Usage Patterns

### Basic Setup
```python
from src.entities import SystemConstants, ExperimentConfig
from src.utils import ConfigLoader, load_and_split_data, get_image_directory_path
from src.control import XRayDataModule, LitResNet

# Load configuration
config_loader = ConfigLoader()
constants = config_loader.create_system_constants()
config = config_loader.create_experiment_config()

# Process data
train_df, val_df = load_and_split_data(constants, config)

# Setup data module
image_dir = get_image_directory_path(constants)
data_module = XRayDataModule(
    train_df=train_df,
    val_df=val_df,
    constants=constants,
    config=config,
    image_dir=image_dir
)

# Create model
model = LitResNet(constants=constants, config=config)
```

### Custom Configuration
```python
# Custom constants
constants = SystemConstants.create_custom(
    img_size=(512, 512),
    batch_size=64,
    sample_fraction=0.5
)

# Custom experiment config
config = ExperimentConfig(
    learning_rate=0.01,
    dropout_rate=0.3,
    fine_tune_layers_count=-2,  # Unfreeze last 2 layers
    use_custom_preprocessing=True
)
```

### Advanced Features
```python
# Custom preprocessing
preprocessing_config = {
    'contrast_stretch': True,
    'adaptive_hist': True,
    'edge_enhance': False,
    'lower_percentile': 2.0,
    'upper_percentile': 98.0
}

# Create data module with custom preprocessing
data_module = XRayDataModule(
    train_df=train_df,
    val_df=val_df,
    constants=constants,
    config=config,
    image_dir=image_dir,
    custom_preprocessing_config=preprocessing_config
)

# Model with class weights
class_weights = torch.tensor([0.3, 0.7])  # Class imbalance handling
model = LitResNet(
    constants=constants,
    config=config,
    class_weights_tensor=class_weights
)
```

## Error Handling

All components implement comprehensive error handling:

- **Configuration Validation**: Parameter validation with clear error messages
- **Data Validation**: CSV structure, image files, missing values
- **Runtime Error Recovery**: Graceful handling of corrupted data
- **Logging**: Detailed logging for debugging and monitoring

## Type Hints

All functions and methods include complete type annotations for better IDE support and code clarity.

## Testing Support

The API is designed for easy testing with:
- Mock-friendly interfaces
- Dependency injection
- Isolated components
- Comprehensive fixtures in `tests/fixtures/`