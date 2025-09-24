# Phase 1 Implementation Documentation

**Federated Pneumonia Detection System - Foundation & Core Infrastructure**

## Overview

Phase 1 establishes the foundational architecture and core infrastructure for the federated pneumonia detection system. This phase implements clean, modular components following SOLID principles with comprehensive configuration management and data processing capabilities.

## Architecture Summary

### Three-Layer Architecture (Entity-Control-Boundary)

```
src/
├── entities/           # Data structures and configuration classes
│   ├── system_constants.py
│   ├── experiment_config.py
│   ├── custom_image_dataset.py
│   └── resnet_with_custom_head.py
├── control/           # Business logic and orchestration
│   ├── xray_data_module.py
│   ├── lit_resnet.py
│   ├── training_callbacks.py       # NEW: Training callbacks and utilities
│   └── centralized_trainer.py      # NEW: Complete training orchestrator
├── boundary/          # External interfaces (planned for future phases)
└── utils/             # Support functions and utilities
    ├── data_processing.py
    ├── config_loader.py
    └── image_transforms.py
```

## Implemented Components

### 1. Configuration System

**Files**: `entities/system_constants.py`, `entities/experiment_config.py`, `utils/config_loader.py`, `config/default_config.yaml`

**Key Features**:
- **SystemConstants**: Immutable global configuration values
- **ExperimentConfig**: Comprehensive experiment parameters with validation
- **ConfigLoader**: YAML-based configuration loading with environment variable overrides
- **Validation**: Automatic parameter validation on initialization

**Usage Example**:
```python
from src.utils import ConfigLoader

config_loader = ConfigLoader()
constants = config_loader.create_system_constants()
config = config_loader.create_experiment_config()
```

**Configuration Categories**:
- System constants (image size, paths, column names)
- Model parameters (learning rate, dropout, architecture)
- Training parameters (epochs, patience, optimization)
- Image processing (augmentation, preprocessing, transforms)
- Federated learning settings (clients, rounds, local epochs)

### 2. Data Processing Pipeline

**Files**: `utils/data_processing.py`

**Architecture Change**: Refactored from monolithic `DataProcessor` class to functional approach with utilities:

**Core Functions**:
- `load_metadata()`: CSV loading with validation
- `sample_dataframe()`: Stratified sampling for class balance
- `create_train_val_split()`: Train/validation splitting
- `load_and_split_data()`: Complete pipeline orchestration

**New Data Flow**:
```
CSV → load_and_split_data() → train_df, val_df → XRayDataModule → CustomImageDataset
```

**Benefits**:
- Cleaner separation of concerns
- Better testability and modularity
- Reduced redundancy with other components

### 3. Image Dataset Management

**Files**: `entities/custom_image_dataset.py`

**Key Features**:
- **Robust Error Handling**: Validates images during initialization
- **Flexible Configuration**: Supports RGB/grayscale, custom transforms
- **Memory Monitoring**: Estimates memory usage for large datasets
- **Graceful Degradation**: Handles missing files with detailed logging

**Capabilities**:
- Image validation and filtering
- Class distribution analysis
- Sample information extraction
- Memory usage estimation

### 4. Image Preprocessing & Transforms

**Files**: `utils/image_transforms.py`

**Components**:
- **XRayPreprocessor**: Specialized X-ray image enhancement
  - Percentile-based contrast stretching
  - Adaptive histogram equalization (CLAHE)
  - Edge enhancement for anatomical structures
- **TransformBuilder**: Configurable transform pipeline creation
  - Training vs validation transforms
  - Customizable augmentation strength
  - Test-time augmentation support

**Preprocessing Options**:
- Contrast enhancement for better X-ray visibility
- Configurable augmentation (rotation, flip, brightness)
- ImageNet vs custom normalization
- Custom preprocessing pipelines

### 5. PyTorch Lightning Integration

**Files**: `control/xray_data_module.py`, `control/lit_resnet.py`

#### XRayDataModule
- **Purpose**: Orchestrates dataset creation and data loading
- **Features**:
  - Automatic dataset validation
  - Memory-efficient batch processing
  - Comprehensive statistics tracking
  - Integration with transform pipelines

#### LitResNet
- **Purpose**: PyTorch Lightning module for training
- **Features**:
  - Comprehensive metrics tracking (accuracy, precision, recall, F1, AUC)
  - Class-weighted loss support
  - Advanced optimization (AdamW + ReduceLROnPlateau)
  - Fine-tuning capabilities

### 6. Model Architecture

**Files**: `entities/resnet_with_custom_head.py`

**ResNetWithCustomHead Features**:
- **Backbone**: ResNet50 V2 with ImageNet pre-training
- **Custom Head**: Configurable fully connected layers (2048→256→64→1)
- **Fine-tuning Control**:
  - Freeze/unfreeze backbone
  - Layer-specific fine-tuning (last N layers)
- **Architecture Flexibility**: Custom head sizes, dropout configuration

**Model Configuration Options**:
```yaml
experiment:
  dropout_rate: 0.5
  fine_tune_layers_count: 0  # Negative to unfreeze last N layers
  num_classes: 1  # Binary classification
  freeze_backbone: true
```

### 7. Training Callbacks & Monitoring

**Files**: `control/training_callbacks.py`

**Key Components**:
- **ModelCheckpoint**: Saves best models based on validation recall
- **EarlyStopping**: Prevents overfitting with configurable patience
- **LearningRateMonitor**: Tracks learning rate changes throughout training
- **HighestValRecallCallback**: Custom callback for medical classification tracking
- **Class Weight Computation**: Automatic handling of imbalanced datasets

**Features**:
- Medical-focused metrics (prioritizes recall for pneumonia detection)
- Comprehensive checkpoint management (saves top 3 + backup)
- GPU/CPU auto-detection with mixed precision
- Deterministic training with seed management

### 8. Centralized Training Orchestrator

**Files**: `control/centralized_trainer.py`

**CentralizedTrainer Features**:
- **Zip File Processing**: Accepts datasets in zip format
- **Automatic Dataset Discovery**: Finds CSV and images automatically
- **Complete Pipeline Orchestration**: From zip file to trained model
- **Temporary File Management**: Safe extraction and cleanup
- **Comprehensive Logging**: TensorBoard integration and detailed progress tracking

**Key Capabilities**:
```python
# Simple usage
trainer = CentralizedTrainer()
results = trainer.train_from_zip("pneumonia_dataset.zip")

# Advanced usage with custom configuration
trainer = CentralizedTrainer(
    config_path="custom_config.yaml",
    checkpoint_dir="my_checkpoints",
    logs_dir="my_logs"
)
```

**Input Requirements**:
- Zip file containing CSV metadata and image files
- CSV with columns: `patientId`, `Target`
- Images named as: `{patientId}.png/.jpg`

**Output Structure**:
- Trained models in checkpoint directory
- TensorBoard logs for monitoring
- Comprehensive training metrics and results

## Configuration Management

### YAML Configuration Structure

```yaml
# System Constants
system:
  img_size: [224, 224]
  batch_size: 128
  sample_fraction: 0.10

# Model Parameters
experiment:
  learning_rate: 0.001
  dropout_rate: 0.5
  fine_tune_layers_count: 0

# Image Processing
  color_mode: "RGB"
  use_custom_preprocessing: false
  augmentation_strength: 1.0
```

### Environment Variable Overrides

Prefix environment variables with `FPD_` to override configuration:
```bash
export FPD_LEARNING_RATE=0.01
export FPD_BATCH_SIZE=64
```

## Usage Examples

### Centralized Training from Zip File (Recommended)

```python
from src.control import CentralizedTrainer

# Simple training from zip file
trainer = CentralizedTrainer(
    checkpoint_dir="experiments/checkpoints",
    logs_dir="experiments/logs"
)

# Train directly from zip file
results = trainer.train_from_zip(
    zip_path="pneumonia_dataset.zip",
    experiment_name="my_pneumonia_experiment"
)

# Access results
print(f"Best model: {results['best_model_path']}")
print(f"Best validation recall: {results['best_model_score']:.4f}")
```

### Command Line Usage

```bash
# Basic training
python federated_pneumonia_detection/train_centralized.py dataset.zip

# With custom settings
python federated_pneumonia_detection/train_centralized.py dataset.zip \
    --experiment-name my_experiment \
    --checkpoint-dir ./checkpoints \
    --logs-dir ./logs

# Validate dataset only
python federated_pneumonia_detection/train_centralized.py dataset.zip --validate-only
```

### Manual Pipeline Setup (Advanced)

```python
from src.entities import SystemConstants, ExperimentConfig
from src.utils import load_and_split_data, get_image_directory_path
from src.control import XRayDataModule, LitResNet, prepare_trainer_and_callbacks_pl

# Load configuration
constants = SystemConstants()
config = ExperimentConfig()

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

# Setup callbacks and model
callback_config = prepare_trainer_and_callbacks_pl(
    train_df_for_weights=train_df,
    constants=constants,
    config=config
)

model = LitResNet(
    constants=constants,
    config=config,
    class_weights_tensor=callback_config['class_weights']
)

# Ready for training with PyTorch Lightning Trainer
```

### Custom Configuration Example

```python
# Create custom configuration
custom_config = ExperimentConfig(
    learning_rate=0.01,
    batch_size=64,
    dropout_rate=0.3,
    fine_tune_layers_count=-2,  # Unfreeze last 2 layers
    use_custom_preprocessing=True,
    augmentation_strength=1.5
)

# Custom preprocessing
preprocessing_config = custom_config.get_custom_preprocessing_config()
# Returns: {'contrast_stretch': True, 'adaptive_hist': False, ...}
```

## Error Handling & Validation

### Comprehensive Validation
- **Configuration Validation**: All parameters validated on initialization
- **Data Validation**: CSV structure, image files, class distributions
- **Runtime Validation**: Graceful handling of missing files, corrupted data

### Error Recovery
- **Image Loading**: Invalid images filtered with detailed logging
- **Data Processing**: Fallback strategies for edge cases
- **Configuration**: Sensible defaults with override capabilities

## Testing Infrastructure

### Test Structure (Prepared for Development)
```
tests/
├── unit/
│   ├── entities/
│   ├── control/
│   └── utils/
├── integration/
│   └── data_pipeline/
├── fixtures/
│   └── sample_data.py
└── conftest.py
```

### Test Features
- **Parametrized Tests**: Multiple scenario coverage
- **Mock Components**: Isolated unit testing
- **Integration Tests**: End-to-end pipeline validation
- **Test Data Generation**: Realistic sample data creation

## Performance Optimizations

### Memory Management
- **Lazy Loading**: Images loaded only when needed
- **Memory Estimation**: Proactive memory usage calculation
- **Efficient Caching**: Strategic use of persistent workers

### Data Loading Optimizations
- **Pin Memory**: GPU transfer optimization
- **Prefetch Factor**: Configurable batch prefetching
- **Worker Processes**: Multi-process data loading

## Extensibility Features

### Plugin Architecture
- **Transform Builder**: Easy addition of new preprocessing methods
- **Custom Heads**: Configurable model architectures
- **Metric System**: Easy addition of new evaluation metrics

### Configuration Extensibility
- **YAML-First**: All parameters configurable via files
- **Environment Overrides**: Runtime configuration changes
- **Validation Framework**: Automatic parameter validation

## Development Guidelines Followed

### SOLID Principles
- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Extensible without modification
- **Dependency Inversion**: Interfaces over concrete implementations

### Code Quality
- **Type Hints**: Full type annotation coverage
- **Error Handling**: Comprehensive exception handling
- **Logging**: Detailed logging for debugging and monitoring
- **Documentation**: Complete docstrings and examples

## Phase 1 Deliverables ✅

### Core Infrastructure
- [x] Project structure and architecture
- [x] Configuration management system
- [x] Data processing pipeline
- [x] Image dataset management
- [x] Transform and preprocessing system

### Model Components
- [x] ResNet50 with custom head
- [x] PyTorch Lightning integration
- [x] Comprehensive metrics tracking
- [x] Training optimization setup

### Training System (NEW)
- [x] **Training callbacks system** with medical-focused metrics
- [x] **Centralized training orchestrator** with zip file processing
- [x] **Complete training pipeline** from data to trained model
- [x] **Command line interface** for easy deployment
- [x] **Comprehensive example system** with documentation

### Quality Assurance
- [x] Error handling and validation
- [x] Test infrastructure preparation
- [x] Performance optimizations
- [x] Documentation and examples
- [x] **Production-ready entry points** for training workflows

## Phase 1+ Enhancement: Complete Training System

**What's Been Added Beyond Original Phase 1 Scope:**

### Advanced Training Infrastructure
- **CentralizedTrainer**: Complete orchestration from zip file to trained model
- **TrainingCallbacks**: Medical-focused callback system with recall prioritization
- **Entry Points**: Both programmatic API and command-line interfaces
- **Example System**: Comprehensive documentation and usage examples

### Key Enhancements
```python
# Now possible in a single call:
trainer = CentralizedTrainer()
results = trainer.train_from_zip("pneumonia_dataset.zip")
```

This advancement effectively **bridges Phase 1 foundation with early Phase 2 capabilities**, providing:
- Complete centralized training workflow
- Production-ready training entry points
- Medical domain-specific optimizations (recall-focused callbacks)
- Comprehensive zip-based dataset processing

## Next Steps (Adjusted for Enhanced Phase 1)

With the enhanced training system, Phase 2 can now focus on:
- **Model Evaluation & Metrics**: Comprehensive post-training analysis
- **Visualization & Reporting**: Training results visualization
- **Advanced Training Strategies**: Hyperparameter tuning, cross-validation
- **Performance Benchmarking**: Systematic model comparison

---

**Status**: ✅ **COMPLETE** - All Phase 1 requirements implemented and validated
**Next Phase**: Phase 2 - Core ML Components (Centralized Training & Evaluation)