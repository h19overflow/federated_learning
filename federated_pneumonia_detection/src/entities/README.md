# Entities Module

**Purpose**: Domain models and data structures for the pneumonia detection system.

This module contains PyTorch nn.Module classes and PyTorch Dataset classes that define the neural network architecture and data loading pipeline for X-ray image classification.

## Table of Contents
- [Overview](#overview)
- [Components](#components)
- [Class Hierarchy](#class-hierarchy)
- [Integration Points](#integration-points)

---

## Overview

The entities module defines the **core domain models** used throughout the federated pneumonia detection system:

1. **ResNetWithCustomHead**: Neural network model for binary pneumonia classification
2. **CustomImageDataset**: PyTorch Dataset for loading and transforming X-ray images

These classes are framework-agnostic at the entity level but are integrated with PyTorch Lightning in the control layer.

---

## Components

### 1. ResNetWithCustomHead

**File**: [resnet_with_custom_head.py](resnet_with_custom_head.py)

**Purpose**: Binary classification neural network based on ResNet50 V2 with a custom classification head.

**Key Characteristics**:
- **Backbone**: ResNet50 V2 (ImageNet pretrained weights)
- **Head Architecture**: Global Average Pooling → Dense layers → Binary output
- **Layer Structure**: 2048 → 256 → 64 → 1 (configurable)
- **Regularization**: Configurable dropout (default 0.5)
- **Fine-tuning**: Supports freezing/unfreezing backbone layers
- **Error Handling**: Comprehensive parameter validation

**Key Methods**:
- `__init__()`: Initialize model with configuration and optional overrides (lines 26-93)
- `_validate_parameters()`: Validate initialization parameters (lines 95-100)
- `_create_backbone()`: Initialize ResNet50 backbone
- `_create_classifier_head()`: Build custom classification head
- `_configure_fine_tuning()`: Apply fine-tuning configuration
- `forward()`: Forward pass through model

**Configuration Parameters Used**:
- `experiment.dropout_rate`: Dropout probability (lines 60-64)
- `experiment.fine_tune_layers_count`: Number of layers to fine-tune (lines 65-69)

**Usage Context**:
- Instantiated in [centralized_trainer.py](../control/dl_model/centralized_trainer.py) for centralized training
- Wrapped in LitResNet (PyTorch Lightning module) in [lit_resnet.py](../control/dl_model/utils/model/lit_resnet.py)
- Used in federated learning clients (core/client_app.py)

---

### 2. CustomImageDataset

**File**: [custom_image_dataset.py](custom_image_dataset.py)

**Purpose**: PyTorch Dataset for efficient loading and transformation of X-ray images with robust error handling.

**Key Characteristics**:
- **Input Format**: Pandas DataFrame with filename and target columns
- **Image Sources**: Local file paths or ZIP-extracted directories
- **Color Modes**: Supports RGB and grayscale (L) images
- **Validation**: Image file existence checking with error recovery
- **Transformations**: Optional augmentation pipeline support
- **Error Handling**: Graceful recovery for missing/corrupted images

**Key Methods**:
- `__init__()`: Initialize dataset with validation (lines 27-90)
- `_validate_inputs()`: Validate dataframe and image directory (lines 92-100)
- `_validate_image_files()`: Check image file accessibility
- `__len__()`: Return dataset size
- `__getitem__()`: Load and transform single image with label
- `get_class_distribution()`: Return class counts

**Configuration Parameters Used**:
- `columns.filename`: Filename column name (line 66)
- `columns.target`: Target column name (line 67)

**Usage Context**:
- Created in [xray_data_module.py](../control/dl_model/utils/model/xray_data_module.py) (line 129)
- Used by [lit_resnet.py](../control/dl_model/utils/model/lit_resnet.py) during training/evaluation
- Data flows from: DataModule → DataLoader → Training/Validation steps

---

## Class Hierarchy

```
torch.nn.Module (PyTorch)
  └── ResNetWithCustomHead
       ├── ResNet50 V2 (backbone)
       └── Custom Classification Head

torch.utils.data.Dataset (PyTorch)
  └── CustomImageDataset
       ├── DataFrame management
       ├── Image loading
       └── Transformation pipeline
```

---

## Integration Points

### With Control Layer

| Component | Usage | Reference |
|-----------|-------|-----------|
| **LitResNet** | Wraps ResNetWithCustomHead for Lightning | [lit_resnet.py:30-50](../control/dl_model/utils/model/lit_resnet.py#L30-L50) |
| **XRayDataModule** | Creates CustomImageDataset instances | [xray_data_module.py:129-150](../control/dl_model/utils/model/xray_data_module.py#L129-L150) |
| **CentralizedTrainer** | Builds ResNetWithCustomHead model | [centralized_trainer.py:185-195](../control/dl_model/centralized_trainer.py#L185-L195) |
| **Federated Clients** | Initialize models for local training | [client_app.py:90-103](../control/federated_new_version/core/client_app.py#L90-L103) |

### With Utils Layer

| Component | Usage | Reference |
|-----------|-------|-----------|
| **image_transforms.py** | Provides transform pipelines | [image_transforms.py:219-380](../../utils/image_transforms.py#L219-L380) |
| **data_processing.py** | Prepares DataFrames for dataset | [data_processing.py:1-150](../../utils/data_processing.py#L1-L150) |

---

## Data Flow

```
CSV Metadata (stage2_train_metadata.csv)
          ↓
    DataProcessor (utils)
          ↓
    Pandas DataFrame (filename, Target)
          ↓
    CustomImageDataset (entities)
          ↓
    PyTorch DataLoader
          ↓
    LitResNet (wrapped ResNetWithCustomHead)
          ↓
    Training/Evaluation/Prediction
```

---

## Key Design Patterns

### Configuration Injection
Both classes accept optional `ConfigManager` for dependency injection:
```
ResNetWithCustomHead(config=config_manager)
CustomImageDataset(config=config_manager, dataframe=df)
```

### Error Handling
- **Parameter Validation**: `_validate_parameters()` and `_validate_inputs()`
- **Image Validation**: File existence checks with fallback behavior
- **Logging**: Structured logging via [logger.py](../../utils/loggers/logger.py)

### Type Hints
Full type hints throughout for IDE support and type checking:
- `Optional[ConfigManager]` for config parameter
- `Union[str, Path]` for file paths
- `Callable` for transformation functions

---

## Related Documentation

- **Model Training**: See [control/dl_model/utils/README.md](../control/dl_model/utils/README.md) for training pipeline
- **Data Processing**: See [utils/README.md](../../utils/README.md) for data loading utilities
- **Configuration**: See [config/README.md](../../config/README.md) for parameter management
