# Entities Layer

Domain models for the ML pipeline: dataset and neural network definitions.

## What It Does

```mermaid
flowchart LR
    subgraph Entities["Entities Layer"]
        Dataset[CustomImageDataset]
        Model[ResNetWithCustomHead]
    end

    CSV[CSV Metadata] --> Dataset
    Images[X-ray Images] --> Dataset
    Dataset --> DataLoader[PyTorch DataLoader]
    DataLoader --> Model
    Model --> Predictions[Predictions]
```

## Architecture

Two core entities following Clean Architecture principles:

```mermaid
classDiagram
    class Dataset {
        <<PyTorch>>
        +__len__()
        +__getitem__(idx)
    }

    class nn_Module {
        <<PyTorch>>
        +forward(x)
    }

    class CustomImageDataset {
        -dataframe
        -image_dir
        -valid_indices
        +get_class_distribution()
        +validate_all_images()
        +get_memory_usage_estimate()
    }

    class ResNetWithCustomHead {
        -features: ResNet50
        -classifier: Dense Head
        +freeze_backbone()
        +unfreeze_backbone()
        +get_feature_maps()
    }

    Dataset <|-- CustomImageDataset
    nn_Module <|-- ResNetWithCustomHead
```

## CustomImageDataset

Robust image loader with graceful error handling:

```mermaid
flowchart LR
    subgraph Input
        DF[DataFrame]
        Dir[Image Directory]
    end

    subgraph Validation
        Check[File Exists?]
        PIL[PIL Opens?]
    end

    subgraph Loading
        RGB[Convert to RGB]
        Mode[Apply Color Mode]
        Trans[Apply Transforms]
    end

    DF --> Check
    Dir --> Check
    Check --> PIL
    PIL -->|valid| RGB
    PIL -->|invalid| Skip[Skip & Log]
    RGB --> Mode --> Trans --> Tensor[Image Tensor]

    style Skip fill:#e74c3c,stroke:#c0392b,color:#fff
```

**Key Features:**
| Feature | Description |
|---------|-------------|
| Valid indices masking | Corrupted files excluded without re-indexing |
| Lazy transforms | Applied at `__getitem__` time |
| Memory estimation | Samples 10 images to estimate footprint |
| Class distribution | Counts per label for imbalance detection |

**Utils:**

- `image_ops.py` - PIL loading with RGB intermediate
- `stats.py` - Class distribution, memory estimation
- `validation.py` - Multi-stage file validation

## ResNetWithCustomHead

Transfer learning model with configurable fine-tuning:

```mermaid
flowchart TB
    subgraph Backbone["ResNet50 Backbone (ImageNet V2)"]
        Conv[Conv Layers]
        BN[BatchNorm]
        Res[Residual Blocks]
    end

    subgraph Head["Custom Classification Head"]
        Pool[AdaptiveAvgPool2d]
        D1[Dense 2048→256]
        Drop1[Dropout]
        D2[Dense 256→64]
        Drop2[Dropout]
        D3[Dense 64→1]
    end

    Input[224x224 Image] --> Backbone
    Backbone --> Head
    Head --> Sigmoid[Sigmoid]
    Sigmoid --> Output[Pneumonia Probability]

    style Backbone fill:#3498db,stroke:#2980b9,color:#fff
    style Head fill:#9b59b6,stroke:#8e44ad,color:#fff
```

**Key Features:**
| Feature | Description |
|---------|-------------|
| Progressive unfreezing | `fine_tune_layers_count=-4` unfreezes last 4 layers |
| Dropout adjustment | `set_dropout_rate()` modifies all dropout layers |
| Feature extraction | `get_feature_maps()` hooks intermediate layers |
| Model info | Parameter counts, architecture summary |

**Utils:**

- `model_builder.py` - Backbone/head creation functions
- `model_ops.py` - Freeze/unfreeze, dropout, feature hooks
- `validation.py` - Parameter validation

## Data Flow

```mermaid
sequenceDiagram
    participant CSV as Metadata CSV
    participant DS as CustomImageDataset
    participant DL as DataLoader
    participant Model as ResNetWithCustomHead

    CSV->>DS: DataFrame + image_dir
    DS->>DS: Validate images
    Note over DS: Filter invalid indices

    loop Training
        DL->>DS: __getitem__(batch_indices)
        DS->>DS: Load + transform images
        DS-->>DL: (image_tensor, label_tensor)
        DL->>Model: forward(batch)
        Model-->>DL: predictions
    end
```

## Key Files

```
entities/
├── custom_image_dataset.py           # PyTorch Dataset
├── custom_image_dataset_utils/
│   ├── image_ops.py                  # PIL loading
│   ├── stats.py                      # Class distribution, memory
│   └── validation.py                 # File validation
│
├── resnet_with_custom_head.py        # nn.Module
└── resnet_with_custom_head_utils/
    ├── model_builder.py              # Backbone + head creation
    ├── model_ops.py                  # Freeze/unfreeze, features
    └── validation.py                 # Parameter validation
```

## Design Principles

```mermaid
flowchart TB
    subgraph Entities["Entities Layer"]
        direction LR
        NoLogic[No Business Logic]
        DI[Dependency Injection]
        Validate[Self-Validation]
    end

    Control[Control Layer] --> Entities
    Entities --> Framework[PyTorch/Pandas]

    style Entities fill:#27ae60,stroke:#1e8449,color:#fff
```

- **Stateless**: No orchestration or side effects
- **Injectable**: Config passed at construction
- **Testable**: Mock config for isolated tests
- **Pure**: `forward()` and `__getitem__()` are functional

## Quick Reference

| Action                 | Method                                            |
| ---------------------- | ------------------------------------------------- |
| Create dataset         | `CustomImageDataset(df, image_dir, config)`       |
| Get class counts       | `dataset.get_class_distribution()`                |
| Validate all images    | `dataset.validate_all_images()`                   |
| Create model           | `ResNetWithCustomHead(config, num_classes=1)`     |
| Freeze backbone        | `model.freeze_backbone()`                         |
| Unfreeze last N layers | `ResNetWithCustomHead(fine_tune_layers_count=-N)` |
| Extract features       | `model.get_feature_maps(x, "layer4")`             |
