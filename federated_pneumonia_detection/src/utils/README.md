# Utils Layer

Shared utilities for data processing, image transforms, and logging across all layers.

## What It Does

```mermaid
flowchart TB
    subgraph Utils["Utils Layer"]
        DP[Data Processing]
        IT[Image Transforms]
        Log[Logging]
    end

    subgraph Consumers
        API[API Layer]
        Control[Control Layer]
        Boundary[Boundary Layer]
        Entities[Entities Layer]
    end

    Consumers --> Utils
    Utils --> Pandas[Pandas]
    Utils --> Torch[TorchVision]
    Utils --> PIL[Pillow]
```

## Module Overview

| Module | Purpose | Key Pattern |
|--------|---------|-------------|
| **data_processing.py** | Load CSV, stratified splits | Config-driven columns |
| **data_processing_functions.py** | Sampling, validation, stats | Stratified preservation |
| **image_transforms.py** | Augmentation pipelines | Builder pattern |
| **loggers/** | Structured logging | Third-party silencing |

## Data Processing

Two-file architecture: `DataProcessor` class orchestrates, functions provide utilities.

```mermaid
flowchart LR
    CSV[metadata.csv] --> Load[load_metadata]
    Load --> DF[DataFrame]
    DF --> Sample[sample_dataframe]
    Sample --> Split[create_train_val_split]
    Split --> Train[Train DF]
    Split --> Val[Val DF]

    style Split fill:#3498db,stroke:#2980b9,color:#fff
```

**Key Functions:**
| Function | Purpose |
|----------|---------|
| `load_metadata()` | CSV → DataFrame with filename column |
| `sample_dataframe()` | Stratified sampling (preserves class dist) |
| `create_train_val_split()` | 80/20 stratified split via sklearn |
| `validate_image_paths()` | Check directory + file existence |
| `get_data_statistics()` | Class counts, balance ratio |

**Stratified Split:**
```mermaid
flowchart LR
    subgraph Before["Original (100 samples)"]
        B1[70 Normal]
        B2[30 Pneumonia]
    end

    subgraph After["Split"]
        subgraph Train["Train (80)"]
            T1[56 Normal]
            T2[24 Pneumonia]
        end
        subgraph Val["Val (20)"]
            V1[14 Normal]
            V2[6 Pneumonia]
        end
    end

    Before --> After
```

## Image Transforms

Configurable augmentation with X-ray specific preprocessing:

```mermaid
flowchart TB
    subgraph Training["Training Pipeline"]
        RRC[RandomResizedCrop]
        RHF[RandomHorizontalFlip]
        RR[RandomRotation ±15°]
        CJ[ColorJitter]
        Norm1[ImageNet Normalize]
    end

    subgraph Validation["Val/Test Pipeline"]
        Resize[Resize 224]
        CC[CenterCrop]
        Norm2[ImageNet Normalize]
    end

    Image[Input Image] --> Training
    Image --> Validation

    style Training fill:#e74c3c,stroke:#c0392b,color:#fff
    style Validation fill:#27ae60,stroke:#1e8449,color:#fff
```

**Augmentation Strength:**
- `strength=0.0` → No augmentation
- `strength=1.0` → Moderate (default)
- `strength=2.0` → Aggressive

**X-Ray Preprocessing (Optional):**
| Method | Purpose |
|--------|---------|
| `contrast_stretch_percentile()` | 5-95th percentile normalization |
| `edge_enhancement()` | UnsharpMask for lung boundaries |
| `adaptive_histogram()` | CLAHE (requires cv2) |

**Normalization:**
- ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Alternative: [-1, 1] range

## Logging

Simple wrapper with third-party silencing:

```mermaid
flowchart LR
    App[Application] --> Config[configure_logging]
    Config --> Level[LOG_LEVEL env var]
    Config --> Silence[Silence Noisy Libs]

    Silence --> LC[langchain_google_genai]
    Silence --> GG[google.genai]
    Silence --> UV[uvicorn.access]

    style Silence fill:#f39c12,stroke:#d35400,color:#fff
```

**Configuration:**
| Setting | Value |
|---------|-------|
| Default level | INFO |
| Format | `LEVEL - MESSAGE - FILENAME - LINENO` |
| Third-party level | WARNING (silenced) |
| Handlers | Console only |

**Usage:**
```python
from utils.loggers import get_logger
logger = get_logger(__name__)
logger.info("Processing started")
```

## Transform Pipeline Detail

```mermaid
flowchart TB
    subgraph Builder["TransformBuilder"]
        BT[build_training_transforms]
        BV[build_validation_transforms]
        TTA[build_test_time_augmentation]
    end

    subgraph Train["Training Augmentation"]
        direction LR
        T1[RandomResizedCrop<br/>scale: 0.7-1.0]
        T2[HorizontalFlip<br/>p=0.5]
        T3[Rotation<br/>±15°]
        T4[ColorJitter<br/>brightness, contrast]
    end

    subgraph TTA_Detail["TTA (5 versions)"]
        direction LR
        TTA1[Original]
        TTA2[Rotate +10°]
        TTA3[Rotate -10°]
        TTA4[HFlip]
        TTA5[HFlip + Rotate]
    end

    BT --> Train
    TTA --> TTA_Detail
```

## Key Files

```
utils/
├── data_processing.py           # DataProcessor class
├── data_processing_functions.py # Standalone utilities
├── image_transforms.py          # TransformBuilder + XRayPreprocessor
└── loggers/
    ├── logger.py                # get_logger() wrapper
    └── logging_config.py        # configure_logging() + silencing
```

## Design Principles

```mermaid
flowchart LR
    subgraph Principles
        S[Stateless]
        I[Injectable]
        P[Pure Functions]
    end

    S --> NoState[No internal state]
    I --> Config[ConfigManager optional]
    P --> NoSideEffects[No side effects]

    style Principles fill:#9b59b6,stroke:#8e44ad,color:#fff
```

- **Stateless**: No internal state across calls
- **Injectable**: All components accept optional config
- **Pure**: Functions return values, no mutations
- **Fail-fast**: Validate inputs immediately

## Quick Reference

| Action | Function |
|--------|----------|
| Load CSV metadata | `load_metadata(path, config)` |
| Stratified split | `create_train_val_split(df, val_split=0.2)` |
| Sample preserving class dist | `sample_dataframe(df, frac=0.5)` |
| Get class counts | `get_data_statistics(df)` |
| Training transforms | `TransformBuilder(config).build_training_transforms()` |
| Validation transforms | `TransformBuilder(config).build_validation_transforms()` |
| Quick transform access | `get_transforms(mode='train', config=config)` |
| Get logger | `get_logger(__name__)` |
| Configure logging | `configure_logging()` |
