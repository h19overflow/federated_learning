# CentralizedTrainer Architecture - Chunked Class Diagrams

**Component**: `CentralizedTrainer` Training Pipeline  
**Domain**: Centralized Deep Learning for Pneumonia Detection  
**Entry Point**: `centralized_trainer.py:25-160`

---

## Chunk 1: Main Orchestrator & Entry Point

**Focus**: The `CentralizedTrainer` class and its immediate collaborators

```mermaid
classDiagram
    class CentralizedTrainer {
        +str checkpoint_dir
        +str logs_dir
        +Logger logger
        +ConfigManager config
        +DataSourceExtractor data_source_extractor
        +__init__(config_path, checkpoint_dir, logs_dir)
        +train(source_path, experiment_name, csv_filename, run_id) Dict
        +get_training_status() Dict
        -_load_config(config_path) ConfigManager
    }

    class DataSourceExtractor {
        +Logger logger
        +Path temp_extract_dir
        +extract_and_validate(source_path, csv_filename) Tuple[Path, Path]
    }

    class ConfigManager {
        +Dict _config
        +get(key, default) Any
        +get_section(section) Dict
    }

    class Logger {
        +info(msg)
        +warning(msg)
        +error(msg)
    }

    CentralizedTrainer --> DataSourceExtractor : uses (line 51)
    CentralizedTrainer --> ConfigManager : loads (line 48, 150)
    CentralizedTrainer --> Logger : uses (line 47)
```

### CentralizedTrainer Role

**File**: `centralized_trainer.py:25-160`

**Purpose**: **Facade Pattern** - Simplifies the complex training pipeline by providing a single `train()` method that orchestrates 9 distinct steps.

**Key Workflow** (train method, lines 53-131):

| Step | Line | Action | Delegates To |
|------|------|--------|--------------|
| 1 | 72-75 | Create/Use training run | `create_training_run()` |
| 2 | 78-81 | Extract & validate data source | `DataSourceExtractor` |
| 3 | 82-87 | Prepare train/val splits | `prepare_dataset()` |
| 4 | 88-94 | Create DataModule | `create_data_module()` |
| 5 | 95-103 | Build model + callbacks | `build_model_and_callbacks()` |
| 6 | 104-110 | Build PyTorch Lightning Trainer | `build_trainer()` |
| 7 | 111 | Execute training | `trainer.fit()` |
| 8 | 113-114 | Mark run complete | `complete_training_run()` |
| 9 | 116-125 | Collect & return results | `collect_training_results()` |

**Error Handling** (lines 127-131): Any exception triggers `fail_training_run()` to update database status before re-raising.

---

## Chunk 2: Data Layer - From Files to Tensors

**Focus**: Data extraction, dataset creation, and DataModule management

```mermaid
classDiagram
    class DataSourceExtractor {
        +Logger logger
        +Path temp_extract_dir
        +extract_and_validate(source_path, csv_filename) Tuple[Path, Path]
        -_extract_zip(zip_path) Path
        -_validate_directory(dir_path) Tuple[Path, Path]
    }

    class DataPrepUtils {
        +prepare_dataset(csv_path, image_dir, config, logger) Tuple[DataFrame, DataFrame]
        +create_data_module(train_df, val_df, image_dir, config, logger) XRayDataModule
    }

    class XRayDataModule {
        +DataFrame train_df
        +DataFrame val_df
        +Path image_dir
        +ConfigManager config
        +CustomImageDataset train_dataset
        +CustomImageDataset val_dataset
        +TransformBuilder transform_builder
        +__init__(train_df, val_df, config, image_dir)
        +setup(stage: str)
        +train_dataloader() DataLoader
        +val_dataloader() DataLoader
    }

    class CustomImageDataset {
        +ndarray filenames
        +ndarray labels
        +ndarray valid_indices
        +Path image_dir
        +Callable transform
        +__init__(dataframe, image_dir, config, transform)
        +__getitem__(idx) Tuple[Tensor, int]
        +__len__() int
        +validate_all_images() Tuple[List, List]
    }

    class TransformBuilder {
        +build_train_transforms() Compose
        +build_val_transforms() Compose
    }

    DataPrepUtils --> XRayDataModule : creates
    XRayDataModule --> CustomImageDataset : creates/manages
    XRayDataModule --> TransformBuilder : uses
    CustomImageDataset --> TransformBuilder : receives transforms
```

### Data Layer Roles

| Component | File | Role |
|-----------|------|------|
| **DataSourceExtractor** | `internals/data/data_source_extractor.py` | Handles ZIP extraction and directory validation. Returns `(image_dir, csv_path)`. |
| **DataPrepUtils** | `centralized_trainer_utils/data_prep.py` | Orchestrates data splitting and DataModule creation. |
| **XRayDataModule** | `internals/data/xray_data_module.py:30-238` | **PyTorch Lightning DataModule**. Manages dataset lifecycle and DataLoader configuration. |
| **CustomImageDataset** | `entities/custom_image_dataset.py:34-206` | **PyTorch Dataset**. Low-level data access with corruption handling. |
| **TransformBuilder** | `internals/image_transforms.py` | Creates augmentation pipelines (rotation, flip, normalize). |

**Why `valid_indices` in CustomImageDataset?**
- Some images may be corrupted or missing
- Dataset filters these at initialization
- Maintains mapping: "dataset index" → "DataFrame index"
- Prevents training crashes from bad files

---

## Chunk 3: Model Layer - Neural Network & Training Logic

**Focus**: Model architecture, Lightning module, and optimization

```mermaid
classDiagram
    class LitResNetEnhanced {
        +ResNetWithCustomHead model
        +ConfigManager config
        +MetricsHandler metrics_handler
        +StepLogic step_logic
        +LossFactory loss_factory
        +bool use_focal_loss
        +__init__(config, class_weights_tensor, use_focal_loss)
        +forward(x) Tensor
        +training_step(batch, batch_idx) Tensor
        +validation_step(batch, batch_idx) Dict
        +test_step(batch, batch_idx) Dict
        +configure_optimizers() Dict
        +progressive_unfreeze(layers_to_unfreeze)
    }

    class ResNetWithCustomHead {
        +nn.Module features
        +nn.Module classifier
        +int num_classes
        +float dropout_rate
        +int fine_tune_layers_count
        +__init__(config, num_classes, dropout_rate, fine_tune_layers_count)
        +forward(x) Tensor
        +freeze_backbone()
        +unfreeze_backbone()
        +_unfreeze_last_n_layers(n_layers)
    }

    class OptimizerFactory {
        +create_configuration(params, config, monitor_metric, use_cosine_scheduler) Dict
        -_create_optimizer(params, config) Optimizer
        -_create_scheduler(optimizer, config, monitor_metric) Scheduler
    }

    class MetricsHandler {
        +Accuracy accuracy
        +Precision precision
        +Recall recall
        +F1Score f1
        +AUROC auroc
        +compute(predictions, targets) Dict
    }

    class StepLogic {
        +execute_training_step(batch, model, loss_fn) Tensor
        +execute_validation_step(batch, model, metrics_handler) Dict
    }

    class LossFactory {
        +create_loss(class_weights, use_focal_loss, focal_alpha, focal_gamma) nn.Module
    }

    LitResNetEnhanced --> ResNetWithCustomHead : wraps (composition)
    LitResNetEnhanced --> OptimizerFactory : uses in configure_optimizers()
    LitResNetEnhanced --> MetricsHandler : uses for metrics
    LitResNetEnhanced --> StepLogic : uses for step execution
    LitResNetEnhanced --> LossFactory : uses for loss function
```

### Model Layer Roles

| Component | File | Role |
|-----------|------|------|
| **LitResNetEnhanced** | `internals/model/lit_resnet_enhanced.py:28-182` | **PyTorch Lightning Module**. Wraps the neural network with training logic, loss computation, and metrics. |
| **ResNetWithCustomHead** | `entities/resnet_with_custom_head.py:40-223` | **Pure PyTorch nn.Module**. The actual neural network. Lives in Entities layer (no framework deps). |
| **OptimizerFactory** | `internals/model/optimizers/factory.py:14-119` | **Factory Pattern**. Creates optimizer + scheduler based on config. |
| **MetricsHandler** | `internals/model/utils/metrics_handler.py` | Computes accuracy, precision, recall, F1, AUROC using torchmetrics. |
| **StepLogic** | `internals/model/utils/step_logic.py` | Encapsulates training/validation step execution logic. |
| **LossFactory** | `internals/model/utils/loss_factory.py` | Creates focal loss or weighted BCE loss. |

**Architecture Flow**:
```
Input (3×256×256)
    ↓
ResNet50 Backbone (pretrained)
    ↓
Global Average Pooling
    ↓
Dense (2048→512) + ReLU
    ↓
Dropout (0.5)
    ↓
Dense (512→1) + Sigmoid
    ↓
Output (pneumonia probability)
```

**Transfer Learning Support**:
- `freeze_backbone()`: Train only classification head
- `unfreeze_backbone()`: Fine-tune entire network
- `progressive_unfreeze()`: Gradually unfreeze layers during training

---

## Chunk 4: Callbacks System - Training Lifecycle Hooks

**Focus**: Callbacks that hook into the training loop for checkpointing, metrics, and monitoring

```mermaid
classDiagram
    class CallbacksSetup {
        +prepare_trainer_and_callbacks_pl(train_df, config, checkpoint_dir) Dict
        +create_trainer_from_config(config, callbacks, is_federated) Trainer
        +compute_class_weights_for_pl(train_df) Tensor
    }

    class MetricsCollectorCallback {
        +Path save_dir
        +int run_id
        +MetricsFilePersister file_persister
        +MetricsWebSocketSender ws_sender
        +List epoch_metrics
        +on_train_start(trainer, pl_module)
        +on_train_epoch_end(trainer, pl_module)
        +on_validation_epoch_end(trainer, pl_module)
        +on_fit_end(trainer, pl_module)
        +_extract_metrics(trainer, pl_module, stage) Dict
    }

    class ModelCheckpoint {
        +str dirpath
        +str filename
        +str monitor = "val_recall"
        +str mode = "max"
        +int save_top_k = 3
        +on_validation_epoch_end(trainer, pl_module)
    }

    class EarlyStopping {
        +str monitor = "val_recall"
        +int patience
        +float min_delta
        +on_validation_epoch_end(trainer, pl_module)
    }

    class ProgressiveUnfreezeCallback {
        +List unfreeze_epochs
        +int layers_per_unfreeze
        +on_epoch_start(trainer, pl_module)
    }

    class EarlyStoppingSignalCallback {
        +on_validation_epoch_end(trainer, pl_module)
    }

    class LearningRateMonitor {
        +logging_interval
        +on_train_epoch_start(trainer, pl_module)
    }

    CallbacksSetup --> MetricsCollectorCallback : creates
    CallbacksSetup --> ModelCheckpoint : creates
    CallbacksSetup --> EarlyStopping : creates
    CallbacksSetup --> ProgressiveUnfreezeCallback : creates
    CallbacksSetup --> EarlyStoppingSignalCallback : creates
    CallbacksSetup --> LearningRateMonitor : creates
```

### Callbacks Chain (Order Matters!)

**File**: `internals/model/callbacks/setup.py:78-244`

| Index | Callback | Purpose | Trigger |
|-------|----------|---------|---------|
| 0 | `ModelCheckpoint` | Save best 3 models by `val_recall` | After validation |
| 1 | `EarlyStopping` | Stop training if no improvement for N epochs | After validation |
| 2 | `EarlyStoppingSignalCallback` | Notify frontend via WebSocket | After early stopping |
| 3 | `LearningRateMonitor` | Log learning rate each epoch | Epoch start |
| 4 | `HighestValRecallCallback` | Track best recall value | After validation |
| 5 | `MetricsCollectorCallback` | Persist metrics to DB/files/WebSocket | Epoch end |
| 6 | `BatchMetricsCallback` | Real-time batch-level metrics | Each batch |
| 7 | `GradientMonitorCallback` | Track gradient norms | After backward |

### MetricsCollectorCallback Details

**File**: `internals/model/collectors/metrics.py:20-510`

**Multi-Channel Output**:
```
┌─────────────────────────────────────┐
│  MetricsCollectorCallback           │
├─────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐ │
│  │ File Persister│  │ WebSocket    │ │
│  │ • JSON        │  │ • Real-time  │ │
│  │ • CSV         │  │   streaming  │ │
│  └──────────────┘  └──────────────┘ │
│  ┌──────────────┐                   │
│  │ Database CRUD │                  │
│  │ • run_metrics │                  │
│  │ • run table   │                  │
│  └──────────────┘                   │
└─────────────────────────────────────┘
```

**Key Lifecycle Hooks**:
- `on_train_start`: Create DB run record, send start event
- `on_train_epoch_end`: Collect training loss/accuracy
- `on_validation_epoch_end`: Collect val metrics, send WebSocket update
- `on_fit_end`: Save all metrics, send completion event

---

## Chunk 5: Utility Modules - Helper Functions

**Focus**: The utility functions that CentralizedTrainer delegates to

```mermaid
classDiagram
    class ModelSetupUtils {
        +build_model_and_callbacks(train_df, config, checkpoint_dir, logs_dir, logger, experiment_name, run_id) Tuple
        +build_trainer(config, callbacks, logs_dir, experiment_name, logger) Trainer
    }

    class DataPrepUtils {
        +prepare_dataset(csv_path, image_dir, config, logger) Tuple[DataFrame, DataFrame]
        +create_data_module(train_df, val_df, image_dir, config, logger) XRayDataModule
    }

    class DBOperationsUtils {
        +create_training_run(source_path, experiment_name, logger) int
        +complete_training_run(run_id, logger)
        +fail_training_run(run_id, logger)
    }

    class ResultsUtils {
        +collect_training_results(trainer, model, metrics_collector, logs_dir, checkpoint_dir, logger, run_id) Dict
    }

    class MetricsFilePersister {
        +save_metrics_to_json(metrics, path)
        +save_metrics_to_csv(metrics, path)
    }

    class MetricsWebSocketSender {
        +send_metrics(metrics)
        +send_training_complete()
        +send_training_failed(error)
    }

    ModelSetupUtils --> CallbacksSetup : uses
    ModelSetupUtils --> LitResNetEnhanced : creates
    ResultsUtils --> MetricsFilePersister : uses
    MetricsCollectorCallback --> MetricsFilePersister : uses
    MetricsCollectorCallback --> MetricsWebSocketSender : uses
```

### Utility Roles

| Module | File | Key Functions | Purpose |
|--------|------|---------------|---------|
| **ModelSetupUtils** | `centralized_trainer_utils/model_setup.py:24-106` | `build_model_and_callbacks()`, `build_trainer()` | Creates model with all callbacks and trainer |
| **DataPrepUtils** | `centralized_trainer_utils/data_prep.py:18-83` | `prepare_dataset()`, `create_data_module()` | Data splitting and DataModule creation |
| **DBOperationsUtils** | `centralized_trainer_utils/db_operations.py` | `create_training_run()`, `complete_training_run()`, `fail_training_run()` | Database lifecycle management |
| **ResultsUtils** | `centralized_trainer_utils/results.py` | `collect_training_results()` | Packages training results into dictionary |
| **MetricsFilePersister** | `internals/data/metrics_file_persister.py` | `save_metrics_to_json()`, `save_metrics_to_csv()` | File-based metrics persistence |
| **MetricsWebSocketSender** | `internals/data/websocket_metrics_sender.py` | `send_metrics()`, `send_training_complete()` | Real-time frontend updates |

---

## Chunk 6: Complete Architecture Overview

**Focus**: How all chunks connect together in the Clean Architecture pattern

```mermaid
flowchart TB
    subgraph API["API Layer (FastAPI)"]
        Endpoint["POST /experiments/centralized"]
    end

    subgraph Control["CONTROL LAYER (Business Logic)"]
        CT["CentralizedTrainer"]
        MS["ModelSetupUtils"]
        DP["DataPrepUtils"]
        DBU["DBOperationsUtils"]
        RU["ResultsUtils"]
        OF["OptimizerFactory"]
        CB["CallbacksSetup"]
    end

    subgraph Entities["ENTITIES LAYER (Domain Models)"]
        RN["ResNetWithCustomHead"]
        DS["CustomImageDataset"]
        Config["ConfigManager"]
    end

    subgraph Boundary["BOUNDARY LAYER (Interface Adapters)"]
        DSE["DataSourceExtractor"]
        RunCRUD["Run CRUD"]
        MetricsCRUD["RunMetrics CRUD"]
    end

    subgraph Frameworks["Framework Integrations"]
        PL["PyTorch Lightning<br/>Trainer, Callbacks"]
        WS["WebSocket Sender"]
        File["File System"]
    end

    Endpoint --> CT
    CT --> MS
    CT --> DP
    CT --> DBU
    CT --> RU
    CT --> DSE
    CT --> Config

    MS --> CB
    MS --> OF
    MS --> RN
    CB --> PL
    OF --> PL

    DP --> DS
    DS --> Config

    DBU --> RunCRUD
    RU --> MetricsCRUD
    RU --> File
    CB --> WS
```

### Clean Architecture (ECB) Mapping

| Layer | Components in This Pipeline |
|-------|----------------------------|
| **API** | FastAPI endpoint calls `CentralizedTrainer.train()` |
| **Control** | `CentralizedTrainer`, all `*Utils` modules, `CallbacksSetup`, `OptimizerFactory` |
| **Entities** | `ResNetWithCustomHead`, `CustomImageDataset`, `ConfigManager` (pure domain) |
| **Boundary** | `DataSourceExtractor`, Database CRUD operations, `MetricsFilePersister` |

**Dependency Rule**: Dependencies point inward. Control depends on Entities and Boundary interfaces, not concrete implementations.

---

## File Reference Summary

| Chunk | Primary File | Supporting Files |
|-------|-------------|------------------|
| **1 - Orchestrator** | `centralized_trainer.py:25-160` | `centralized_trainer_utils/__init__.py` |
| **2 - Data Layer** | `internals/data/xray_data_module.py` | `entities/custom_image_dataset.py`, `internals/data/data_source_extractor.py`, `centralized_trainer_utils/data_prep.py` |
| **3 - Model Layer** | `internals/model/lit_resnet_enhanced.py` | `entities/resnet_with_custom_head.py`, `internals/model/optimizers/factory.py` |
| **4 - Callbacks** | `internals/model/callbacks/setup.py` | `internals/model/collectors/metrics.py`, `internals/model/callbacks/progressive.py` |
| **5 - Utilities** | `centralized_trainer_utils/*.py` | `internals/data/metrics_file_persister.py`, `internals/data/websocket_metrics_sender.py` |

---

## Related Documentation

- **Control Layer Guide**: [`src/control/AGENTS.md`](../../../src/control/AGENTS.md)
- **Full Class Diagram** (single page): [`centralized_trainer.md`](./centralized_trainer.md)
- **Entities Layer**: [`src/entities/README.md`](../../../src/entities/README.md)
