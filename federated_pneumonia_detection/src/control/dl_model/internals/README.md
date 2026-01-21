# Model & Training Utilities

**Purpose**: Core deep learning training infrastructure including PyTorch Lightning modules, metrics collection, data loading, and training callbacks.

This module orchestrates the training pipeline for both centralized learning and serves as the foundation for federated learning clients.

## Table of Contents

- [Overview](#overview)
- [Core Components](#core-components)
- [Training Pipeline](#training-pipeline)
- [Data Management](#data-management)
- [Metrics & Monitoring](#metrics--monitoring)
- [Integration](#integration)

---

## Overview

The dl_model/utils module contains four key subdirectories:

1. **model/**: PyTorch Lightning modules and metrics tracking
2. **data/**: Data extraction, persistence, and WebSocket communication
3. **callbacks/**: Training callbacks and configuration

These components work together to:

- Load X-ray images and prepare training data
- Define the neural network (ResNet50 + custom head)
- Execute training with PyTorch Lightning
- Collect and persist metrics in real-time
- Stream training progress to frontend

---

## Core Components

### 1. PyTorch Lightning Model (`model/lit_resnet.py`)

**File**: [model/lit_resnet.py](model/lit_resnet.py)

**Class**: `LitResNet` (extends `pl.LightningModule`)

**Purpose**: Wraps ResNetWithCustomHead in PyTorch Lightning framework for automated training.

**Key Methods**:

| Method                      | Purpose                                                          | Reference     |
| --------------------------- | ---------------------------------------------------------------- | ------------- |
| `_setup_metrics()`          | Initialize torchmetrics (Accuracy, Precision, Recall, F1, AUROC) | lines 101-123 |
| `training_step()`           | Single training batch forward pass                               | lines 183-202 |
| `validation_step()`         | Validation batch processing                                      | lines 203-228 |
| `test_step()`               | Test set evaluation                                              | lines 230-255 |
| `predict_step()`            | Inference without labels                                         | lines 257-270 |
| `configure_optimizers()`    | Setup AdamW + ReduceLROnPlateau scheduler                        | lines 272-310 |
| `on_validation_epoch_end()` | Compute and log epoch metrics                                    | lines 312-330 |

**Metrics Tracked**:

- Training: loss, accuracy, F1
- Validation: loss, accuracy, precision, recall, F1, AUROC
- Test: same as validation

**Loss Function**: `BCEWithLogitsLoss` with optional class weighting

**Optimizer**: AdamW with configurable learning rate and weight decay

**Scheduler**: ReduceLROnPlateau (monitors validation recall)

**Configuration Parameters**:

- `experiment.learning_rate`: Learning rate (default 0.001)
- `experiment.weight_decay`: L2 regularization (default 0.0001)
- `experiment.dropout_rate`: Dropout in custom head
- `experiment.fine_tune_layers_count`: Layers to fine-tune

---

### 2. Data Module (`model/xray_data_module.py`)

**File**: [model/xray_data_module.py](model/xray_data_module.py)

**Class**: `XRayDataModule` (extends `pl.LightningDataModule`)

**Purpose**: Manages data loading and batching for training/validation/test sets.

**Key Methods**:

| Method               | Purpose                            | Reference     |
| -------------------- | ---------------------------------- | ------------- |
| `setup()`            | Create datasets and data loaders   | lines 129-200 |
| `train_dataloader()` | Return training DataLoader         | lines 216-225 |
| `val_dataloader()`   | Return validation DataLoader       | lines 227-235 |
| `test_dataloader()`  | Return test DataLoader             | lines 237-245 |
| `_create_dataset()`  | Create CustomImageDataset instance | lines 248-300 |

**Configuration Parameters**:

- `system.batch_size`: Batch size (default 32)
- `system.num_workers`: DataLoader workers (default 0)
- `system.pin_memory`: Pin memory for GPU (default true)
- `system.persistent_workers`: Keep workers alive (default false)
- `system.prefetch_factor`: Batch prefetch count (default 2)
- `system.validation_split`: Train/val ratio (default 0.2)

**Features**:

- Automatic train/val/test splitting
- Class distribution awareness
- Configurable augmentation
- Memory efficiency with worker management

---

### 3. Metrics Collection (`model/metrics_collector.py`)

**File**: [model/metrics_collector.py](model/metrics_collector.py)

**Class**: `MetricsCollectorCallback` (extends `pl.Callback`)

**Purpose**: Collect, persist, and stream training metrics to database and frontend.

**Key Methods**:

| Method                      | Purpose                              | Reference     |
| --------------------------- | ------------------------------------ | ------------- |
| `on_train_start()`          | Initialize metrics collection        | lines 101-115 |
| `on_train_epoch_end()`      | Collect training metrics per epoch   | lines 164-177 |
| `on_validation_epoch_end()` | Collect validation metrics per epoch | lines 178-200 |
| `on_fit_end()`              | Finalize and persist all metrics     | lines 203-210 |
| `_extract_metrics()`        | Extract logged metrics from trainer  | lines 263-310 |
| `persist_to_database()`     | Save metrics to PostgreSQL           | lines 425-481 |

**Features**:

- **Real-time WebSocket streaming**: Send metrics to frontend as training progresses
- **Database persistence**: Store metrics in run_metrics table with federated context
- **File export**: Save metrics to JSON and CSV files
- **Federated support**: Track client_id and round_number
- **Error handling**: Graceful degradation on WebSocket failures

**Data Persisted**:

- Per-epoch: train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auroc
- Metadata: learning_rate, training duration, best_epoch, model_params

---

### 4. Training Callbacks (`model/training_callbacks.py`)

**File**: [model/training_callbacks.py](model/training_callbacks.py)

**Key Functions/Classes**:

| Component                            | Purpose                                   | Reference     |
| ------------------------------------ | ----------------------------------------- | ------------- |
| `EarlyStoppingSignalCallback`        | Detect early stopping and notify frontend | lines 8-45    |
| `HighestValRecallCallback`           | Track highest validation recall           | lines 47-80   |
| `compute_class_weights_for_pl()`     | Compute balanced class weights            | lines 82-110  |
| `prepare_trainer_and_callbacks_pl()` | Create trainer with all callbacks         | lines 112-200 |

**Callbacks Created**:

1. **ModelCheckpoint**: Save best models based on validation recall (top 3 + last)
2. **EarlyStopping**: Stop if validation recall plateaus (patience=7)
3. **LearningRateMonitor**: Track learning rate changes
4. **MetricsCollector**: Custom metrics collection callback
5. **EarlyStoppingSignal**: Notify frontend when training stops early

---

### 5. Data Extraction (`data/data_source_handler.py`)

**File**: [data/data_source_handler.py](data/data_source_handler.py)

**Class**: `DataSourceExtractor`

**Purpose**: Extract and validate training datasets from ZIP files or directories.

**Key Methods**:

- `extract_and_validate()`: Extract ZIP, validate contents, return metadata path
- `validate_contents()`: Validate without processing
- `cleanup()`: Clean up temporary extraction directories

---

### 6. Metrics Persistence (`data/metrics_file_persister.py`)

**File**: [data/metrics_file_persister.py](data/metrics_file_persister.py)

**Class**: `MetricsFilePersister`

**Purpose**: Save collected metrics to JSON and CSV files.

**Methods**:

- `save_metrics()`: Persist metrics dictionary to files
- Output format: `metrics_{run_id}.json`, `metrics_{run_id}.csv`

---

### 7. WebSocket Streaming (`data/websocket_metrics_sender.py`)

**File**: [data/websocket_metrics_sender.py](data/websocket_metrics_sender.py)

**Class**: `MetricsWebSocketSender`

**Purpose**: Stream training metrics to frontend in real-time via WebSocket.

**Key Methods**:

| Method                 | Purpose                            | Reference     |
| ---------------------- | ---------------------------------- | ------------- |
| `send_training_mode()` | Signal federated/centralized mode  | lines 201-222 |
| `send_round_metrics()` | Send round-end metrics (federated) | lines 224-245 |
| `send_metrics()`       | Generic method for any metrics     | lines 44-75   |
| `send_training_end()`  | Signal training completion         | lines 27-42   |

**Connection**:

- WebSocket URL: `ws://localhost:8765`
- Auto-reconnect with exponential backoff (max 10 attempts)
- Ping/pong keepalive mechanism

---

## Training Pipeline

```
1. DATA PREPARATION
   ├─ DataSourceExtractor
   │   ├─ Extract ZIP (if provided)
   │   └─ Validate contents (metadata.csv + images/)
   └─ Return: metadata_path, image_directory

2. DATASET CREATION
   ├─ XRayDataModule.setup()
   │   ├─ Load metadata CSV
   │   ├─ Train/val split (80/20 stratified)
   │   └─ Create CustomImageDataset instances
   └─ Return: DataLoaders for train/val/test

3. MODEL INITIALIZATION
   ├─ Build ResNetWithCustomHead
   ├─ Wrap in LitResNet
   └─ Attach metrics (Accuracy, Precision, Recall, F1, AUROC)

4. CALLBACK SETUP
   ├─ ModelCheckpoint (save best by validation_recall)
   ├─ EarlyStopping (patience=7 on validation_recall)
   ├─ LearningRateMonitor
   ├─ MetricsCollector (this module)
   └─ EarlyStoppingSignal (notify frontend)

5. TRAINING EXECUTION
   ├─ PyTorch Lightning Trainer.fit()
   ├─ For each epoch:
   │   ├─ training_step() → compute loss, update metrics
   │   ├─ validation_step() → validate, compute metrics
   │   ├─ on_train_epoch_end() → collect training metrics
   │   ├─ on_validation_epoch_end() → collect validation metrics
   │   ├─ Log metrics to PyTorch Lightning (captured by callbacks)
   │   ├─ MetricsCollector extracts and streams
   │   └─ WebSocketSender broadcasts to frontend
   └─ Loop until early stopping or max_epochs

6. RESULTS PERSISTENCE
   ├─ Metrics saved to JSON/CSV files
   ├─ Metrics persisted to database (run_metrics table)
   └─ Training end signal sent via WebSocket
```

---

## Data Flow

```
Raw Image Files (PNG/JPG)
         ↓
   CustomImageDataset
         ↓
   XRayDataModule
         ↓
   PyTorch DataLoader
         ↓
   LitResNet.training_step()
    ├─ Forward pass (ResNetWithCustomHead)
    ├─ Loss computation (BCEWithLogitsLoss)
    ├─ Backward pass (gradient update)
    └─ Metrics update (Accuracy, Precision, Recall, F1, AUROC)
         ↓
   Trainer callback system
    ├─ on_train_epoch_end()
    ├─ on_validation_epoch_end()
    └─ log() methods
         ↓
   MetricsCollectorCallback
    ├─ Extract logged metrics
    ├─ Send to WebSocket
    └─ Persist to database
         ↓
   Frontend Dashboard
    └─ Real-time progress display
```

---

## Metrics Monitoring

### Real-Time Streaming (During Training)

**WebSocket Messages**:

1. `training_start`: Initial signal with run_id
2. `epoch_end`: Per-epoch metrics (loss, accuracy, etc.)
3. `training_end`: Completion signal with final status

**Frontend Reception**:

- TrainingExecution component receives messages
- Updates progress bar, training log, metrics cards
- Live updates as training progresses

### Post-Training Access

**Database**:

- Query run_metrics table for complete training history
- Aggregate per-epoch metrics for visualization

**API**:

- `/runs/{run_id}/metrics` endpoint returns all metrics
- Used by frontend ResultsVisualization component

---

## Configuration

**Key Parameters** (from config/default_config.yaml):

```yaml
experiment:
  learning_rate: 0.001
  epochs: 10
  batch_size: 32
  weight_decay: 0.0001
  dropout_rate: 0.3
  early_stopping_patience: 7
  reduce_lr_patience: 3
  monitor_metric: val_loss

system:
  batch_size: 32
  num_workers: 0
  pin_memory: true
  validation_split: 0.2
```

---

## Integration Points

### With Centralized Trainer

[centralized_trainer.py](../centralized_trainer.py) coordinates:

1. Data extraction
2. DataModule creation
3. Model + callbacks initialization
4. Trainer execution
5. Results collection

### With Federated Learning

Clients use same components:

- [client_app.py](../../federated_new_version/core/client_app.py): Builds trainer locally (lines 113-126)
- Same metrics tracking on client side
- Federated context injected (client_id, round_number)

### With Frontend

[websocket.ts](../../../../../xray-vision-ai-forge/src/services/websocket.ts):

- Connects to WebSocket server
- Receives streaming metrics
- Updates TrainingExecution component

---

## Error Handling

All components include:

- Type hints for IDE support
- Structured logging via logger.py
- Specific exception types
- Graceful degradation (e.g., WebSocket failures)
- Parameter validation

---

## Related Documentation

- **Model Architecture**: See [entities/README.md](../../entities/README.md) for ResNetWithCustomHead
- **Data Processing**: See [utils/README.md](../../utils/README.md) for transformations
- **Centralized Training**: See [control/dl_model/README.md](../README.md)
- **Federated Learning**: See [control/federated_new_version/README.md](../../federated_new_version/README.md)
- **Database**: See [boundary/README.md](../../boundary/README.md) for metrics persistence
- **System Overview**: See [README.md](../../../../README.md)
