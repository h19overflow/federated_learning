# Federated Learning Module

**Purpose**: Implements Flower-based federated learning for distributed training of the pneumonia detection model across multiple clients.

This module orchestrates server-client communication, model aggregation, and centralized evaluation while maintaining data privacy through local-only training.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Federated Learning Workflow](#federated-learning-workflow)
- [Data Partitioning](#data-partitioning)
- [Key Components](#key-components)
- [Configuration](#configuration)
- [Integration](#integration)

---

## Overview

**Flower Framework**: Federated learning using [Flower (flwr.dev)](https://flower.dev)

**Key Characteristics**:
- **Decentralized Training**: Each client trains locally on its data partition
- **Centralized Aggregation**: Server aggregates client updates via FedAvg
- **Server Evaluation**: Global model tested on server-side held-out dataset
- **Privacy-Preserving**: Raw data never leaves client machines
- **Configuration-Driven**: Rounds, epochs, clients controlled via config/pyproject.toml

---

## Architecture

### Client/Server Model

```
┌─────────────────────────────────────────────────────┐
│ Flower Server (orchestrator & aggregator)          │
│ ┌─────────────────────────────────────────────────┐│
│ │ ServerApp: Manages FL rounds                    ││
│ ├─ Initialize global model                       ││
│ ├─ Configure clients each round                  ││
│ ├─ Aggregate client updates (FedAvg)             ││
│ ├─ Evaluate global model on server test set      ││
│ └─ Persist metrics to database                   ││
│ ┌─────────────────────────────────────────────────┐│
│ │ Strategy: ConfigurableFedAvg                    ││
│ ├─ Weighted aggregation (weight = num_examples)  ││
│ ├─ Flexible configuration passing                ││
│ └─ Round metrics broadcasting                     ││
└─────────────────────────────────────────────────────┘
                         ↕ (Round N)
    ┌────────────────┬────────────────────┬──────────────────┐
    │ Client 0       │ Client 1           │ Client K         │
    ├────────────────┼────────────────────┼──────────────────┤
    │ Partition 0    │ Partition 1        │ Partition K      │
    │ (IID data)     │ (IID data)         │ (IID data)       │
    │                │                    │                  │
    │ ┌────────────┐ │ ┌────────────┐     │ ┌────────────┐  │
    │ │ ClientApp  │ │ │ ClientApp  │ ... │ │ ClientApp  │  │
    │ ├─ Load      │ │ ├─ Load      │     │ ├─ Load      │  │
    │ │   global   │ │ │   global   │     │ │   global   │  │
    │ │ Train      │ │ │ Train      │     │ │ Train      │  │
    │ │   local    │ │ │   local    │     │ │   local    │  │
    │ │ Evaluate   │ │ │ Evaluate   │     │ │ Evaluate   │  │
    │ │ Return: W, │ │ │ Return: W, │     │ │ Return: W, │  │
    │ │   metrics  │ │ │   metrics  │ ... │ │   metrics  │  │
    │ └────────────┘ │ └────────────┘     │ └────────────┘  │
    └────────────────┴────────────────────┴──────────────────┘
```

---

## Federated Learning Workflow

### Round-by-Round Process

**File**: [core/server_app.py](core/server_app.py)

**Round Sequence**:

```
ROUND N
├─1. SERVER: Prepare global model weights
│
├─2. SERVER → CLIENTS: Send global weights + config (train_cfg)
│   ├─ Global model (as ArrayRecord)
│   ├─ Data paths, seeds for reproducibility
│   ├─ Configuration (epochs, batch_size, etc.)
│   └─ Run ID for database tracking
│
├─3. CLIENTS: Local training (parallel)
│   ├─ Client 0:
│   │  ├─ Load global model from server
│   │  ├─ Train on partition 0 (local epochs)
│   │  ├─ Compute training metrics
│   │  └─ Return: Updated weights + training metrics
│   ├─ Client 1: Same process on partition 1
│   └─ Client K: Same process on partition K
│
├─4. SERVER: Receive all client updates
│   ├─ Aggregate model weights (FedAvg)
│   ├─ Aggregate metrics (weighted by num_examples)
│   └─ Log aggregated metrics
│
├─5. SERVER → CLIENTS: Send aggregated weights for evaluation
│
├─6. CLIENTS: Evaluate global model (local validation sets)
│   ├─ Test aggregated weights on local validation data
│   └─ Return: Evaluation metrics
│
├─7. SERVER: Aggregate evaluation metrics
│
├─8. SERVER: Centralized evaluation
│   ├─ Evaluate aggregated model on server-held test set
│   ├─ Compute: loss, accuracy, precision, recall, f1, auroc
│   ├─ Compute: confusion matrix (TP, TN, FP, FN)
│   └─ Persist to ServerEvaluation table
│
├─9. METRICS: Database + WebSocket
│   ├─ Store metrics in run_metrics table
│   ├─ Broadcast via WebSocket to frontend
│   └─ Update UI in real-time
│
└─10. REPEAT: For next round (until num_server_rounds)
```

---

## Data Partitioning

### IID Partition Strategy

**File**: [partioner.py](partioner.py)

**Process**:
1. Load full dataset metadata (all patient IDs and labels)
2. Randomly shuffle all indices (seed-controlled)
3. Split evenly into `num_partitions` (e.g., 2 clients = 2 equal parts)
4. Each client loads its partition deterministically by node_id

**Example** (100 samples, 2 clients):
- Partition 0 (Client 0): Samples 0-49 (50 samples)
- Partition 1 (Client 1): Samples 50-99 (50 samples)

**Class Distribution**: Stratified splits ensure balanced labels in each partition

**Reproducibility**: Seed propagated from server ensures consistent partitions across rounds

---

## Key Components

### 1. Server Application (`core/server_app.py`)

**File**: [core/server_app.py](core/server_app.py)

**Lifespan Management** (lines 40-68):
- `on_startup()`: Create database run entity
- `on_shutdown()`: Finalize run, update status
- Environment setup, configuration loading

**Main Logic** (lines 71-274):
- Initialize global model (LitResNet)
- Configure strategy with FedAvg and evaluation function
- Execute federated rounds via `strategy.start()`
- Persist results to database and JSON
- Send training completion signal

**Configuration** (lines 133-146):
- `num_server_rounds`: Total federated rounds
- `max_epochs`: Local epochs per client per round
- `num_partitions`: Total number of clients
- Applied to all client configurations

---

### 2. Client Application (`core/client_app.py`)

**File**: [core/client_app.py](core/client_app.py)

**Training** (lines 28-171):
- Load global model from server
- Get data partition by `context.node_id % num_partitions`
- Train locally for `max_epochs` epochs
- Return: Updated weights + training metrics

**Evaluation** (lines 174-269):
- Evaluate global model on validation set
- Return: Validation metrics (accuracy, precision, recall, f1, auroc)
- Include `num-examples` for weighted aggregation

**Data Pipeline**:
- Uses [utils.py::_prepare_partition_and_split()](core/utils.py#L53-L75) to load partition
- Creates XRayDataModule with local data
- Trains with identical LitResNet as centralized

---

### 3. Server Evaluation (`core/server_evaluation.py`)

**File**: [core/server_evaluation.py](core/server_evaluation.py)

**Purpose**: Centralized evaluation of aggregated model on server-held test set

**Process** (lines 38-173):
1. Receive aggregated model weights from FedAvg
2. Load model into LitResNet
3. Evaluate on server test set (last 20% of full dataset)
4. Compute metrics: loss, accuracy, precision, recall, f1, auroc
5. Extract confusion matrix: TP, TN, FP, FN
6. Return as MetricRecord

**Server Test Set**:
- Independent of client partitions
- Provides unbiased global performance estimate
- Used for monitoring convergence

---

### 4. Custom Strategy (`core/custom_strategy.py`)

**File**: [core/custom_strategy.py](core/custom_strategy.py)

**Class**: `ConfigurableFedAvg` (extends Flower's `FedAvg`)

**Features**:
- **configure_train()**: Pass custom config (paths, seeds) to clients
- **configure_evaluate()**: Pass model + evaluation config to clients
- **aggregate_evaluate()**: Weighted aggregation of client metrics
- **Metrics Broadcasting**: Send round metrics to WebSocket for frontend

**Weighted Aggregation Formula**:
```
aggregated_metric = Σ(client_metric_i × num_examples_i) / Σ(num_examples_i)
```

**Critical Convention**: Clients must include `num-examples` (with HYPHEN, not underscore) in MetricRecord for proper weighting.

---

### 5. Utilities (`core/utils.py`)

**File**: [core/utils.py](core/utils.py)

**Key Functions**:

| Function | Purpose | Reference |
|----------|---------|-----------|
| `_prepare_partition_and_split()` | Load and split client data partition | lines 53-75 |
| `_extract_metrics_from_result()` | Map various metric names to standard format | lines 127-166 |
| `_create_metric_record_dict()` | Create metric dict with num-examples (CRITICAL) | lines 169-186 |
| `_persist_server_evaluations()` | Save server metrics to database | lines 279-376 |
| `read_configs_to_toml()` | Sync YAML config to Flower pyproject.toml | lines 189-264 |

---

## Configuration

### Flower Configuration (`pyproject.toml`)

**File**: [pyproject.toml](pyproject.toml)

```toml
[tool.flwr.app.config]
num-server-rounds = 2
max-epochs = 2

[tool.flwr.federations.local-simulation.options]
num-supernodes = 2  # Number of clients
```

### YAML Configuration (`config/default_config.yaml`)

```yaml
experiment:
  num_rounds: 2
  num_clients: 2
  clients_per_round: 2
  local_epochs: 2
  num-server-rounds: 2
  max-epochs: 2
  options:
    num-supernodes: 2
```

**Synchronization**: `utils.py::read_configs_to_toml()` keeps values in sync

---

## Metrics Tracking

### Three Types of Metrics

| Type | Source | Aggregation | Storage |
|------|--------|-------------|---------|
| **Client Training** | Client local training | Per-client (not aggregated) | run_metrics |
| **Client Evaluation** | Client validation sets | Weighted (by num_examples) | run_metrics (aggregated) |
| **Server Evaluation** | Server test set | Not aggregated (global) | server_evaluations |

### Database Tables

- **run**: Training session metadata
- **client**: Federated participants (multiple per federated run, NULL for centralized)
- **round**: FL communication rounds per client
- **run_metrics**: Training metrics (supports both centralized and federated)
- **server_evaluations**: Global model performance per round

---

## Real-Time Monitoring

### WebSocket Streaming

**File**: [websocket_metrics_sender.py](../dl_model/utils/data/websocket_metrics_sender.py)

**Messages Sent**:
1. `training_mode`: Federated vs centralized + num_rounds, num_clients
2. `round_end`: Per-round aggregated metrics
3. `training_end`: Final status + run_id

**Frontend Reception**: [services/websocket.ts](../../../../../xray-vision-ai-forge/src/services/websocket.ts) receives and updates TrainingExecution component

---

## Execution

### Command Line

```bash
# Run federated simulation (2 clients, 2 rounds, local simulation)
uv run flwr run federated_pneumonia_detection/src/control/federated_new_version

# Via PowerShell script
./federated_pneumonia_detection/src/rf.ps1
```

### Via API

**POST /experiments/federated**:
```json
{
  "run_name": "federated_exp_01",
  "dataset_path": "/path/to/data.zip",
  "config_overrides": {
    "experiment.num_rounds": 3,
    "experiment.num_clients": 2
  }
}
```

---

## Integration with Centralized Training

**Shared Components**:
- LitResNet (model wrapper)
- XRayDataModule (data loading)
- MetricsCollector (metrics tracking)
- CustomImageDataset (image loading)
- image_transforms (augmentation)

**Key Difference**:
- Centralized: Full dataset on single machine
- Federated: Data distributed across clients, coordinated training

---

## Error Handling & Logging

- **Server Failures**: Graceful shutdown, partial results saved
- **Client Failures**: Flower handles offline clients, continues with online ones
- **Communication Errors**: Auto-retry with exponential backoff
- **Metrics Errors**: Logged but don't block training
- **Database Errors**: Logged with rollback support

---

## Related Documentation

- **Client Training Utils**: See [dl_model/utils/README.md](../dl_model/utils/README.md)
- **Model Architecture**: See [entities/README.md](../../entities/README.md)
- **Data Partitioning**: See [partioner.py](partioner.py)
- **Database Schema**: See [boundary/README.md](../../boundary/README.md)
- **API Integration**: See [api/README.md](../../api/README.md)
- **System Architecture**: See [README.md](../../../README.md)
