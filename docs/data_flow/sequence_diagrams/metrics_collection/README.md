# Metrics Collection & Database Persistence Documentation

**Purpose**: Comprehensive technical documentation for metrics collection, database persistence, and real-time broadcasting across centralized and federated training modes.

---

## Table of Contents

1. **[Centralized Training Flow](01_centralized_training_flow.md)**
   - API entry point through background task
   - Training loop with callback orchestration
   - Database persistence at training completion
   - Results collection and final stats calculation

2. **[Federated Client Flow](02_federated_client_flow.md)**
   - Client initialization and partition loading
   - Local training with global model initialization
   - Metrics collection with federated context (client_id, round_number)
   - Model updates and metrics return to server

3. **[Federated Server Flow](03_federated_server_flow.md)**
   - Server initialization and run creation
   - Central evaluation function setup
   - FedAvg strategy and round orchestration
   - Metrics aggregation and persistence
   - Training completion with final stats

4. **[Database Persistence Patterns](04_database_persistence_patterns.md)**
   - Bulk metric persistence
   - Eager loading to prevent N+1 queries
   - Federated context resolution
   - Querying metrics by context (client, round, dataset type)
   - Final epoch stats calculation
   - Session management and error handling

---

## Quick Reference

### Entry Points

| Training Mode | File | Function | Lines |
|---------------|------|----------|-------|
| **Centralized** | `centralized_tasks.py` | `run_centralized_training_task()` | 11-65 |
| **Federated Client** | `client_app.py` | `@app.train(msg, context)` | 36-179 |
| **Federated Server** | `server_app.py` | `@app.main(grid, context)` | 55-144 |

### Core Components

| Component | Purpose | File | Lines |
|-----------|---------|------|-------|
| **MetricsCollectorCallback** | Aggregate epoch metrics | `metrics.py` | 20-510 |
| **BatchMetricsCallback** | Sample batch-level metrics | `batch_metrics.py` | 13-145 |
| **GradientMonitorCallback** | Track gradient norms | `gradient_monitor.py` | 12-178 |
| **WebSocketSender** | Real-time frontend broadcast | `websocket_metrics_sender.py` | 17-106 |
| **MetricsFilePersister** | CSV/JSON export | `metrics_file_persister.py` | 9-54 |
| **RunCRUD** | Database operations for runs | `run.py` | 18-539 |
| **RunMetricCRUD** | Database operations for metrics | `run_metric.py` | 12-278 |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                   Training Orchestration                      │
│  (CentralizedTrainer / Federated Client/Server)              │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────┐
│              PyTorch Lightning Trainer + Callbacks            │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Callback Chain (8 callbacks)                        │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ 1. ModelCheckpoint - Save best models               │    │
│  │ 2. EarlyStopping - Stop if no improvement           │    │
│  │ 3. MetricsCollectorCallback - Aggregate metrics     │    │
│  │ 4. BatchMetricsCallback - Sample batch metrics      │    │
│  │ 5. GradientMonitorCallback - Track gradients        │    │
│  │ 6. EarlyStoppingSignal - Send stop events           │    │
│  │ 7. LearningRateMonitor - Track LR changes           │    │
│  │ 8. HighestValRecall - Track best recall             │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────┐
│                   Metrics Persistence (3 Channels)            │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Database   │  │  File Export │  │  WebSocket   │       │
│  │              │  │              │  │              │       │
│  │ RunMetric    │  │ CSV + JSON   │  │ Real-time    │       │
│  │ table        │  │ files        │  │ broadcast    │       │
│  │              │  │              │  │              │       │
│  │ Bulk insert  │  │ On fit_end() │  │ Per epoch/   │       │
│  │ on fit_end() │  │              │  │ batch        │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

---

## Metrics Collection Flow (High-Level)

### Centralized Training

```
API Request
  ↓
Background Task (centralized_tasks.py)
  ↓
Create Run (db_operations.py)
  ↓
Initialize Trainer + Callbacks (training_callbacks.py)
  ↓
Training Loop
  ├─ on_train_start → Create run, send "training_start" WS event
  ├─ on_train_epoch_end → Extract train metrics
  ├─ on_validation_epoch_end → Extract val metrics, send "epoch_end" WS event
  ├─ on_train_batch_end (sampled) → Send "batch_metrics" WS event
  └─ on_before_optimizer_step (sampled) → Send "gradient_stats" WS event
  ↓
on_fit_end
  ├─ Save metrics to CSV/JSON (MetricsFilePersister)
  ├─ Persist metrics to DB (RunCRUD.persist_metrics → bulk_create)
  └─ Send "training_end" WS event
  ↓
Collect Results (results.py)
  ↓
Complete Run (db_operations.py → RunCRUD.complete_run)
  ├─ Set status="completed", end_time=NOW()
  └─ Calculate final confusion matrix stats
```

### Federated Client Training

```
Flower Framework → ClientApp.train(msg, context)
  ↓
Extract client_id, round_number from context
  ↓
Load Partition (IidPartitioner)
  ↓
Create XRayDataModule (train_df, val_df)
  ↓
Initialize Model + Callbacks (federated_mode=True)
  ├─ _ensure_run_exists() → Get run_id from server config
  ├─ _ensure_client_exists() → Create Client entity in DB
  └─ Set federated_context = {client_id, round_number}
  ↓
Load Global Model (msg.content["arrays"].to_torch_state_dict())
  ↓
Local Training Loop
  ├─ on_train_start → Send "training_start" WS event (with client_id)
  ├─ on_train_epoch_end → Extract train metrics
  ├─ on_validation_epoch_end → Extract val metrics, send "epoch_end" WS event (with client_id, round_number)
  ├─ on_train_batch_end (sampled) → Send "batch_metrics" WS event (with client_id)
  └─ on_before_optimizer_step (sampled) → Send "gradient_stats" WS event
  ↓
on_fit_end
  ├─ Persist metrics to DB with federated_context (client_id, round_id)
  └─ NO "training_end" WS event (server sends it)
  ↓
Collect Results (filter metrics + add num_examples)
  ↓
Return to Server
  ├─ ArrayRecord(model.state_dict()) → Updated weights
  └─ MetricRecord(metrics_history) → Client metrics
```

### Federated Server Orchestration

```
Flower Framework → ServerApp.main(grid, context)
  ↓
Initialize Database Run (training_mode="federated")
  ↓
Setup Config Manager + Build Global Model
  ↓
Create Central Evaluation Function (server_evaluation.py)
  ↓
Initialize Strategy (FederatedCustomStrategy)
  ↓
Send "training_start" WS event
  ↓
For each round (1 to num_rounds):
  ├─ Send global model to clients
  ├─ Clients train locally → return (weights, metrics)
  ├─ Aggregate fit results (FedAvg weighted by num_examples)
  ├─ Central evaluation on server test set (central_evaluate_fn)
  ├─ Persist round metrics (Round table + RunMetric table)
  └─ Send "round_metrics" WS event
  ↓
Training Complete
  ├─ Collect all metrics (train, eval_client, eval_server)
  ├─ Save to results_{run_id}.json
  ├─ Persist server evaluations to DB (context="aggregated")
  ├─ Complete run (status="completed", end_time=NOW())
  ├─ Calculate final stats (best_epoch, best_val_recall, training_duration)
  └─ Send "training_end" WS event
```

---

## Database Schema (Key Tables)

### Run Table
| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `experiment_name` | String | User-defined name |
| `training_mode` | String | "centralized" or "federated" |
| `status` | String | "in_progress", "completed", "failed" |
| `start_time` | DateTime | Training start timestamp |
| `end_time` | DateTime | Training end timestamp |
| `source_path` | String | Dataset path |

### RunMetric Table
| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `run_id` | Integer | FK to Run |
| `metric_name` | String | e.g., "train_loss", "val_accuracy" |
| `metric_value` | Float | Actual metric value |
| `step` | Integer | Epoch/round number |
| `dataset_type` | String | "train", "validation", "test", "other" |
| `context` | String | "epoch_end", "batch", "aggregated", "final_epoch" |
| `client_id` | Integer | FK to Client (federated only) |
| `round_id` | Integer | FK to Round (federated only) |

### Client Table (Federated)
| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `run_id` | Integer | FK to Run |
| `client_id` | Integer | Flower node_id |
| `client_name` | String | e.g., "client_5" |

### Round Table (Federated)
| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `run_id` | Integer | FK to Run |
| `round_number` | Integer | 1 to num_rounds |
| `server_metrics` | JSON | Central evaluation results |
| `aggregated_client_metrics` | JSON | Weighted average of client metrics |

---

## WebSocket Event Types

| Event Type | Sent By | Payload | Frequency |
|------------|---------|---------|-----------|
| `training_start` | Client/Server | {run_id, experiment_name, max_epochs, training_mode} | Once at start |
| `epoch_end` | Client | {epoch, phase, metrics, client_id?, round_number?} | Per epoch |
| `batch_metrics` | Client | {step, batch_idx, loss, accuracy, recall, f1, client_id?} | Every Nth batch |
| `gradient_stats` | Client | {step, total_norm, layer_norms, max_norm, min_norm} | Every Nth step |
| `round_metrics` | Server | {round, total_rounds, fit_metrics, eval_metrics} | Per round (federated) |
| `training_end` | Client/Server | {run_id, status, best_epoch, best_val_recall, training_duration} | Once at end |
| `early_stopping` | Client | {epoch, best_metric_value, patience} | On early stop |

---

## Key Design Patterns

| Pattern | Purpose | Implementation |
|---------|---------|----------------|
| **Bulk Persistence** | Minimize DB transactions | `RunCRUD.persist_metrics()` → `bulk_create()` |
| **Eager Loading** | Prevent N+1 queries | `selectinload(Run.metrics)` |
| **Federated Context Propagation** | Link metrics to client/round | `federated_context={client_id, round_number}` |
| **Metric Transformation** | Convert epoch dict to DB rows | `_transform_epoch_to_metrics()` |
| **Session Management** | Ensure cleanup | `@contextmanager get_db_session()` |
| **Callback Chain** | Separation of concerns | 8 specialized callbacks |
| **Real-Time Broadcasting** | Frontend updates | WebSocketSender + async |

---

## File Organization

```
docs/data_flow/sequence_diagrams/metrics_collection/
├── README.md (this file)
├── 01_centralized_training_flow.md
├── 02_federated_client_flow.md
├── 03_federated_server_flow.md
└── 04_database_persistence_patterns.md

federated_pneumonia_detection/src/
├── api/endpoints/experiments/utils/
│   └── centralized_tasks.py (API entry)
├── control/
│   ├── dl_model/
│   │   ├── centralized_trainer.py (orchestrator)
│   │   ├── centralized_trainer_utils/
│   │   │   ├── db_operations.py (run CRUD wrappers)
│   │   │   └── results.py (collect results)
│   │   └── internals/
│   │       ├── model/
│   │       │   ├── training_callbacks.py (callback factory)
│   │       │   ├── collectors/metrics.py (MetricsCollectorCallback)
│   │       │   └── callbacks/
│   │       │       ├── batch_metrics.py
│   │       │       └── gradient_monitor.py
│   │       └── data/
│   │           ├── websocket_metrics_sender.py
│   │           └── metrics_file_persister.py
│   └── federated_new_version/core/
│       ├── client_app.py (client training)
│       ├── server_app.py (server orchestration)
│       ├── custom_strategy.py (FedAvg)
│       └── server_evaluation.py (central eval)
└── boundary/CRUD/
    ├── base.py (BaseCRUD)
    ├── run.py (RunCRUD)
    └── run_metric.py (RunMetricCRUD)
```

---

## Common Queries

### Get all metrics for a run
```python
from boundary.CRUD.run import run_crud
run = run_crud.get_with_metrics(db, run_id=123)
for metric in run.metrics:
    print(f"{metric.metric_name}: {metric.metric_value} @ step {metric.step}")
```

### Get only validation metrics
```python
from boundary.CRUD.run_metric import run_metric_crud
val_metrics = run_metric_crud.get_by_dataset_type(db, run_id=123, dataset_type="validation")
```

### Get client-specific metrics (federated)
```python
grouped_metrics = run_metric_crud.get_by_run_grouped_by_client(db, run_id=123)
for client_id, metrics in grouped_metrics.items():
    print(f"Client {client_id}: {len(metrics)} metrics")
```

### Get server aggregated metrics (federated)
```python
server_metrics = run_metric_crud.get_aggregated_metrics_by_run(db, run_id=123)
```

---

## Related Documentation

- **Control Layer CLAUDE.md**: `federated_pneumonia_detection/src/control/CLAUDE.md`
- **API Layer CLAUDE.md**: `federated_pneumonia_detection/src/api/CLAUDE.md`
- **Global Instructions**: `~/.claude/CLAUDE.md`

---

## Contributors

- **MetricsCollectorCallback**: Core metrics aggregation across all training modes
- **BatchMetricsCallback**: Real-time batch-level monitoring
- **GradientMonitorCallback**: Gradient flow tracking for debugging
- **RunCRUD**: Database operations for runs and metrics persistence
- **WebSocketSender**: Real-time frontend updates

---

**Last Updated**: 2026-02-01
**Documentation Pattern**: tech-doc-principal (step-based, file references, mermaid diagrams)
