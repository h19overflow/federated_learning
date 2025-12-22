# Boundary Module - Data Access Layer

**Purpose**: Database operations, external service integration, and data persistence for the federated pneumonia detection system.

The boundary layer implements the **Data Access Object (DAO)** pattern, separating business logic from database operations and providing a clean interface for data persistence.

## Table of Contents
- [Overview](#overview)
- [Database Schema](#database-schema)
- [CRUD Operations](#crud-operations)
- [Module Structure](#module-structure)
- [Integration Points](#integration-points)

---

## Overview

The boundary module is responsible for:
1. **Database Configuration**: Connection management and engine setup
2. **Data Models**: ORM models defining database tables and relationships
3. **CRUD Operations**: Create, Read, Update, Delete operations via specialized classes
4. **External Services**: WandB integration (placeholder)
5. **Vector Database**: Queries for RAG pipeline

**Technology Stack**:
- **ORM**: SQLAlchemy 2.x
- **Database**: PostgreSQL
- **Connection**: Pydantic environment-based configuration

---

## Database Schema

### Entity Relationship Diagram

```
Run (training sessions)
├── Client (FL participants)
│   └── Round (FL communication rounds)
│       └── RunMetric (metrics per round per client)
├── RunMetric (centralized metrics per epoch)
└── ServerEvaluation (global model evaluation per round)
```

### Data Models

#### 1. `Run` Table
**File**: [engine.py:21-36](engine.py#L21-L36)

**Purpose**: Represents a complete training execution (centralized or federated).

**Fields**:
| Field | Type | Purpose |
|-------|------|---------|
| `id` | Integer (PK) | Unique run identifier |
| `run_description` | String(1024) | Training description/notes |
| `training_mode` | String(50) | 'centralized' or 'federated' |
| `status` | String(50) | Status: 'in_progress', 'completed', 'failed' |
| `start_time` | TIMESTAMP | Run start time |
| `end_time` | TIMESTAMP | Run completion time (NULL if in progress) |
| `wandb_id` | String(255) | Weights & Biases integration ID |
| `source_path` | String(1024) | Dataset source path |

**Relationships**:
- 1 Run → Many RunMetric
- 1 Run → Many Client (federated only)
- 1 Run → Many ServerEvaluation (federated only)

#### 2. `Client` Table
**File**: [engine.py:38-55](engine.py#L38-L55)

**Purpose**: Represents a federated learning participant (NULL for centralized).

**Fields**:
| Field | Type | Purpose |
|-------|------|---------|
| `id` | Integer (PK) | Unique client identifier |
| `run_id` | Integer (FK) | Parent run ID |
| `client_identifier` | String(255) | Client label (e.g., 'client_0', 'client_1') |
| `created_at` | TIMESTAMP | Client registration time |
| `client_config` | JSON | Client-specific configuration (optional) |

**Relationships**:
- Many Client → 1 Run
- 1 Client → Many Round

#### 3. `Round` Table
**File**: [engine.py:57-70](engine.py#L57-L70)

**Purpose**: Tracks federated learning communication rounds per client.

**Fields**:
| Field | Type | Purpose |
|-------|------|---------|
| `id` | Integer (PK) | Unique round identifier |
| `client_id` | Integer (FK) | Client participating in round |
| `round_number` | Integer | Round number (1, 2, 3, ...) |
| `start_time` | TIMESTAMP | Round start time |
| `end_time` | TIMESTAMP | Round completion time |
| `round_metadata` | JSON | Flexible metadata (aggregation strategy, weights, etc.) |

**Relationships**:
- Many Round → 1 Client

#### 4. `RunMetric` Table
**File**: [engine.py:72-99](engine.py#L72-L99)

**Purpose**: Stores training metrics (loss, accuracy, precision, etc.) per epoch/round.

**Fields**:
| Field | Type | Purpose |
|-------|------|---------|
| `id` | Integer (PK) | Unique metric record ID |
| `run_id` | Integer (FK) | Parent run ID (always required) |
| `client_id` | Integer (FK) | Client ID (NULL for centralized) |
| `round_id` | Integer (FK) | Round ID (NULL for centralized) |
| `metric_name` | String(255) | Metric name (e.g., 'val_loss', 'train_accuracy') |
| `metric_value` | Float | Metric value |
| `step` | Integer | Epoch or step number |
| `timestamp` | TIMESTAMP | When metric was recorded |

**Relationships**:
- Many RunMetric → 1 Run
- Many RunMetric → 1 Client (optional, federated only)
- Many RunMetric → 1 Round (optional, federated only)

#### 5. `ServerEvaluation` Table
**File**: [engine.py:100-133](engine.py#L100-L133)

**Purpose**: Centralized server-side evaluation results per federated round.

**Fields**:
| Field | Type | Purpose |
|-------|------|---------|
| `id` | Integer (PK) | Unique evaluation ID |
| `run_id` | Integer (FK) | Parent federated run |
| `round_number` | Integer | Federated round number |
| `loss` | Float | Validation loss |
| `accuracy` | Float | Classification accuracy |
| `precision` | Float | Precision score |
| `recall` | Float | Recall score |
| `f1_score` | Float | F1 score |
| `auroc` | Float | AUROC score |
| `num_samples` | Integer | Samples in evaluation set |
| `evaluation_time` | Float | Evaluation time (seconds) |
| `true_positives` | Integer | Confusion matrix TP |
| `true_negatives` | Integer | Confusion matrix TN |
| `false_positives` | Integer | Confusion matrix FP |
| `false_negatives` | Integer | Confusion matrix FN |
| `additional_metrics` | JSON | Additional metrics in JSON format |

**Relationships**:
- Many ServerEvaluation → 1 Run

---

## CRUD Operations

### Module Structure

```
boundary/
├── engine.py              # Database configuration and models
├── CRUD/                  # Data access objects
│   ├── __init__.py
│   ├── base.py            # BaseCRUD abstract class
│   ├── run.py             # Run CRUD operations
│   ├── client.py          # Client CRUD operations
│   ├── round.py           # Round CRUD operations
│   ├── run_metric.py      # RunMetric CRUD operations
│   ├── server_evaluation.py # ServerEvaluation CRUD operations
│   └── fetch_documents.py # Vector DB queries (RAG)
└── wandb_Integration/     # WandB integration (placeholder)
```

### CRUD Classes

#### 1. `RunCRUD`
**File**: [CRUD/run.py](CRUD/run.py)

**Key Methods**:
| Method | Purpose |
|--------|---------|
| `create_run()` | Create new run with initial status |
| `get_run()` | Retrieve run by ID |
| `get_all_runs()` | List all runs with optional filtering |
| `update_run_status()` | Update run status and end_time |
| `get_latest_run()` | Get most recent run |
| `persist_metrics()` | Save epoch metrics with federated context |

#### 2. `ClientCRUD`
**File**: [CRUD/client.py](CRUD/client.py)

**Key Methods**:
| Method | Purpose |
|--------|---------|
| `create_client()` | Register new federated client |
| `get_client()` | Retrieve client by ID |
| `get_clients_for_run()` | List all clients in a run |

#### 3. `RoundCRUD`
**File**: [CRUD/round.py](CRUD/round.py)

**Key Methods**:
| Method | Purpose |
|--------|---------|
| `create_round()` | Create new federated round record |
| `get_round()` | Retrieve round by ID |
| `get_rounds_for_client()` | List rounds for client |
| `update_round_end_time()` | Mark round as complete |

#### 4. `RunMetricCRUD`
**File**: [CRUD/run_metric.py](CRUD/run_metric.py)

**Key Methods**:
| Method | Purpose |
|--------|---------|
| `create_metric()` | Insert single metric record |
| `get_by_run()` | Retrieve all metrics for run |
| `get_by_metric_name()` | Filter by metric type |
| `get_best_metric()` | Find best value for metric |
| `get_metric_stats()` | Compute statistics for run |

#### 5. `ServerEvaluationCRUD`
**File**: [CRUD/server_evaluation.py](CRUD/server_evaluation.py)

**Key Methods**:
| Method | Purpose |
|--------|---------|
| `create_evaluation()` | Insert server evaluation record |
| `get_by_run()` | Retrieve all evaluations for run |
| `get_summary_stats()` | Compute best metrics across rounds |
| `get_best_by_metric()` | Find best round for metric |

**Features**:
- Confusion matrix extraction and storage
- Weighted metric computation
- Support for both centralized and federated evaluations

---

## Data Flow Examples

### Centralized Training

```
CentralizedTrainer
  ↓
MetricsCollectorCallback (fires on_train_epoch_end)
  ↓
RunCRUD.persist_metrics()
  ↓
Insert into run_metrics:
  - run_id: 42
  - client_id: NULL
  - round_id: NULL
  - metric_name: 'train_loss'
  - metric_value: 0.245
  - step: 5
```

### Federated Training

```
FederatedServer (round 1)
  ├─ Client 0: train_metrics
  ├─ Client 1: train_metrics
  └─ Server: evaluate() → ServerEvaluation
      ↓
Insert into:
  - run_metrics (per-client training metrics)
  - server_evaluations (global model evaluation)
  ├─ run_id: 42
  ├─ round_number: 1
  ├─ loss: 0.312
  ├─ accuracy: 0.876
  ├─ true_positives: 245
  ├─ true_negatives: 189
  ├─ false_positives: 23
  ├─ false_negatives: 17
  └─ additional_metrics: {confusion_matrix: {...}}
```

---

## Integration Points

### With Control Layer

| Component | Usage | Reference |
|-----------|-------|-----------|
| **MetricsCollector** | Calls RunCRUD.persist_metrics() | [metrics_collector.py:425-480](../control/dl_model/utils/model/metrics_collector.py#L425-L480) |
| **Server App** | Creates Run, calls ServerEvalCRUD | [server_app.py:92-112, 249-257](../control/federated_new_version/core/server_app.py#L92-L112) |
| **Utils (Federated)** | Persists server evaluations | [utils.py:279-376](../control/federated_new_version/core/utils.py#L279-L376) |

### With API Layer

| Component | Usage | Reference |
|-----------|-------|-----------|
| **runs_metrics.py** | Fetches RunMetric for display | [runs_metrics.py](../api/endpoints/runs_endpoints/runs_metrics.py) |
| **runs_server_evaluation.py** | Fetches ServerEvaluation | [runs_server_evaluation.py](../api/endpoints/runs_endpoints/runs_server_evaluation.py) |

---

## Key Design Patterns

### Base CRUD Class
**File**: [CRUD/base.py](CRUD/base.py)

Abstract base class providing:
- Generic CRUD template methods
- Session management
- Error handling with rollback
- Logging

**Pattern**:
```python
class BaseCRUD(Generic[T]):
    def __init__(self, session: Session, model: Type[T]):
        self.session = session
        self.model = model

    def create(self, **kwargs) -> T:
        instance = self.model(**kwargs)
        self.session.add(instance)
        self.session.commit()
        return instance
```

### Separation of Concerns
- **engine.py**: Only models and configuration
- **CRUD/*.py**: Only data access logic
- No business logic in boundary layer

### Error Handling
- Specific exception types
- Transaction rollback on failure
- Comprehensive logging

---

## Configuration

**Database Connection** (via environment variables):
- `POSTGRES_HOST`: Database host
- `POSTGRES_PORT`: Database port
- `POSTGRES_USER`: Database user
- `POSTGRES_PASSWORD`: Database password
- `POSTGRES_DB`: Database name
- Full URI: `POSTGRES_DB_URI`

**Setup**:
```python
from federated_pneumonia_detection.src.boundary.engine import Base, get_session

# Create tables
Base.metadata.create_all(bind=engine)

# Get session for CRUD operations
session = get_session()
```

---

## Related Documentation

- **Data Models**: See schema definitions in [engine.py](engine.py)
- **Control Layer Integration**: See [control/dl_model/utils/README.md](../control/dl_model/utils/README.md)
- **API Integration**: See [api/README.md](../api/README.md)
- **Configuration**: See [config/README.md](../../config/README.md)
- **System Overview**: See [README.md](../../README.md)
