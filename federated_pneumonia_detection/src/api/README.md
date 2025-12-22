# API Layer - FastAPI Endpoints

**Purpose**: RESTful API interface for training orchestration, results retrieval, and system configuration management.

The API layer provides the bridge between the React frontend and backend training systems, enabling real-time experiment management and results visualization.

## Overview

- **Framework**: FastAPI (async Python web framework)
- **Server**: Uvicorn ASGI server
- **Port**: 127.0.0.1:8001
- **WebSocket**: ws://localhost:8765 (embedded server for real-time metrics)

---

## Endpoint Groups

### 1. Experiments - Training Orchestration

**Base**: `/experiments/`

#### Centralized Training

**POST** `/centralized`

**File**: [endpoints/experiments/centralized_endpoints.py](endpoints/experiments/centralized_endpoints.py)

**Request**:
```json
{
  "run_name": "experiment_01",
  "dataset_path": "/path/to/data.zip",
  "config_overrides": {
    "experiment.epochs": 15,
    "experiment.learning_rate": 0.0005
  }
}
```

**Response**: `{ "run_id": 42, "status": "training" }`

**Process**:
- Validate request via schemas.py
- Dispatch to background task (centralized_tasks.py)
- Initialize CentralizedTrainer
- Return immediately with run_id (async training)
- Metrics streamed via WebSocket

**Background Task** (centralized_tasks.py):
- Extract dataset
- Prepare data
- Execute training
- Persist results

#### Federated Training

**POST** `/federated`

**File**: [endpoints/experiments/federated_endpoints.py](endpoints/experiments/federated_endpoints.py)

**Request**:
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

**Response**: `{ "run_id": 43, "status": "federated_training" }`

**Process**:
- Trigger Flower federated simulation
- Manage server and client processes
- Monitor round completion
- Persist metrics per round

### 2. Runs - Results Management

**Base**: `/runs/`

**File**: [endpoints/runs_endpoints/](endpoints/runs_endpoints/)

#### Get Run Metrics

**GET** `/{run_id}/metrics`

**File**: [runs_metrics.py](endpoints/runs_endpoints/runs_metrics.py)

**Response** (Centralized):
```json
{
  "run_id": 42,
  "training_mode": "centralized",
  "status": "completed",
  "training_history": [
    {"epoch": 1, "train_loss": 0.523, "val_accuracy": 0.845, ...},
    ...
  ],
  "best_metrics": {
    "accuracy": 0.891,
    "precision": 0.856,
    "recall": 0.923,
    "f1": 0.889
  },
  "confusion_matrix": {
    "true_positives": 245,
    "true_negatives": 189,
    "false_positives": 23,
    "false_negatives": 17
  }
}
```

**Data Source**: Queries run_metrics table, transforms via utils.py::_transform_run_to_results()

#### Get Federated Round Metrics

**GET** `/{run_id}/federated-rounds`

**File**: [runs_federated_rounds.py](endpoints/runs_endpoints/runs_federated_rounds.py)

**Response**:
```json
{
  "rounds": [
    {
      "round": 1,
      "client_metrics": { "accuracy": 0.821, "loss": 0.312, ... },
      "timestamp": "2024-01-15T10:30:00Z"
    },
    ...
  ]
}
```

#### Get Server Evaluation

**GET** `/{run_id}/server-evaluation`

**File**: [runs_server_evaluation.py](endpoints/runs_endpoints/runs_server_evaluation.py)

**Response** (Federated only):
```json
{
  "evaluations": [
    {
      "round": 1,
      "loss": 0.267,
      "accuracy": 0.878,
      "precision": 0.852,
      "recall": 0.931,
      "f1": 0.889,
      "auroc": 0.945,
      "confusion_matrix": { "tp": 245, "tn": 189, "fp": 23, "fn": 17 },
      "timestamp": "2024-01-15T10:35:00Z"
    },
    ...
  ]
}
```

**Data Source**: Queries ServerEvaluation table

#### List Runs

**GET** `/`

**File**: [runs_list.py](endpoints/runs_endpoints/runs_list.py)

**Response**:
```json
{
  "runs": [
    {"run_id": 42, "training_mode": "centralized", "status": "completed", "created_at": "..."},
    {"run_id": 43, "training_mode": "federated", "status": "in_progress", "created_at": "..."}
  ]
}
```

#### Download Run Results

**POST** `/{run_id}/download`

**File**: [runs_download.py](endpoints/runs_endpoints/runs_download.py)

**Request**:
```json
{
  "format": "csv"  // or "json" or "summary"
}
```

**Response**: File download (metrics exported from database/files)

### 3. Configuration Settings

**Base**: `/configuration/`

**File**: [endpoints/configuration_settings/](endpoints/configuration_settings/)

#### Get Current Configuration

**GET** `/current`

**File**: [configuration_endpoints.py](endpoints/configuration_settings/configuration_endpoints.py)

**Response**:
```json
{
  "system": { "batch_size": 32, "img_size": [256, 256], ... },
  "experiment": { "learning_rate": 0.001, "epochs": 10, ... },
  "paths": { "base_path": ".", "images_subfolder": "Images", ... }
}
```

#### Update Configuration

**POST** `/update`

**Request**:
```json
{
  "experiment": {
    "learning_rate": 0.0005,
    "epochs": 15
  }
}
```

**Response**: `{ "updated_fields": 2, "verification_passed": true }`

**Process**:
- Validate via Pydantic schemas
- Update ConfigManager
- Verify changes
- Save to YAML

### 4. Chat & Documentation

**Base**: `/chat/`

**File**: [endpoints/chat/](endpoints/chat/)

**POST** `/query`

Integration with MCP-based arXiv agent for research paper queries related to federated learning.

---

## Request/Response Schemas

**File**: [endpoints/configuration_settings/schemas.py](endpoints/configuration_settings/schemas.py)

**Key Schemas**:
- `CentralizedTrainingRequest`: Centralized experiment parameters
- `FederatedTrainingRequest`: Federated experiment parameters
- `ExperimentResults`: Training results format
- `RunMetrics`: Per-epoch metric structure
- `ConfusionMatrix`: 2x2 confusion matrix
- `ServerEvaluation`: Server-side evaluation metrics

---

## Dependency Injection

**File**: [deps.py](deps.py)

**Key Dependencies**:
- `get_config()`: Provides ConfigManager instance
- `get_session()`: Database session for queries
- `get_run_crud()`: RunCRUD instance for queries

---

## Error Handling

**Response Codes**:
- `200`: Successful request
- `400`: Invalid request parameters
- `404`: Run/resource not found
- `422`: Validation error
- `500`: Server error (logged)

**Error Response**:
```json
{
  "detail": "Description of what went wrong",
  "error_code": "RUN_NOT_FOUND"
}
```

---

## WebSocket Real-Time Metrics

**URL**: `ws://localhost:8765`

**Embedded Server** (api/main.py:106-193):
- Auto-starts with FastAPI lifespan
- Broadcasts training progress
- No manual connection required

**Message Types**:

| Type | Payload | Sent When |
|------|---------|-----------|
| `training_start` | `run_id`, `training_mode`, `num_rounds` | Training begins |
| `epoch_end` | `run_id`, `epoch`, `metrics` | Epoch completes |
| `round_end` | `run_id`, `round`, `aggregated_metrics` | FL round completes |
| `training_end` | `run_id`, `status`, `summary` | Training finished |
| `error` | `run_id`, `error_message` | Failure occurs |

**Frontend Integration**: [services/websocket.ts](../../../xray-vision-ai-forge/src/services/websocket.ts)

---

## CORS & Security

**CORS Configuration** (api/main.py):
- Allow localhost frontend access
- Development-only settings (update for production)

**API Key** (Optional): Can be added via FastAPI dependencies

---

## Main Entry Point

**File**: [main.py](main.py)

**FastAPI App Creation**:
```python
app = FastAPI(title="Federated Pneumonia Detection API")

# WebSocket auto-starts via lifespan
# Routes registered from endpoints
```

**Startup**:
```bash
uvicorn federated_pneumonia_detection.src.api.main:app --reload --port 8001
```

---

## Integration Flow

```
React Frontend
      ↓
    HTTPS/WebSocket
      ↓
FastAPI API Layer
  ├─ Request validation (Pydantic)
  ├─ Dependency injection
  ├─ Route handling
  └─ Error handling
      ↓
Control Layer (Training)
  ├─ CentralizedTrainer
  └─ Federated processes
      ↓
Boundary Layer (CRUD)
  ├─ Database persistence
  └─ Metrics retrieval
      ↓
Frontend (Results Display)
```

---

## Related Documentation

- **Control Layer**: See [control/README.md](../control/README.md)
- **Database**: See [boundary/README.md](../boundary/README.md)
- **Configuration**: See [config/README.md](../../config/README.md)
- **System Architecture**: See [README.md](../../../README.md)
