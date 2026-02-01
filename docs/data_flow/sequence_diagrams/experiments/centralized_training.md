# Centralized Training Flow

**API**: `POST /api/experiments/centralized/train`
**Entry Point**: `centralized_endpoints.py:21-80` → `centralized_tasks.py:11-66`

---

## Overview

Centralized training uses all training data in a single unified trainer (no data partitioning across clients). The endpoint accepts a ZIP archive containing medical images and metadata, extracts the data, and executes training in a background task.

---

## Step 1: Upload & Validation

**Files**:
- `centralized_endpoints.py` (lines 21-59)
- `file_handling.py` (lines 17-77)

```mermaid
sequenceDiagram
    participant Client as Frontend
    participant EP as Endpoint<br/>centralized_endpoints.py
    participant FH as FileHandler<br/>file_handling.py

    Client->>EP: POST /train<br/>(data_zip, params)
    Note over EP: Line 22-28<br/>Parse form data
    EP->>FH: prepare_zip(data_zip)
    Note over FH: Lines 34-48<br/>Create temp dir<br/>Save & extract ZIP
    FH->>FH: Handle nested directories
    Note over FH: Lines 50-67<br/>Detect Images/ & CSV
    FH-->>EP: source_path
```

**Key Code**:
```python
# centralized_endpoints.py lines 59-67
source_path = await prepare_zip(data_zip, logger, experiment_name)
background_tasks.add_task(
    run_centralized_training_task,
    source_path=source_path,
    checkpoint_dir=checkpoint_dir,
    logs_dir=logs_dir,
    experiment_name=experiment_name,
    csv_filename=csv_filename,
)
```

**Form Parameters**:
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `data_zip` | (required) | ZIP containing Images/ + metadata CSV |
| `checkpoint_dir` | `results/centralized/checkpoints` | Model checkpoint save location |
| `logs_dir` | `results/centralized/logs` | Training logs location |
| `experiment_name` | `pneumonia_centralized` | Experiment identifier |
| `csv_filename` | `stage2_train_metadata.csv` | Metadata file name |

---

## Step 2: Background Task Execution

**Files**:
- `centralized_tasks.py` (lines 11-66)
- `centralized_trainer.py` (lines TBD - Core training logic)

```mermaid
sequenceDiagram
    participant BG as BackgroundTask
    participant CT as CentralizedTrainer<br/>centralized_trainer.py
    participant Config as ConfigManager

    BG->>BG: run_centralized_training_task()
    Note over BG: Lines 31-44<br/>Initialize logger & trainer
    BG->>Config: Load default_config.yaml
    BG->>CT: CentralizedTrainer(config_path, dirs)
    Note over CT: Initialize model,<br/>optimizer, scheduler
    BG->>CT: train(source_path, experiment_name)
    Note over CT: Load dataset<br/>Train epochs<br/>Validate & checkpoint
    CT-->>BG: results (final_metrics)
    Note over BG: Lines 54-61<br/>Log final metrics
```

**Key Code**:
```python
# centralized_tasks.py lines 40-52
trainer = CentralizedTrainer(
    config_path=config_path,
    checkpoint_dir=checkpoint_dir,
    logs_dir=logs_dir,
)

results = trainer.train(
    source_path=source_path,
    experiment_name=experiment_name,
    csv_filename=csv_filename,
)
```

---

## Step 3: Response & Tracking

**Files**:
- `centralized_endpoints.py` (lines 69-75)

```mermaid
sequenceDiagram
    participant EP as Endpoint
    participant Client as Frontend
    participant FS as FileSystem

    EP->>Client: Return status
    Note over Client: {"message": "Training started",<br/>"status": "queued"}

    loop Monitor Progress
        Client->>FS: Poll logs_dir
        FS-->>Client: Log entries
        Client->>FS: Check checkpoint_dir
        FS-->>Client: Checkpoint files
    end
```

**Response Format**:
```json
{
  "message": "Centralized training started successfully",
  "experiment_name": "pneumonia_centralized",
  "checkpoint_dir": "results/centralized/checkpoints",
  "logs_dir": "results/centralized/logs",
  "status": "queued"
}
```

---

## Error Handling

**Files**:
- `centralized_endpoints.py` (lines 76-80)
- `centralized_tasks.py` (lines 63-65)

| Error Type | Handler Location | Response |
|------------|------------------|----------|
| File extraction failure | `file_handling.py:72-76` | Cleanup temp dir, re-raise |
| Training exception | `centralized_tasks.py:63-65` | `{"status": "failed", "error": str(e)}` |
| Endpoint exception | `centralized_endpoints.py:76-80` | Log error, cleanup temp dir, raise |

**Error Flow**:
```python
# centralized_tasks.py lines 63-65
except Exception as e:
    task_logger.error(f"Error: {type(e).__name__}: {str(e)}")
    return {"status": "failed", "error": str(e)}
```

---

## File Reference

| Layer | File | Key Lines | Purpose |
|-------|------|-----------|---------|
| **API** | `centralized_endpoints.py` | 21-80 | Endpoint definition, validation |
| **Utils** | `file_handling.py` | 17-77 | ZIP extraction & path handling |
| **Task** | `centralized_tasks.py` | 11-66 | Background training orchestration |
| **Core** | `centralized_trainer.py` | N/A | Model training logic |
| **Config** | `default_config.yaml` | N/A | Hyperparameters, paths |

---

## Configuration Dependencies

```python
# centralized_tasks.py line 39
config_path = r"federated_pneumonia_detection\config\default_config.yaml"
```

**Config Keys Used**:
- Model architecture settings
- Training hyperparameters (epochs, batch size, learning rate)
- Data augmentation parameters
- Early stopping criteria

---

## Monitoring Points

1. **Log Files**: `{logs_dir}/` - Training progress, loss curves, validation metrics
2. **Checkpoints**: `{checkpoint_dir}/` - Best model weights based on validation performance
3. **Return Value**: `results["final_metrics"]` - Final accuracy, loss, precision, recall

---

## Integration with Control Layer

```mermaid
graph TD
    A[API Layer] --> B[Background Task]
    B --> C[CentralizedTrainer]
    C --> D[ConfigManager]
    C --> E[DataLoader]
    C --> F[Model]
    C --> G[Checkpoint Manager]
    C --> H[Logger]
```

**Dependency Flow**:
- API → Task (orchestration)
- Task → Trainer (execution)
- Trainer → Config (settings)
- Trainer → Data/Model/Logging (resources)
