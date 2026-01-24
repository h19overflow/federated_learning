# Boundary CRUD Operations

This module provides the Data Access Object (DAO) layer for interacting with the database. It implements the CRUD (Create, Read, Update, Delete) pattern to abstract database logic from the rest of the application.

## Base CRUD (`base.py`)
The `BaseCRUD` class is a generic base class that provides standard operations for any SQLAlchemy model:
- `create`: Add a new record.
- `get`: Retrieve a record by ID.
- `get_multi`: Retrieve multiple records with optional filtering and pagination.
- `update`: Modify an existing record.
- `delete`: Remove a record.
- `count`: Get the total number of records.
- `exists`: Check if a record exists.
- `bulk_create`: Efficiently insert multiple records.

## Specialized CRUD Classes

### 1. RunCRUD (`run.py`)
Manages `Run` entities.
- **Key Operations**:
  - `persist_metrics`: Handles complex logic for saving metrics in both centralized and federated modes.
  - `complete_run`: Marks a run as finished and records the end time.
  - `get_by_status_and_mode`: Filtered retrieval of runs.
  - `batch_get_final_metrics`: Efficiently fetches final stats for multiple runs.

### 2. RunMetricCRUD (`run_metric.py`)
Manages `RunMetric` entities.
- **Key Operations**:
  - `get_best_metric`: Finds the maximum or minimum value for a specific metric.
  - `create_final_epoch_stats`: Persists summary statistics for the final epoch.
  - `get_metric_stats`: Calculates summary statistics (min, max, mean) for a metric.

### 3. ClientCRUD (`client.py`)
Manages `Client` entities for federated learning.
- **Key Operations**:
  - `create_client`: Registers a new participant.
  - `get_clients_by_run_id`: Retrieves all clients for a specific training session.

### 4. RoundCRUD (`round.py`)
Manages `Round` entities.
- **Key Operations**:
  - `get_or_create_round`: Thread-safe method to ensure a round exists before recording metrics.
  - `complete_round`: Marks a communication round as finished.

### 5. ServerEvaluationCRUD (`server_evaluation.py`)
Manages `ServerEvaluation` entities.
- **Key Operations**:
  - `create_evaluation`: Extracts and saves core metrics and confusion matrix data.
  - `get_summary_stats`: Aggregates evaluation results across all rounds.

### 6. Chat History (`chat_history.py`)
Provides functional CRUD for `ChatSession` entities.
- **Key Operations**: `create_chat_session`, `get_chat_session`, `get_all_chat_sessions`, `update_chat_session_title`, `delete_chat_session`.

### 7. Fetch Documents (`fetch_documents.py`)
Specialized utility for retrieving documents from the vector database (`langchain_pg_embedding` table) for the RAG pipeline.

## Mapping between Models and CRUD

Each CRUD class is typically associated with a specific SQLAlchemy model:

| CRUD Class | Model |
|------------|-------|
| `RunCRUD` | `Run` |
| `RunMetricCRUD` | `RunMetric` |
| `ClientCRUD` | `Client` |
| `RoundCRUD` | `Round` |
| `ServerEvaluationCRUD` | `ServerEvaluation` |
| `chat_history.py` | `ChatSession` |

The CRUD classes use these models to perform database operations, ensuring that the data structure remains consistent across the application.
